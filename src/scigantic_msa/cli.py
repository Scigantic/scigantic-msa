"""scigantic-msa command line.

    scigantic-msa run SEQ_OR_FASTA --model boltz --out DIR [--server colabfold|nvidia] [--yes]
    scigantic-msa run SEQ_OR_FASTA --model boltz --out DIR --open-only
    scigantic-msa convert ALIGNMENT.a3m [MORE.a3m ...] --model af3 --out DIR [--uniprot U.sto ...]
    scigantic-msa cache     (prints the cache directory)
    scigantic-msa version   (or --version)

``run`` merges the published open alignments of the sequence with a search
on the ColabFold server, and asks once before it sends the sequence there
(``--yes`` skips the question). ``--open-only`` uses open data alone and
sends the sequence nowhere. ``convert`` reads alignments you already have
and sends nothing.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .a3m import A3MError
from .adapters import PrepareError
from .alignment import Alignment
from .cache import cache_dir
from .remote import BACKENDS, remote_disabled
from .remote.colabfold import JobFailed, RemoteError
from .seqkey import InvalidSequence, canonical_sequence
from .tool import MSA, RemoteNotAllowed, _is_default_host, read_alignment
from .write import MODELS, _fname, check_model, write_model



def die(msg: str, code: int = 1) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


#: A sequence typed on the command line shorter than this is refused unless
#: it starts with '=', because it is more likely a mistyped file name.
MIN_LITERAL = 10
#: Most chains in one complex, and most jobs in one --batch run.
MAX_CHAINS = 20
MAX_JOBS = 100
#: Most new sequences one run sends to the public server, which takes about
#: one new job a minute for everyone.
MAX_PUBLIC_SEARCHES = 100
#: Below this many sequences an alignment is called shallow.
LOW_DEPTH = 20


class InputError(ValueError):
    """The run argument cannot be used as a sequence or a FASTA file."""


def _read_text(arg: str) -> Tuple[str, bool]:
    """(text, typed): typed is True when the argument itself is the
    sequence and was not marked with a leading '='."""
    if arg == "-":
        return _decode(sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else sys.stdin.read(), "stdin"), False
    if arg.startswith("="):
        return arg[1:], False
    if os.path.isdir(arg):
        raise InputError(f"{arg} is a directory. Pass a FASTA file, a sequence, or - for stdin.")
    if os.path.isfile(arg):
        return _decode(Path(arg).read_bytes(), arg), False
    return arg, True


def _decode(raw: "bytes | str", what: str) -> str:
    """FASTA text from a file as editors save it: UTF-8 with or without a
    BOM, UTF-16 with a BOM (Windows Notepad), gzip, or a Latin-1 header.
    Only the headers can be non-ASCII; a sequence that is not is refused
    later."""
    if isinstance(raw, str):
        return raw.lstrip("\ufeff")
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as e:
            raise InputError(f"{what}: not a readable gzip file ({e})") from None
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16").lstrip("\ufeff")
        except UnicodeDecodeError:
            raise InputError(f"{what}: not readable UTF-16 text") from None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_records(arg: str) -> List[Tuple[str, List[str]]]:
    """[(name, chains)], one entry per FASTA record (one entry for a bare
    sequence). ':' inside a sequence separates chains of one record."""
    text, _ = _read_text(arg)
    return _parse_records(text)


def _parse_records(text: str) -> List[Tuple[str, List[str]]]:
    recs: List[Tuple[str, List[str]]] = []
    lines = [line.strip() for line in text.splitlines()]
    if next((line for line in lines if line), "").startswith(">"):
        name: Optional[str] = None
        cur: Optional[List[str]] = None
        for line in lines:
            if line.startswith(">"):
                if cur is not None:
                    recs.append((name or "", cur))
                name = line[1:].split()[0] if line[1:].strip() else ""
                cur = []
            elif cur is not None and line:
                cur.append(line)
        if cur is not None:
            recs.append((name or "", cur))
        parts = [(n, "".join(c)) for n, c in recs]
    else:
        parts = [("", "".join(text.split()))]
    out: List[Tuple[str, List[str]]] = []
    for k, (n, p) in enumerate(parts, 1):
        chains = p.split(":")
        cs = []
        for i, c in enumerate(chains, 1):
            try:
                cs.append(canonical_sequence(c))
            except InvalidSequence as e:
                where = _where(n, k, len(parts), i, len(chains))
                raise InvalidSequence(f"{where} is empty" if not c.strip() else f"{where}: {e}") from None
        out.append((n, cs))
    return out


def _where(name: str, k: int, nrec: int, i: int, nchain: int) -> str:
    """'record 2 (b) chain 1', or as much of that as tells the chains apart."""
    parts = []
    if nrec > 1:
        parts.append(f"record {k} ({name})" if name else f"record {k}")
    if nchain > 1:
        parts.append(f"chain {i}")
    return " ".join(parts) or "the sequence"


def read_chains(arg: str) -> Tuple[str, List[str]]:
    """(name, chains): every record's chains in order, named after the
    first record. ':' inside a sequence also separates chains."""
    recs = read_records(arg)
    name = recs[0][0] if recs and recs[0][0] else "query"
    return name, [c for _, cs in recs for c in cs]


def _eq_hint(arg: str) -> str:
    """The argument with '=' in front, quoted: zsh replaces an unquoted word
    that starts with '=' by the path of a command of that name, and the
    quoted form reads the same in bash, zsh and fish."""
    return "'=" + arg.replace("'", "'\\''") + "'"


def _check_near_file(arg: str) -> None:
    """Refuse a typed argument when a file of that name plus an extension
    exists ('query' next to 'query.fasta'). Called only when no file or
    directory of that name exists."""
    if len(arg) < 200 and os.sep not in arg:
        try:
            near = sorted(f.name for f in Path(".").glob(glob.escape(arg) + ".*") if f.is_file())
        except (OSError, ValueError):
            near = []
        if near:
            raise InputError(f"there is no file named {arg!r}. Did you mean {near[0]}? "
                             f"To send {arg!r} as a sequence, write it as {_eq_hint(arg)}.")


def _check_typed(arg: str, chains: List[str]) -> None:
    """Refuse a typed argument that is more likely a file name than a
    sequence because it is so short."""
    short = min(len(c) for c in chains)
    if short < MIN_LITERAL:
        if len(arg) > 40:
            # Too long to be a file name: do not echo the sequence back.
            raise InputError(f"a chain of the sequence has only {short} residues. To send a short chain "
                             "on purpose, put '=' in front of the whole argument and quote it, as in '=SEQ'.")
        raise InputError(f"{arg!r} is not a file, and as a sequence it has only {short} residues. "
                         f"To send a short sequence on purpose, write it as {_eq_hint(arg)}.")


def _jobs(arg: str, complex_: bool, batch: bool, name: Optional[str]) -> List[Tuple[str, List[str]]]:
    """The jobs to run as (name, chains). Raises InputError or InvalidSequence."""
    text, typed = _read_text(arg)
    if typed:
        _check_near_file(arg)
    recs = _parse_records(text)
    try:
        for k, (n, cs) in enumerate(recs, 1):
            for i, c in enumerate(cs, 1):
                bad = sorted({ch for ch in c if not ("A" <= ch <= "Z")})
                if bad:
                    raise InvalidSequence(f"{_where(n, k, len(recs), i, len(cs))} has characters that are "
                                          f"not amino acid letters: {''.join(bad)!r}")
    except InvalidSequence as e:
        if typed and (len(arg) <= 40 or os.sep in arg):
            # Name the argument only when it could be a file name, so a long
            # mistyped sequence is not echoed back to the terminal or a log.
            raise InvalidSequence(f"{e}. No file named {arg!r} exists either.") from None
        raise
    if typed:
        _check_typed(arg, [c for _, cs in recs for c in cs])
    if len(recs) > 1 and not (complex_ or batch):
        raise InputError(f"{arg} has {len(recs)} records. Pass --complex to fold them as one complex of "
                         f"{len(recs)} chains, or --batch to run each record as its own job.")
    if batch:
        if len(recs) > MAX_JOBS:
            raise InputError(f"{arg} has {len(recs)} records; --batch takes at most {MAX_JOBS}")
        jobs: List[Tuple[str, List[str]]] = []
        seen: set = set()
        for i, (n, cs) in enumerate(recs, 1):
            # --name is a prefix shared by every job of the batch.
            base = _fname(f"{name}_{n or f'job{i}'}" if name else n or f"job{i}")
            nm, k = base, 1
            while _fold(nm) in seen:
                k += 1
                nm = f"{base}_{k}"
            seen.add(_fold(nm))
            jobs.append((nm, cs))
    else:
        jobs = [(name or recs[0][0] or "query", [c for _, cs in recs for c in cs])]
    for n, cs in jobs:
        if len(cs) > MAX_CHAINS:
            what = f"record {n}" if batch else "the complex"
            raise InputError(f"{what} has {len(cs)} chains; one complex takes at most {MAX_CHAINS}")
    return jobs


def _fold(name: str) -> str:
    """The key two folder names share when a file system that ignores case
    and Unicode normalization (APFS, NTFS) takes them for one folder: 'SS'
    and 'ß', or a precomposed Hangul syllable and its jamo."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def _unusable_out(out: Path, model: Optional[str] = None, jobs: Optional[List[str]] = None) -> Optional[str]:
    """Why --out cannot be written, checked before anything is sent, or None.
    With a model, also the folders the model's files go in, under each
    job's folder for a batch."""
    targets = [out]
    if model:
        sub = (model,) if model in ("af3", "colabfold") else (model, "msas")
        for base in [out / j for j in jobs] if jobs else [out]:
            targets.append(base.joinpath(*sub))
    for t in targets:
        p = t
        while not p.exists() and p != p.parent:
            p = p.parent
        if not p.is_dir():
            return f"--out {out}: {p} is not a directory"
        if not os.access(p, os.W_OK | os.X_OK):
            return f"--out {out}: cannot write to {p}"
    return None


def _warn_nucleic(jobs: List[Tuple[str, List[str]]]) -> None:
    for n, cs in jobs:
        for i, c in enumerate(cs, 1):
            if len(c) >= MIN_LITERAL and set(c) <= set("ACGTUN"):
                what = f"chain {i}" if len(cs) > 1 else "the sequence"
                where = f" of {n}" if len(jobs) > 1 else ""
                print(f"warning: {what}{where} has only A, C, G, T, U and N, so it looks like DNA or RNA. "
                      "This tool searches protein databases.", file=sys.stderr)


def _n(x: float) -> str:
    return f"{int(round(x)):,}"


def _depths(chains: List[Alignment], quiet: bool, job: Optional[str] = None) -> None:
    """Say how deep each chain's alignment is, how much of it came from
    open data, and warn when it is so shallow that the model will in
    effect run on the sequence alone. In a batch, ``job`` names the job,
    since -q hides the job lines."""
    from .writers import entities

    ents = entities(chains)
    names = _ent_names(ents)
    for (c, _, _), nm in zip(ents, names):
        what = f"chain {nm}" if len(ents) > 1 or len(chains) > 1 else "sequence"
        if job:
            what = f"{job}: {what}"
        d = c.depth
        dep = c.provenance.get("depth") if isinstance(c.provenance.get("depth"), dict) else None
        if not quiet:
            if dep and c.source == "merged":
                kept = (c.provenance.get("merge") or {}).get("kept_open_rows") or {}
                by = ", ".join(f"{k} {_n(v)}" for k, v in kept.items())
                print(f"{what}: {_n(dep['rows'])} rows, about {_n(dep['neff80'])} Neff (80%), {_n(dep['species'])} "
                      f"species. ColabFold alone: {_n(dep['server_rows'])} rows, about {_n(dep['server_neff80'])} Neff, "
                      f"{_n(dep['server_species'])} species. Open data added {_n(dep['open_rows'])} rows"
                      + (f" ({by})" if by else "") + ".", file=sys.stderr)
            elif dep and c.source == "open":
                kept = (c.provenance.get("merge") or {}).get("kept_open_rows") or {}
                by = ", ".join(f"{k} {_n(v)}" for k, v in kept.items())
                print(f"{what}: {_n(dep['rows'])} rows from open data" + (f" ({by})" if by else "")
                      + f", about {_n(dep['neff80'])} Neff (80%), {_n(dep['species'])} species", file=sys.stderr)
            else:
                print(f"{what}: depth {d}", file=sys.stderr)
        if d <= 1:
            hint = (" No open alignment holds this sequence; run without --open-only to search it."
                    if c.source == "open" else "")
            print(f"warning: {what} has depth 1 (the query only). The model will run in single-sequence "
                  f"mode, which is usually much less accurate.{hint}", file=sys.stderr)
        elif d < LOW_DEPTH:
            print(f"warning: {what} has only {d} sequences in its alignment. Predictions from a shallow "
                  "MSA are often poor.", file=sys.stderr)


def _sent(chains: List[Alignment], quiet: bool) -> None:
    """Say what the open data lookups of this job sent, and where."""
    if quiet:
        return
    sent: List[str] = []
    looked = False
    for c in chains:
        o = c.provenance.get("open")
        if isinstance(o, dict):
            looked = True
            for x in o.get("sent") or []:
                if x not in sent:
                    sent.append(x)
    if looked:
        print("open data sent: " + ("; ".join(sent) if sent else "nothing (tables, answers and files came from the cache)"),
              file=sys.stderr)


def _ent_names(ents) -> List[str]:
    from .adapters import chain_id

    return [", ".join(chain_id(i) for i in idx) for _, _, idx in ents]


def _report(files: List[str], nxt: str, quiet: bool) -> None:
    if quiet:
        return
    for f in files:
        print(f"wrote {f}", file=sys.stderr)
    print(f"next: {nxt}", file=sys.stderr)


def _confirm(tool: MSA, pending: List[str], yes: bool, jobs: Optional[List[Tuple[str, List[str]]]] = None,
             pairs: int = 0) -> Optional[int]:
    """None to go ahead, or an exit code."""
    n = len(pending)
    if n > 10:
        lo, hi = min(map(len, pending)), max(map(len, pending))
        lengths = f"{lo} to {hi}" if lo != hi else f"{lo} each"
    else:
        lengths = ", ".join(str(len(q)) for q in pending)
    if jobs and len(jobs) > 1:
        print(f"Batch of {len(jobs)} separate jobs.", file=sys.stderr)
    elif jobs and len(jobs[0][1]) > 1:
        print(f"One complex of {len(jobs[0][1])} chains.", file=sys.stderr)
    if n:
        print(f"This sends {n} sequence{'s' if n > 1 else ''} (length {lengths}) to {tool.describe_server()}.",
              file=sys.stderr)
    if pairs:
        print(f"It also sends {'the chains of the complex' if pairs == 1 else f'the chains of {pairs} complexes'} "
              f"together for {tool.server_name}'s paired search, as colabfold_batch does "
              f"({pairs} more job{'s' if pairs > 1 else ''}).", file=sys.stderr)
    if n + pairs > 1 and tool.server == "colabfold" and _is_default_host(tool.backend):
        n = n + pairs
        print(f"The public server takes about one new job per minute, so this can take {n} minutes or more "
              "(less for sequences the server has already searched).", file=sys.stderr)
    if yes:
        return None
    if not sys.stdin.isatty():
        return die("not sending the sequence without a confirmation. Run again with --yes to allow it, "
                   "or use 'scigantic-msa convert' with an alignment you already have.", 2)
    try:
        ans = input("Send it? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes"):
        print("Nothing was sent.", file=sys.stderr)
        return 1
    return None


def cmd_run(a: argparse.Namespace) -> int:
    try:
        jobs = _jobs(a.sequence, a.complex, a.batch, a.name)
    except (InvalidSequence, InputError, OSError) as e:
        return die(str(e))
    _warn_nucleic(jobs)
    bad_out = _unusable_out(Path(a.out), a.model, [n for n, _ in jobs] if a.batch else None)
    if bad_out:
        return die(bad_out)
    # What the model's writer would refuse is refused here, before the
    # sequence is sent: sending it would publish it on the public server
    # for nothing. In a batch only that job is refused.
    refused: List[str] = []
    for name, chains in list(jobs):
        try:
            check_model(a.model, chains)
        except ImportError as e:
            return die(f"{e}. Nothing was sent.")
        except PrepareError as e:
            if not a.batch:
                return die(f"{e} Nothing was sent.")
            print(f"error: {name}: {e}", file=sys.stderr)
            refused.append(name)
    if refused:
        if len(refused) == len(jobs):
            return die(f"{a.model} cannot read any job of the batch, so nothing was sent.")
        jobs = [j for j in jobs if j[0] not in refused]
    if a.open_only and a.no_open:
        return die("--open-only and --no-open leave nothing to search; pass one of them")
    if a.open_only and (a.host or a.server != "colabfold"):
        return die("--open-only sends the sequence to no server, so --host and --server do not apply")
    try:
        tool = MSA(None if a.open_only else a.server, host=a.host, cache=not a.no_cache, quiet=a.quiet,
                   open_data=not a.no_open, openproteinset=a.openproteinset, near=a.near, open_pairs=a.open_pairs)
    except ValueError as e:
        return die(str(e))
    if not a.quiet and tool.open is not None:
        print(f"Also uses {tool.describe_open()}.", file=sys.stderr)
    if a.open_only:
        if not a.quiet:
            print("--open-only: the sequence is not sent to any search server.", file=sys.stderr)
        pending: List[str] = []
        pairs = 0
    else:
        pending = [q for q in dict.fromkeys(c for _, cs in jobs for c in cs) if tool.cached(q) is None]
        pairs = sum(1 for _, cs in jobs if tool.pair_pending(cs))
    if (pending or pairs) and remote_disabled():
        return die("remote search is disabled (SCIGANTIC_MSA_NO_REMOTE is set), so nothing was sent. "
                   "Use 'scigantic-msa convert' with an alignment you already have.")
    if len(pending) + pairs > MAX_PUBLIC_SEARCHES and tool.server == "colabfold" and _is_default_host(tool.backend):
        return die(f"this would send {len(pending) + pairs} new jobs to the public ColabFold server; one run sends "
                   f"at most {MAX_PUBLIC_SEARCHES}. Nothing was sent. Split the input into smaller runs, or use "
                   "your own server with --host.")
    check = getattr(tool.backend, "check_ready", None)
    if (pending or pairs) and check is not None:
        try:
            check()
        except RemoteError as e:
            return die(f"{e}. Nothing was sent.")
    if pending or pairs:
        rc = _confirm(tool, pending, a.yes, jobs, pairs)
        if rc is not None:
            return rc
    elif not a.quiet and not a.open_only:
        print("Every chain is in the local cache; nothing is sent to the search server.", file=sys.stderr)
    out = Path(a.out)
    failed: List[str] = []
    total = len(jobs) + len(refused)
    try:
        for k, (name, chains) in enumerate(jobs, 1):
            if a.batch and not a.quiet:
                print(f"scigantic-msa: job {k} of {len(jobs)}: {name}", file=sys.stderr)
            try:
                got = tool.get(chains, remote=tool.backend is not None)
                files, nxt = write_model(a.model, got, out / name if a.batch else out, name, a.quiet)
            except (JobFailed, A3MError, PrepareError) as e:
                # Only this job's sequence or result is at fault, so the rest
                # of a batch can still run. Rate limits and an unreachable
                # server stop everything below.
                if not a.batch:
                    raise
                print(f"error: {name}: {e}", file=sys.stderr)
                # The model refusing the input is not fixed by running again.
                (refused if isinstance(e, PrepareError) else failed).append(name)
                continue
            _report(files, nxt, a.quiet)
            _depths(got, a.quiet, name if a.batch else None)
            _sent(got, a.quiet)
    except (RemoteError, RemoteNotAllowed, A3MError, PrepareError, ImportError, OSError, ValueError) as e:
        return die(str(e))
    except KeyboardInterrupt:
        if pending:
            print("Interrupted. The search may still be running on the server; run the same command "
                  "again to pick up the same job without sending it again.", file=sys.stderr)
        return 130
    if failed or refused:
        msg = f"{len(failed) + len(refused)} of {total} jobs failed: {', '.join(failed + refused)}."
        if not refused:
            return die(f"{msg} The others were written; run the same command again to retry, and finished "
                       "jobs come from the cache.")
        msg += (f" The others were written. {a.model} cannot read {', '.join(refused)}; change "
                f"{'its' if len(refused) == 1 else 'their'} input, since running again gives the same error.")
        if failed:
            msg += f" Run the same command again to retry {', '.join(failed)}; finished jobs come from the cache."
        return die(msg)
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    try:
        uni = a.uniprot or []
        if any(not u.strip() for u in uni):
            return die("--uniprot was given an empty file name")
        if uni and len(uni) != len(a.alignments):
            return die(f"--uniprot names {len(uni)} file(s) for {len(a.alignments)} alignment(s); give one per chain")
        chains = [read_alignment(Path(f), uniprot=Path(u) if u else None)
                  for f, u in zip(a.alignments, uni or [None] * len(a.alignments))]
        if a.sequence:
            want = [canonical_sequence(c) for c in a.sequence.split(":")]
            if len(want) != len(chains):
                return die(f"--sequence names {len(want)} chain(s) for {len(chains)} file(s)")
            for f, w, c in zip(a.alignments, want, chains):
                if w != c.query:
                    return die(f"{f}: the first record is not the sequence given with --sequence")
        name = a.name or Path(a.alignments[0]).name.split(".")[0]
        files, nxt = write_model(a.model, chains, Path(a.out), name, a.quiet)
    except (A3MError, PrepareError, InvalidSequence, ImportError, OSError, ValueError) as e:
        return die(str(e))
    _report(files, nxt, a.quiet)
    _depths(chains, a.quiet)
    return 0


def cmd_cache(a: argparse.Namespace) -> int:
    """Print the cache directory, so it can be found and deleted. It may
    not exist yet."""
    print(cache_dir())
    return 0


def cmd_sources(a: argparse.Namespace) -> int:
    """The open data sets, and which of their tables this cache holds."""
    from .opendata.lookup import OpenData
    from .opendata.sources import SOURCES, UPSTREAM
    from .opendata.tables import SIZES_MB

    od = OpenData(cache_dir(), openproteinset=a.openproteinset, say=lambda m: print(f"scigantic-msa: {m}", file=sys.stderr))
    for sid, s in SOURCES.items():
        print(f"{sid}: {s['title']}, {s['license']}{'' if s['default'] else ' (opt in)'}")
        print(f"  {s['where']}")
        print(f"  searched: {s['databases']}")
    for tid in SIZES_MB:
        if a.fetch and (tid != "openproteinset" or a.openproteinset):
            try:
                od.tables.ensure(tid)
            except Exception as e:  # noqa: BLE001 - say it and go on with the next table
                print(f"error: {tid}: {e}", file=sys.stderr)
        t = od.tables.info(tid)
        if t is None:
            print(f"table {tid}: not downloaded (about {SIZES_MB[tid]} MB)")
        else:
            f = t.files[0] if t.files else {}
            print(f"table {tid}: {t.entries:,} sequences, built {t.built_at}, ETag {f.get('etag')}")
    print("Terms of the searched databases:")
    for k, v in UPSTREAM.items():
        print(f"  {k}: {v['terms']}{'' if v['checked'] else ' (not confirmed)'}")
    return 0


def cmd_version(a: argparse.Namespace) -> int:
    print(f"scigantic-msa {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scigantic-msa",
        description="Get a protein MSA once and write the input files for a structure predictor.")
    p.add_argument("--version", action="version", version=f"scigantic-msa {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="build the MSA from open data and a server search, then write the model's input")
    r.add_argument("sequence", help="a FASTA file, - for stdin, or a sequence (start it with '=' and quote it, "
                                    "as in '=MKVLAAG', if it is shorter than 10 residues); ':' separates chains of a complex")
    r.add_argument("--model", "-m", choices=MODELS, required=True)
    r.add_argument("--out", "-o", required=True, help="output directory")
    r.add_argument("--server", choices=BACKENDS, default="colabfold", help="where the sequence is sent (default colabfold)")
    r.add_argument("--host", help="URL of another server of that kind, for example your own ColabFold server")
    r.add_argument("--open-only", action="store_true",
                   help="use only published open alignments; the sequence is sent to no search server "
                        "(only its MD5 goes to UniProt). For unpublished sequences")
    r.add_argument("--no-open", action="store_true", help="use the search server alone, without open data")
    r.add_argument("--openproteinset", action="store_true",
                   help="also use OpenProteinSet's PDB alignments (about 610 MB of tables the first time)")
    r.add_argument("--near", action="store_true",
                   help="also use open entries at 95%% identity or more over 90%% of the query (needs mmseqs on PATH)")
    r.add_argument("--open-pairs", choices=("auto", "always", "never"), default="auto",
                   help="for a complex, when to add pairs by species from open data to the server's pairs: auto "
                        "(default: only when the chains share an organism), always or never")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--complex", action="store_true",
                   help=f"a FASTA with several records is one complex, one chain per record (at most {MAX_CHAINS})")
    g.add_argument("--batch", action="store_true",
                   help=f"each FASTA record is its own job, written to its own folder under --out (at most {MAX_JOBS}); "
                        "with --name, job names are NAME_<record>")
    r.add_argument("--yes", "-y", action="store_true", help="do not ask before sending the sequence")
    r.add_argument("--no-cache", action="store_true", help="do not read or write the local cache")
    r.add_argument("--name", help="job name used for file names (default: FASTA header or 'query'); "
                                  "with --batch, a prefix for every job's name")
    r.add_argument("--quiet", "-q", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("convert", help="write the model's input from alignments you already have (no network)")
    c.add_argument("alignments", nargs="+", help="one .a3m (or .a3m.gz, .sto) file per chain, in chain order")
    c.add_argument("--model", "-m", choices=MODELS, required=True)
    c.add_argument("--out", "-o", required=True, help="output directory")
    c.add_argument("--sequence", help="check the files' first records against these chains (':' between chains)")
    c.add_argument("--uniprot", action="append", metavar="FILE",
                   help="a UniProt alignment of the chain (.a3m or .sto, tr|ACC|NAME_SPECIES headers) used for "
                        "pairing; repeat once per chain, in chain order")
    c.add_argument("--name", help="job name used for file names (default: the first file's name)")
    c.add_argument("--quiet", "-q", action="store_true")
    c.set_defaults(func=cmd_convert)

    sub.add_parser("cache", help="print the local cache directory (delete it to forget every result)"
                   ).set_defaults(func=cmd_cache)
    o = sub.add_parser("sources", help="list the open data sets, their licences and the tables in the cache")
    o.add_argument("--fetch", action="store_true", help="download (or refresh) the tables now")
    o.add_argument("--openproteinset", action="store_true", help="with --fetch, also the OpenProteinSet tables")
    o.set_defaults(func=cmd_sources)
    sub.add_parser("version", help="print the version").set_defaults(func=cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.func(a) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
