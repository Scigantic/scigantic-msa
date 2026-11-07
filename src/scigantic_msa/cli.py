"""scigantic-msa command line.

    scigantic-msa run SEQ_OR_FASTA --model boltz --out DIR [--server colabfold|nvidia] [--yes]
    scigantic-msa convert ALIGNMENT.a3m [MORE.a3m ...] --model af3 --out DIR [--uniprot U.sto ...]
    scigantic-msa version

``run`` sends the sequence to the search server and asks once before it
does (``--yes`` skips the question). ``convert`` reads alignments you
already have and sends nothing.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .a3m import A3MError
from .adapters import PrepareError
from .alignment import Alignment
from .remote import BACKENDS, remote_disabled
from .remote.colabfold import JobFailed, RemoteError
from .seqkey import InvalidSequence, canonical_sequence
from .tool import MSA, RemoteNotAllowed, _is_default_host, read_alignment
from .write import MODELS, _fname, write_model



def die(msg: str, code: int = 1) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


#: A sequence typed on the command line shorter than this is refused unless
#: it starts with '=', because it is more likely a mistyped file name.
MIN_LITERAL = 10
#: Most chains in one complex, and most jobs in one --batch run.
MAX_CHAINS = 20
MAX_JOBS = 100
#: Below this many sequences an alignment is called shallow.
LOW_DEPTH = 20


class InputError(ValueError):
    """The run argument cannot be used as a sequence or a FASTA file."""


def _read_text(arg: str) -> Tuple[str, bool]:
    """(text, typed): typed is True when the argument itself is the
    sequence and was not marked with a leading '='."""
    if arg == "-":
        return sys.stdin.read().lstrip("\ufeff"), False
    if arg.startswith("="):
        return arg[1:], False
    if os.path.isdir(arg):
        raise InputError(f"{arg} is a directory. Pass a FASTA file, a sequence, or - for stdin.")
    if os.path.isfile(arg):
        return Path(arg).read_text(encoding="utf-8-sig"), False
    return arg, True


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
                             f"To send {arg!r} as a sequence, write it as ={arg}.")


def _check_typed(arg: str, chains: List[str]) -> None:
    """Refuse a typed argument that is more likely a file name than a
    sequence because it is so short."""
    short = min(len(c) for c in chains)
    if short < MIN_LITERAL:
        raise InputError(f"{arg!r} is not a file, and as a sequence it has only {short} residues. "
                         f"To send a short sequence on purpose, write it as ={arg}.")


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
            base = _fname(n or f"job{i}")
            nm, k = base, 1
            while nm.lower() in seen:
                k += 1
                nm = f"{base}_{k}"
            seen.add(nm.lower())
            jobs.append((nm, cs))
    else:
        jobs = [(name or recs[0][0] or "query", [c for _, cs in recs for c in cs])]
    for n, cs in jobs:
        if len(cs) > MAX_CHAINS:
            what = f"record {n}" if batch else "the complex"
            raise InputError(f"{what} has {len(cs)} chains; one complex takes at most {MAX_CHAINS}")
    return jobs


def _unusable_out(out: Path) -> Optional[str]:
    """Why --out cannot be written, checked before anything is sent, or None."""
    p = out
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


def _depths(chains: List[Alignment], quiet: bool) -> None:
    """Say how deep each chain's alignment is, and warn when it is so
    shallow that the model will in effect run on the sequence alone."""
    for i, c in enumerate(chains, 1):
        what = f"chain {i}" if len(chains) > 1 else "sequence"
        d = c.depth
        if not quiet:
            print(f"{what}: depth {d}", file=sys.stderr)
        if d <= 1:
            print(f"warning: {what} has depth 1 (the query only). The model will run in single-sequence "
                  "mode, which is usually much less accurate.", file=sys.stderr)
        elif d < LOW_DEPTH:
            print(f"warning: {what} has only {d} sequences in its alignment. Predictions from a shallow "
                  "MSA are often poor.", file=sys.stderr)


def _report(files: List[str], nxt: str, quiet: bool) -> None:
    if quiet:
        return
    for f in files:
        print(f"wrote {f}", file=sys.stderr)
    print(f"next: {nxt}", file=sys.stderr)


def _confirm(tool: MSA, pending: List[str], yes: bool, jobs: Optional[List[Tuple[str, List[str]]]] = None) -> Optional[int]:
    """None to go ahead, or an exit code."""
    n = len(pending)
    if n > 10:
        lengths = f"{min(map(len, pending))} to {max(map(len, pending))}"
    else:
        lengths = ", ".join(str(len(q)) for q in pending)
    if jobs and len(jobs) > 1:
        print(f"Batch of {len(jobs)} separate jobs.", file=sys.stderr)
    elif jobs and len(jobs[0][1]) > 1:
        print(f"One complex of {len(jobs[0][1])} chains.", file=sys.stderr)
    print(f"This sends {n} sequence{'s' if n > 1 else ''} (length {lengths}) to {tool.describe_server()}.",
          file=sys.stderr)
    if n > 1 and tool.server == "colabfold" and _is_default_host(tool.backend):
        print(f"The public server takes about one new job per minute, so this takes at least {n} minutes.",
              file=sys.stderr)
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
    bad_out = _unusable_out(Path(a.out))
    if bad_out:
        return die(bad_out)
    try:
        tool = MSA(a.server, host=a.host, cache=not a.no_cache, quiet=a.quiet)
    except ValueError as e:
        return die(str(e))
    pending = [q for q in dict.fromkeys(c for _, cs in jobs for c in cs) if tool.cached(q) is None]
    if pending and remote_disabled():
        return die("remote search is disabled (SCIGANTIC_MSA_NO_REMOTE is set), so nothing was sent. "
                   "Use 'scigantic-msa convert' with an alignment you already have.")
    if pending:
        rc = _confirm(tool, pending, a.yes, jobs)
        if rc is not None:
            return rc
    elif not a.quiet:
        print("Every chain is in the local cache; nothing is sent.", file=sys.stderr)
    out = Path(a.out)
    failed: List[str] = []
    try:
        for k, (name, chains) in enumerate(jobs, 1):
            if a.batch and not a.quiet:
                print(f"scigantic-msa: job {k} of {len(jobs)}: {name}", file=sys.stderr)
            try:
                got = tool.get(chains, remote=True)
                files, nxt = write_model(a.model, got, out / name if a.batch else out, name, a.quiet)
            except (JobFailed, A3MError, PrepareError) as e:
                # Only this job's sequence or result is at fault, so the rest
                # of a batch can still run. Rate limits and an unreachable
                # server stop everything below.
                if not a.batch:
                    raise
                print(f"error: {name}: {e}", file=sys.stderr)
                failed.append(name)
                continue
            _report(files, nxt, a.quiet)
            _depths(got, a.quiet)
    except (RemoteError, RemoteNotAllowed, A3MError, PrepareError, ImportError, OSError, ValueError) as e:
        return die(str(e))
    except KeyboardInterrupt:
        if pending:
            print("Interrupted. The search may still be running on the server; run the same command "
                  "again to pick up the same job without sending it again.", file=sys.stderr)
        return 130
    if failed:
        return die(f"{len(failed)} of {len(jobs)} jobs failed: {', '.join(failed)}. The others were written; "
                   "run the same command again to retry, and finished jobs come from the cache.")
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    try:
        uni = a.uniprot or []
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


def cmd_version(a: argparse.Namespace) -> int:
    print(f"scigantic-msa {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scigantic-msa",
        description="Get a protein MSA once and write the input files for a structure predictor.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="search a server for the MSA, then write the model's input")
    r.add_argument("sequence", help="a FASTA file, - for stdin, or a sequence (start it with '=' if it is "
                                    "shorter than 10 residues); ':' separates chains of a complex")
    r.add_argument("--model", "-m", choices=MODELS, required=True)
    r.add_argument("--out", "-o", required=True, help="output directory")
    r.add_argument("--server", choices=BACKENDS, default="colabfold", help="where the sequence is sent (default colabfold)")
    r.add_argument("--host", help="URL of another server of that kind, for example your own ColabFold server")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--complex", action="store_true",
                   help=f"a FASTA with several records is one complex, one chain per record (at most {MAX_CHAINS})")
    g.add_argument("--batch", action="store_true",
                   help=f"each FASTA record is its own job, written to its own folder under --out (at most {MAX_JOBS})")
    r.add_argument("--yes", "-y", action="store_true", help="do not ask before sending the sequence")
    r.add_argument("--no-cache", action="store_true", help="do not read or write the local cache")
    r.add_argument("--name", help="job name used for file names (default: FASTA header or 'query')")
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
