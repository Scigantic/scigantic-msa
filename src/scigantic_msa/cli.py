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
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .a3m import A3MError
from .adapters import PrepareError
from .remote import BACKENDS
from .remote.colabfold import RemoteError
from .seqkey import InvalidSequence, canonical_sequence
from .tool import MSA, RemoteNotAllowed, protein_sequence, read_alignment
from .write import MODELS, write_model



def die(msg: str, code: int = 1) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


def read_chains(arg: str) -> Tuple[str, List[str]]:
    """(name, chains). A FASTA with several records is one complex, one
    chain per record. ':' inside a sequence also separates chains."""
    if arg == "-":
        text = sys.stdin.read()
    elif os.path.isfile(arg):
        text = Path(arg).read_text()
    else:
        text = arg
    name = "query"
    parts: List[str] = []
    if text.lstrip().startswith(">"):
        cur: Optional[List[str]] = None
        for line in text.splitlines():
            if line.startswith(">"):
                if cur is not None:
                    parts.append("".join(cur))
                elif line[1:].strip():
                    name = line[1:].strip().split()[0]
                cur = []
            elif cur is not None and line.strip():
                cur.append(line.strip())
        if cur is not None:
            parts.append("".join(cur))
    else:
        parts.append("".join(text.split()))
    chains = [canonical_sequence(c) for p in parts for c in p.split(":")]
    return name, chains


def _report(files: List[str], nxt: str, quiet: bool) -> None:
    if quiet:
        return
    for f in files:
        print(f"wrote {f}", file=sys.stderr)
    print(f"next: {nxt}", file=sys.stderr)


def _confirm(tool: MSA, pending: List[str], yes: bool) -> Optional[int]:
    """None to go ahead, or an exit code."""
    lengths = ", ".join(str(len(q)) for q in pending)
    n = len(pending)
    print(f"This sends {n} sequence{'s' if n > 1 else ''} (length {lengths}) to {tool.describe_server()}.",
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
        name, chains = read_chains(a.sequence)
        chains = [protein_sequence(c, i) for i, c in enumerate(chains, 1)]
    except (InvalidSequence, OSError) as e:
        return die(str(e))
    name = a.name or name
    try:
        tool = MSA(a.server, host=a.host, cache=not a.no_cache, quiet=a.quiet)
    except ValueError as e:
        return die(str(e))
    pending = [q for q in dict.fromkeys(chains) if tool.cached(q) is None]
    if pending:
        rc = _confirm(tool, pending, a.yes)
        if rc is not None:
            return rc
    elif not a.quiet:
        print("Every chain is in the local cache; nothing is sent.", file=sys.stderr)
    try:
        got = tool.get(chains, remote=True)
        files, nxt = write_model(a.model, got, Path(a.out), name, a.quiet)
    except (RemoteError, RemoteNotAllowed, A3MError, PrepareError, ImportError, OSError, ValueError) as e:
        return die(str(e))
    _report(files, nxt, a.quiet)
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
    r.add_argument("sequence", help="a sequence, a FASTA file (one record per chain), or - for stdin; "
                                    "':' separates chains of a complex")
    r.add_argument("--model", "-m", choices=MODELS, required=True)
    r.add_argument("--out", "-o", required=True, help="output directory")
    r.add_argument("--server", choices=BACKENDS, default="colabfold", help="where the sequence is sent (default colabfold)")
    r.add_argument("--host", help="URL of another server of that kind, for example your own ColabFold server")
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
