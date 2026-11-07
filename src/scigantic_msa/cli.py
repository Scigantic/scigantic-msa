"""scigantic-msa command line.

    scigantic-msa get MKTAYIAKQRQISFVKSHFSRQ --format af3 --out input.json
    scigantic-msa get chains.fasta --format chai --out msas/
    scigantic-msa where MKTAYIAKQRQISFVKSHFSRQ
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from . import __version__
from .a3m import A3MError
from .core import MSANotFound, get_msa, get_msas, resolve_remote, where
from .index import Index, IndexIntegrityError, IndexUnavailable, index_url
from .msa import MSA
from .remote import remote_disabled
from .remote.colabfold import TERMS
from .seqkey import InvalidSequence, canonical_sequence, seq_sha256

FORMATS = ("a3m", "af3", "boltz", "chai", "colabfold", "openfold3")


def die(msg: str, code: int = 1) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


# ----------------------------------------------------------------------------- input


def read_jobs(arg: str) -> List[Tuple[str, List[str]]]:
    """(name, chains) per job. A FASTA file gives one job per record; a
    record or argument with ':' is a complex (ColabFold convention)."""
    if arg == "-":
        text = sys.stdin.read()
    elif os.path.exists(arg):
        text = Path(arg).read_text()
    else:
        text = arg
    jobs: List[Tuple[str, List[str]]] = []
    if text.lstrip().startswith(">"):
        name: Optional[str] = None
        parts: List[str] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if name is not None:
                    jobs.append((name, "".join(parts).split(":")))
                name = line[1:].strip().split()[0] if line[1:].strip() else f"seq{len(jobs) + 1}"
                parts = []
            elif line.strip():
                parts.append(line.strip())
        if name is not None:
            jobs.append((name, "".join(parts).split(":")))
    else:
        jobs.append(("query", "".join(text.split()).split(":")))
    return [(name, [canonical_sequence(c) for c in chains]) for name, chains in jobs]


# ----------------------------------------------------------------------------- output


def write_output(result: Any, fmt: str, out: Optional[str], *, af3_input: Optional[str], name: str) -> str:
    """Write one job. Returns a short description of what was written."""
    chains = list(result) if not isinstance(result, MSA) else [result]
    if fmt == "a3m":
        if len(chains) > 1:
            raise ValueError("--format a3m is for one chain; use colabfold, af3, boltz, chai or openfold3 for a complex")
        if out is None or out == "-":
            sys.stdout.write(chains[0].a3m())
            return "stdout"
        p = Path(out)
        if p.is_dir() or out.endswith("/"):
            p = p / f"{name}.a3m"
        chains[0].to_a3m(p)
        return str(p)
    if out is None:
        raise ValueError(f"--format {fmt} needs --out")
    p = Path(out)
    if fmt == "colabfold":
        if p.is_dir() or out.endswith("/"):
            p = p / f"{name}.a3m"
        result.to_colabfold(p)
        return str(p)
    if fmt == "af3":
        if p.is_dir() or out.endswith("/"):
            p = p / f"{name}.json"
        kw = {"input": af3_input} if af3_input else {"name": name}
        result.to_af3_json(p, **kw)
        return str(p)
    if fmt == "boltz":
        lines = result.to_boltz(p) if not isinstance(result, MSA) else [result.to_boltz(p)]
        return f"{p} (YAML: " + "; ".join(lines) + ")"
    if fmt == "chai":
        written = result.to_chai(p)
        return str(p) if not isinstance(written, Path) else str(written)
    if fmt == "openfold3":
        fields = result.to_openfold3(p) if not isinstance(result, MSA) else [result.to_openfold3(p)]
        return f"{p} (query JSON fields: {json.dumps(fields)})"
    raise ValueError(f"unknown format {fmt}")


# ----------------------------------------------------------------------------- get


def _ask_remote(name: str) -> bool:
    if not sys.stdin.isatty():
        return False
    print(f"No stored MSA for {name}.", file=sys.stderr)
    print(TERMS, file=sys.stderr)
    ans = input("Send this sequence to api.colabfold.com to compute one? [y/N] ").strip().lower()
    return ans in ("y", "yes")


def cmd_get(a: argparse.Namespace) -> int:
    try:
        jobs = read_jobs(a.sequence)
    except (InvalidSequence, OSError) as e:
        return die(str(e))
    if a.no_remote:
        remote: Any = False
    elif a.remote:
        remote = a.remote
    else:
        remote = None
    if len(jobs) > 1 and a.out and not (a.out.endswith("/") or Path(a.out).is_dir()) and a.format != "chai":
        return die("several jobs in the input: give --out a directory (ending in /)")
    rc = 0
    kw = dict(near=not a.no_near,
              purpose="af3" if a.format == "af3" else "a3m", quiet=a.quiet,
              min_identity=a.min_identity)
    for name, chains in jobs:
        try:
            try:
                result = _get(chains, remote=remote, **kw)
            except MSANotFound:
                if remote is None and not remote_disabled() and resolve_remote(None) is None and _ask_remote(name):
                    result = _get(chains, remote="colabfold", **kw)
                else:
                    raise
            where_to = write_output(result, a.format, a.out, af3_input=a.input, name=name)
        except MSANotFound as e:
            rc = die(f"{name}: {e}")
            continue
        except (IndexIntegrityError, IndexUnavailable, A3MError, ValueError, OSError, ImportError) as e:
            rc = die(f"{name}: {e}")
            continue
        if where_to != "stdout" and not a.quiet:
            print(f"{name}: wrote {where_to}", file=sys.stderr)
    return rc


def _get(chains: List[str], *, remote: Any, **kw: Any) -> Any:
    if len(chains) == 1:
        return get_msa(chains[0], remote=remote, **kw)
    return get_msas(chains, remote=remote, **kw)


# ----------------------------------------------------------------------------- where


def cmd_where(a: argparse.Namespace) -> int:
    try:
        jobs = read_jobs(a.sequence)
    except (InvalidSequence, OSError) as e:
        return die(str(e))
    for name, chains in jobs:
        for c in chains:
            q = canonical_sequence(c)
            print(f"{name}: length {len(q)}, seq_sha256 {seq_sha256(q)}")
            try:
                steps = where(q, remote=False if a.no_remote else (a.remote or None))
            except (IndexIntegrityError, ValueError) as e:
                print(f"  error: {e}")
                continue
            for s in steps:
                print(f"  {s}")
            decided = next((s for s in steps if s.outcome in ("hit", "would-run")), None)
            if decided is None:
                print("  -> nothing would answer. Turn on a server with --remote colabfold.")
            else:
                print(f"  -> {decided.name} would answer")
    print(f"index: {index_url()}")
    return 0


# ----------------------------------------------------------------------------- index / version


def cmd_index(a: argparse.Namespace) -> int:
    idx = Index()
    try:
        m = idx.manifest()
    except (IndexUnavailable, IndexIntegrityError) as e:
        return die(str(e))
    if a.action == "info":
        print(f"index:    {idx.base}")
        print(f"build:    {m['build_id']} ({m.get('built_at', '-')})")
        for s in m.get("sources", []):
            print(f"source:   {s.get('id')}: {s.get('title', '')}, {s.get('license', '')}")
        print(f"counts:   {json.dumps(m.get('counts', {}))}")
        return 0
    n = idx.download_all(progress=lambda i: print(f"\r{i}/4096 shards", end="", file=sys.stderr) if i % 64 == 0 else None)
    print(f"\ndownloaded {n} shards and the near-search files to {idx.cache}", file=sys.stderr)
    return 0


def cmd_version(a: argparse.Namespace) -> int:
    print(f"scigantic-msa {__version__}")
    return 0


# ----------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scigantic-msa", description="Protein MSAs for structure predictors.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("get", help="find or compute an MSA and write it")
    g.add_argument("sequence", help="a sequence, a FASTA file, or - for stdin; ':' separates chains of a complex")
    g.add_argument("--format", "-f", choices=FORMATS, default="a3m")
    g.add_argument("--out", "-o", help="output file or directory (a3m goes to stdout without it)")
    g.add_argument("--input", help="existing AlphaFold 3 JSON to fill in (with --format af3)")
    g.add_argument("--remote", choices=("colabfold", "nvidia"), help="search this server when nothing is stored")
    g.add_argument("--no-remote", action="store_true", help="never send the sequence to a server")
    g.add_argument("--no-near", action="store_true", help="do not reuse a near-identical sequence's MSA")
    g.add_argument("--min-identity", type=float, default=0.95)
    g.add_argument("--quiet", "-q", action="store_true")
    g.set_defaults(func=cmd_get)

    w = sub.add_parser("where", help="explain which source would answer, without sending the sequence")
    w.add_argument("sequence")
    w.add_argument("--remote", choices=("colabfold", "nvidia"))
    w.add_argument("--no-remote", action="store_true")
    w.set_defaults(func=cmd_where)

    ix = sub.add_parser("index", help="index info, or download it all for offline use")
    ix.add_argument("action", choices=("info", "download"))
    ix.set_defaults(func=cmd_index)

    sub.add_parser("version").set_defaults(func=cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    try:
        return int(a.func(a) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
