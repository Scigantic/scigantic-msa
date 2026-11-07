"""scigantic-msa command line.

    scigantic-msa get MKTAYIAKQRQISFVKSHFSRQ --format af3 --out input.json
    scigantic-msa get chains.fasta --format chai --out msas/
    scigantic-msa where MKTAYIAKQRQISFVKSHFSRQ
    scigantic-msa login --email you@lab.edu
    scigantic-msa list
    scigantic-msa pull <alignment id> --out x.a3m
    scigantic-msa rm <alignment id>
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from . import __version__
from .a3m import A3MError
from .account import AccountClient, AccountError, NotLoggedIn, config_path, load_config, save_config
from .core import MSANotFound, get_msa, get_msas, resolve_remote, where
from .index import Index, IndexIntegrityError, IndexUnavailable, index_url
from .msa import MSA
from .remote import remote_disabled
from .remote.colabfold import TERMS
from .seqkey import InvalidSequence, canonical_sequence, is_seq_sha256, seq_sha256

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
    kw = dict(save=True if a.save else None, label=a.label, near=not a.no_near,
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
        except (NotLoggedIn, AccountError) as e:
            return die(e.message)
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
                print("  -> nothing would answer. Turn on a server with --remote colabfold, or store one with push.")
            else:
                print(f"  -> {decided.name} would answer")
    print(f"index: {index_url()}")
    return 0


# ----------------------------------------------------------------------------- login


def cmd_login(a: argparse.Namespace) -> int:
    c = AccountClient(a.api_url, token=None)
    email = a.email or input("Email: ").strip()
    if not email:
        return die("no email given")
    try:
        sent = c.login_start(email)
    except AccountError as e:
        return die(e.message)
    print(f"Code sent to {sent.get('email', email)}. Check your inbox (and spam).")
    accept = a.accept_terms
    if sent.get("accountExists") is False and not accept:
        terms = sent.get("termsUrl") or "https://scigantic.com/terms"
        print(f"If {email} has no Scigantic account, one is created when you confirm the code.")
        if sys.stdin.isatty():
            ans = input(f"Accept the terms of service ({terms})? [y/N] ").strip().lower()
            accept = ans in ("y", "yes")
            if not accept:
                return die("the terms must be accepted to create an account")
        else:
            print(f"Pass --accept-terms to accept the terms of service ({terms}).")
    code = a.code or (getpass.getpass("Code: ") if sys.stdin.isatty() else input("Code: "))
    code = code.strip()
    if not code:
        return die("no code entered")
    try:
        tok = c.login_finish(email, code, accept_terms=accept)
    except AccountError as e:
        if e.code == "terms_required":
            return die("this address has no account yet. Run again with --accept-terms to create one.")
        return die(e.message)
    exp = tok.get("expiresAt")
    until = datetime.fromtimestamp(int(exp), tz=timezone.utc).strftime("%Y-%m-%d") if exp else "30 days"
    print(f"Logged in as {tok.get('email', email)} until {until}. Token saved to {config_path()}.")
    if tok.get("accountCreated"):
        print("Your Scigantic account is ready. To use the website, sign in with Google using this "
              "address, or choose Forgot password to set one.")
    return 0


def cmd_logout(a: argparse.Namespace) -> int:
    AccountClient.logout()
    print("Logged out on this machine.")
    return 0


def cmd_account(a: argparse.Namespace) -> int:
    c = AccountClient(a.api_url)
    try:
        d = c.account()
    except AccountError as e:
        return die(e.message)
    lim, use = d.get("limits", {}), d.get("usage", {})
    print(f"Account: {d.get('email', '-')} ({d.get('plan', '-')} plan, signed in via {c.auth_method()})")
    print(f"Stored:  {use.get('count', 0)} alignments, {_mb(use.get('bytes', 0))} "
          f"of {lim.get('maxCount', '-')} and {_mb(lim.get('maxBytes', 0))}")
    return 0


def _mb(n: Any) -> str:
    try:
        return f"{int(n) / 1024 / 1024:.1f} MiB"
    except (TypeError, ValueError):
        return "-"


# ----------------------------------------------------------------------------- list / pull / push / rm


def cmd_list(a: argparse.Namespace) -> int:
    c = AccountClient(a.api_url)
    sha = None
    if a.seq:
        sha = a.seq if is_seq_sha256(a.seq) else seq_sha256(a.seq)
    try:
        rows = list(c.list(sha))
    except AccountError as e:
        return die(e.message)
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    if not rows:
        print("No stored alignments.")
        return 0
    print(f"{'id':<38}{'seq_sha256':<15}{'len':>6}{'size':>11}  {'source':<15}{'created':<12}label")
    for r in rows:
        print(f"{r.get('alignmentId', ''):<38}{str(r.get('seqSha256', ''))[:12] + '...':<15}"
              f"{r.get('queryLength', ''):>6}{_mb(r.get('sizeBytes')):>11}  {str(r.get('source', '')):<15}"
              f"{str(r.get('createdAt', ''))[:10]:<12}{r.get('label') or ''}")
    return 0


def cmd_pull(a: argparse.Namespace) -> int:
    c = AccountClient(a.api_url)
    try:
        msa = c.download(a.id)
    except (AccountError, A3MError) as e:
        return die(str(e))
    try:
        where_to = write_output(msa, a.format, a.out, af3_input=a.input, name=msa.seq_sha256[:12])
    except (ValueError, OSError, ImportError) as e:
        return die(str(e))
    if where_to != "stdout":
        print(f"wrote {where_to}", file=sys.stderr)
    return 0


def cmd_push(a: argparse.Namespace) -> int:
    try:
        text = Path(a.file).read_text()
        msa = MSA.from_a3m(text, source="cache", query=a.seq, provenance={"from": "upload", "file": Path(a.file).name})
    except (OSError, A3MError, InvalidSequence) as e:
        return die(str(e))
    c = AccountClient(a.api_url)
    try:
        aid = c.save(msa, label=a.label)
    except AccountError as e:
        return die(e.message)
    print(f"stored {a.file} as {aid} (private), depth {msa.depth}, seq_sha256 {msa.seq_sha256}")
    return 0


def cmd_rm(a: argparse.Namespace) -> int:
    c = AccountClient(a.api_url)
    rc = 0
    for aid in a.ids:
        try:
            c.delete(aid)
            print(f"deleted {aid}")
        except AccountError as e:
            rc = die(f"{aid}: {e.message}")
    return rc


# ----------------------------------------------------------------------------- config / index / version


def cmd_config(a: argparse.Namespace) -> int:
    cfg = load_config()
    if a.action == "show":
        safe = {k: ("<set>" if k == "token" else v) for k, v in cfg.items()}
        print(json.dumps(safe, indent=2))
        print(f"file: {config_path()}")
        return 0
    key, value = a.key, a.value
    if key == "remote":
        if value not in ("colabfold", "nvidia", "off"):
            return die("remote must be colabfold, nvidia or off")
        cfg["remote"] = None if value == "off" else value
    elif key == "autoSave":
        cfg["autoSave"] = value.lower() in ("1", "true", "yes", "on")
    else:
        return die("settable keys: remote, autoSave")
    save_config(cfg)
    print(f"{key} = {cfg[key]}")
    return 0


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
    p.add_argument("--api-url", help="Scigantic API (default https://api.scigantic.com or SCIGANTIC_API_URL)")
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
    g.add_argument("--save", action="store_true", help="store the result in your Scigantic account (private)")
    g.add_argument("--label", help="label for --save")
    g.add_argument("--quiet", "-q", action="store_true")
    g.set_defaults(func=cmd_get)

    w = sub.add_parser("where", help="explain which source would answer, without sending the sequence")
    w.add_argument("sequence")
    w.add_argument("--remote", choices=("colabfold", "nvidia"))
    w.add_argument("--no-remote", action="store_true")
    w.set_defaults(func=cmd_where)

    lg = sub.add_parser("login", help="sign in with an emailed code (creates an account if needed)")
    lg.add_argument("--email")
    lg.add_argument("--code", help="the code from the email (asked for if missing)")
    lg.add_argument("--accept-terms", action="store_true", help="accept the terms of service for a new account")
    lg.set_defaults(func=cmd_login)

    sub.add_parser("logout", help="forget the saved token").set_defaults(func=cmd_logout)
    sub.add_parser("account", help="plan, usage and limits").set_defaults(func=cmd_account)

    ls = sub.add_parser("list", help="your stored alignments")
    ls.add_argument("--seq", help="only this sequence (or seq_sha256)")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=cmd_list)

    pl = sub.add_parser("pull", help="download a stored alignment")
    pl.add_argument("id")
    pl.add_argument("--format", "-f", choices=FORMATS, default="a3m")
    pl.add_argument("--out", "-o")
    pl.add_argument("--input", help="existing AlphaFold 3 JSON to fill in")
    pl.set_defaults(func=cmd_pull)

    ps = sub.add_parser("push", help="store an A3M file you already have")
    ps.add_argument("file")
    ps.add_argument("--seq", help="check the file's first record against this sequence")
    ps.add_argument("--label")
    ps.set_defaults(func=cmd_push)

    rm = sub.add_parser("rm", help="delete stored alignments")
    rm.add_argument("ids", nargs="+")
    rm.set_defaults(func=cmd_rm)

    cf = sub.add_parser("config", help="show or set defaults")
    cf.add_argument("action", choices=("show", "set"))
    cf.add_argument("key", nargs="?")
    cf.add_argument("value", nargs="?")
    cf.set_defaults(func=cmd_config)

    ix = sub.add_parser("index", help="index info, or download it all for offline use")
    ix.add_argument("action", choices=("info", "download"))
    ix.set_defaults(func=cmd_index)

    sub.add_parser("version").set_defaults(func=cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.api_url:
        os.environ["SCIGANTIC_API_URL"] = a.api_url
    if a.cmd == "config" and a.action == "set" and (not a.key or a.value is None):
        return die("usage: scigantic-msa config set KEY VALUE")
    try:
        return int(a.func(a) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
