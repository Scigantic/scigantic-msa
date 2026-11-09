"""Near matches: published entries whose sequence is close to the query
(95% identity or more over 90% or more of the query), found with MMseqs2 on
this machine over the sequences of the downloaded tables.

Optional: it runs only when asked (``near=True``, ``--near``) and an
``mmseqs`` binary is on PATH. The query is written to a temporary file
inside the private cache directory and removed after the search.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..cache import FileLock, private_dirs
from .tables import Tables

__all__ = ["NearUnavailable", "NearHit", "search", "MIN_ID", "MIN_QCOV", "MAX_HITS"]

MIN_ID = 0.95
MIN_QCOV = 0.90
MAX_HITS = 5


class NearUnavailable(RuntimeError):
    pass


@dataclass
class NearHit:
    entry_sha: str
    entry_seq: str
    fident: float
    qcov: float
    qstart: int
    tstart: int
    qaln: str
    taln: str


def _database(tables: Tables, tids: List[str], root: Path, base: Path) -> Path:
    infos = [tables.ensure(t) for t in tids]
    stamp = hashlib.sha256("|".join(str(i.db.name) if i else "-" for i in infos).encode()).hexdigest()[:16]
    d = root / stamp
    db = d / "seqdb"
    if (d / "seqdb.dbtype").exists():
        return db
    private_dirs(d, base)
    lock = FileLock(d / "build", timeout=3600)
    lock.acquire()
    try:
        if (d / "seqdb.dbtype").exists():
            return db
        fa = d / "sequences.fasta"
        seen = set()
        with open(fa, "w") as out:
            for t in tids:
                for sha, seq in tables.sequences(t):
                    if sha not in seen:
                        seen.add(sha)
                        out.write(f">{sha}\n{seq}\n")
        subprocess.run(["mmseqs", "createdb", str(fa), str(db), "-v", "1"], check=True, stdout=subprocess.DEVNULL)
        fa.unlink()
        for old in root.iterdir():
            if old.is_dir() and old != d:
                shutil.rmtree(old, ignore_errors=True)
    finally:
        lock.release()
    return db


def search(query: str, tables: Tables, tids: List[str], root: Path, base: Path, *, threads: int = 4) -> List[NearHit]:
    if not shutil.which("mmseqs"):
        raise NearUnavailable("near matches need mmseqs on PATH; skipped")
    db = _database(tables, tids, root, base)
    seqs = {}
    for t in tids:
        seqs.update(dict(tables.sequences(t)))
    with tempfile.TemporaryDirectory(dir=str(root)) as tmp:
        qf = os.path.join(tmp, "q.fasta")
        with open(qf, "w") as f:
            f.write(f">q\n{query}\n")
        out = os.path.join(tmp, "res.m8")
        cmd = ["mmseqs", "easy-search", qf, str(db), out, os.path.join(tmp, "t"),
               "--min-seq-id", str(MIN_ID), "-c", str(MIN_QCOV), "--cov-mode", "2", "-s", "7.5",
               "--max-seqs", "2000", "-e", "1e-3", "--threads", str(threads),
               "--format-output", "target,fident,qcov,qstart,tstart,qaln,taln", "-v", "1"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise NearUnavailable(f"mmseqs failed ({e.stderr.decode(errors='replace')[-200:].strip()}); "
                                  "near matches skipped") from None
        hits: List[NearHit] = []
        with open(out) as f:
            for line in f:
                t, fid, qcov, qs, ts, qa, ta = line.rstrip("\n").split("\t")
                if t in seqs:
                    hits.append(NearHit(t, seqs[t], float(fid), float(qcov), int(qs) - 1, int(ts) - 1, qa, ta))
    hits.sort(key=lambda h: (-h.fident * h.qcov, h.entry_sha))
    seen, out_hits = set(), []
    for h in hits:
        if h.entry_sha not in seen:
            seen.add(h.entry_sha)
            out_hits.append(h)
    return out_hits[:MAX_HITS + 1]
