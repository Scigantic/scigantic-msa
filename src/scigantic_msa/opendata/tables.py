"""The sequence tables each open set publishes, downloaded once and kept in
the local cache.

A table answers "is there a published alignment for exactly this
sequence?" on this machine, so looking a sequence up sends nothing. Each
table is the publisher's own file:

- OpenFold3: ``msas/pdb_msas_completed_representives.fasta`` (55 MB), the
  sequence of every MSA directory.
- LiteFold: ``manifest.tsv`` (12 MB) at a pinned commit, with the
  sequence and the A3M path of every human protein.
- OpenProteinSet (opt in, about 610 MB): ``data_caches/chain_data_cache.json``
  (the sequence of every PDB chain) and ``alignment_db/alignment_db.index``
  (where each chain's files sit in the alignment_db files).

Each download is recorded with its URL, ETag, Last-Modified, byte count and
sha256 in ``meta.json``, and turned into a small SQLite table keyed by the
sha256 of the canonical sequence. The ETag is checked again at most once a
day (``SCIGANTIC_MSA_OPEN_REFRESH_HOURS``), and a changed ETag downloads the
table again. While SCIGANTIC_MSA_NO_REMOTE is set, the tables already here
are used as they are.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .. import http as _http
from ..cache import FileLock, atomic_write, private_dirs
from ..seqkey import InvalidSequence, canonical_sequence, seq_sha256
from . import net
from .sources import BASES, LITEFOLD_COMMIT, LITEFOLD_REPO

__all__ = ["TABLES", "Tables", "TableInfo", "iter_json_object"]

#: Version of the derived SQLite layout; a change rebuilds the tables.
LAYOUT = 1


def _urls(tid: str) -> List[Tuple[str, str]]:
    """(name, url) of the files a table is built from."""
    if tid == "openfold3":
        return [("fasta", BASES["of3"] + "/msas/pdb_msas_completed_representives.fasta")]
    if tid == "litefold":
        return [("manifest", f"{BASES['hf']}/datasets/{LITEFOLD_REPO}/resolve/{LITEFOLD_COMMIT}/manifest.tsv")]
    if tid == "openproteinset":
        return [("chain_cache", BASES["ops"] + "/data_caches/chain_data_cache.json"),
                ("alignment_index", BASES["ops"] + "/alignment_db/alignment_db.index")]
    raise KeyError(tid)


#: Approximate first download, for messages.
SIZES_MB = {"openfold3": 55, "litefold": 12, "openproteinset": 610}
TABLES = tuple(SIZES_MB)


@dataclass
class TableInfo:
    id: str
    files: List[Dict[str, Any]]
    entries: int
    built_at: str
    checked_at: float
    db: Path

    def public(self) -> Dict[str, Any]:
        """What the attribution file records about the table version used."""
        return {"table": self.id, "entries": self.entries, "built_at": self.built_at,
                "files": [{k: f.get(k) for k in ("url", "etag", "last_modified", "bytes", "sha256")}
                          for f in self.files]}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _refresh_seconds() -> float:
    v = os.environ.get("SCIGANTIC_MSA_OPEN_REFRESH_HOURS", "")
    try:
        return max(0.0, float(v)) * 3600 if v.strip() else 24 * 3600.0
    except ValueError:
        return 24 * 3600.0


def _canon(seq: str) -> Optional[str]:
    try:
        return canonical_sequence(seq)
    except InvalidSequence:
        return None


class Tables:
    """The tables under ``root`` (``<cache>/open/tables``)."""

    def __init__(self, root: Path, cache_base: Path, say: Callable[[str], None] = lambda m: None):
        self.root = root
        self.base = cache_base
        self.say = say
        self._dbs: Dict[str, sqlite3.Connection] = {}
        self._info: Dict[str, TableInfo] = {}
        #: Tables that could not be downloaded in this process: not tried again.
        self._dead: Dict[str, str] = {}

    # ------------------------------------------------------------ state
    def _dir(self, tid: str) -> Path:
        return self.root / tid

    def info(self, tid: str) -> Optional[TableInfo]:
        """The table as it is on disk now, without any network."""
        meta = self._dir(tid) / "meta.json"
        try:
            m = json.loads(meta.read_text())
        except (OSError, ValueError):
            return None
        db = self._dir(tid) / str(m.get("db", ""))
        if m.get("layout") != LAYOUT or not db.is_file():
            return None
        return TableInfo(tid, m.get("files", []), int(m.get("entries", 0)), m.get("built_at", ""),
                         float(m.get("checked_at", 0)), db)

    def ensure(self, tid: str) -> Optional[TableInfo]:
        """The table, downloaded if it is not here and refreshed if its
        ETag changed (checked at most once a day). None when it is not here
        and the network is off."""
        if tid in self._info:
            return self._info[tid]
        if tid in self._dead:
            raise net.OpenDataError(self._dead[tid])
        have = self.info(tid)
        if _http.remote_disabled():
            if have:
                self._info[tid] = have
            return have
        if have and time.time() - have.checked_at < _refresh_seconds():
            self._info[tid] = have
            return have
        private_dirs(self._dir(tid), self.base)
        lock = FileLock(self._dir(tid) / "table", timeout=3600)
        lock.acquire()
        try:
            have = self.info(tid)
            if have and time.time() - have.checked_at < _refresh_seconds():
                self._info[tid] = have
                return have
            if have and self._unchanged(have):
                self._write_meta(have, checked_at=time.time())
                have.checked_at = time.time()
                self._info[tid] = have
                return have
            try:
                got = self._download(tid)
            except (net.OpenDataError, OSError, ValueError) as e:
                self._dead[tid] = str(e)
                raise
            self._info[tid] = got
            return got
        finally:
            lock.release()

    def _unchanged(self, have: TableInfo) -> bool:
        try:
            for f in have.files:
                h = net.head(f["url"])
                if h.status != 200 or not h.etag or h.etag != f.get("etag"):
                    return False
            return True
        except net.OpenDataError:
            # Cannot check now: keep using the table we have.
            return True

    def _write_meta(self, t: TableInfo, checked_at: float) -> None:
        meta = {"layout": LAYOUT, "id": t.id, "files": t.files, "entries": t.entries, "built_at": t.built_at,
                "checked_at": checked_at, "db": t.db.name}
        atomic_write(self._dir(t.id) / "meta.json", json.dumps(meta, indent=1).encode(), mode=0o600)

    # ------------------------------------------------------------ download and build
    def _download(self, tid: str) -> TableInfo:
        d = self._dir(tid)
        self.say(f"downloading the {tid} sequence table (about {SIZES_MB[tid]} MB, once; "
                 "later runs only check whether it changed)")
        raw: Dict[str, Path] = {}
        files: List[Dict[str, Any]] = []
        try:
            for name, url in _urls(tid):
                p = d / f".{name}.download"
                files.append(_fetch_to(url, p))
                raw[name] = p
            stamp = hashlib.sha256("|".join(f["sha256"] for f in files).encode()).hexdigest()[:16]
            db = d / f"{stamp}.sqlite"
            tmp = d / f".{stamp}.sqlite.tmp"
            if tmp.exists():
                tmp.unlink()
            n = _BUILDERS[tid](raw, tmp)
            os.replace(tmp, db)
            try:
                os.chmod(db, 0o600)
            except OSError:
                pass
        finally:
            for p in raw.values():
                try:
                    p.unlink()
                except OSError:
                    pass
        for old in d.glob("*.sqlite"):
            if old != db:
                try:
                    old.unlink()
                except OSError:
                    pass
        t = TableInfo(tid, files, n, _now(), time.time(), db)
        self._write_meta(t, t.checked_at)
        self._dbs.pop(tid, None)
        return t

    # ------------------------------------------------------------ lookup
    def _conn(self, t: TableInfo) -> sqlite3.Connection:
        c = self._dbs.get(t.id)
        if c is None:
            c = sqlite3.connect(f"file:{t.db}?mode=ro", uri=True, check_same_thread=False)
            self._dbs[t.id] = c
        return c

    def lookup(self, tid: str, sha: str) -> List[Dict[str, Any]]:
        """Entries of table ``tid`` whose sequence has this seq_sha256."""
        t = self.ensure(tid)
        if t is None:
            return []
        cur = self._conn(t).execute("SELECT name, data FROM entries WHERE sha = ? ORDER BY name", (sha,))
        return [dict(json.loads(data), name=name) for name, data in cur]

    def sequences(self, tid: str) -> Iterator[Tuple[str, str]]:
        """(sha, sequence) of every entry, for a near-match search."""
        t = self.ensure(tid)
        if t is None:
            return
        cur = self._conn(t).execute("SELECT DISTINCT sha, seq FROM entries")
        yield from cur

    def entry_by_sha(self, tid: str, sha: str) -> List[Dict[str, Any]]:
        return self.lookup(tid, sha)

    def close(self) -> None:
        for c in self._dbs.values():
            c.close()
        self._dbs.clear()


def _fetch_to(url: str, path: Path) -> Dict[str, Any]:
    """Stream ``url`` into ``path``; return its record."""
    for attempt in range(net.ATTEMPTS):
        sha = hashlib.sha256()
        n = 0
        try:
            with net.stream(url, timeout=120) as (r, h):
                with open(path, "wb") as out:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        sha.update(chunk)
                        n += len(chunk)
                        out.write(chunk)
                want = h.get("content-length")
                if want and want.isdigit() and int(want) != n:
                    raise ConnectionError(f"got {n} of {want} bytes")
                return {"url": url, "final_url": getattr(r, "url", url), "etag": net.etag_of(h),
                        "last_modified": h.get("last-modified"), "bytes": n, "sha256": sha.hexdigest(),
                        "fetched_at": _now()}
        except (ConnectionError, OSError) as e:
            if isinstance(e, net.OpenDataError) or attempt == net.ATTEMPTS - 1:
                raise net.OpenDataError(f"could not download {url}: {e}") from None
            _http._sleep(2.0 * (attempt + 1))
    raise net.OpenDataError(url)  # pragma: no cover


def _new_db(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(str(path))
    c.execute("CREATE TABLE entries (sha TEXT NOT NULL, name TEXT NOT NULL, seq TEXT NOT NULL, data TEXT NOT NULL)")
    return c


def _finish(c: sqlite3.Connection) -> int:
    c.execute("CREATE INDEX entries_sha ON entries (sha)")
    n = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    c.commit()
    c.close()
    return int(n)


def _build_openfold3(raw: Dict[str, Path], out: Path) -> int:
    c = _new_db(out)
    rows = []
    with open(raw["fasta"], encoding="utf-8", errors="replace") as fh:
        name: Optional[str] = None
        parts: List[str] = []
        for line in list(fh) + [">"]:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    q = _canon("".join(parts))
                    if q and not set(q) <= set("ACGUTNX"):
                        rows.append((seq_sha256(q), name, q, "{}"))
                name, parts = line[1:].split()[0] if line[1:].strip() else None, []
            elif line:
                parts.append(line)
    c.executemany("INSERT INTO entries VALUES (?,?,?,?)", rows)
    return _finish(c)


def _build_litefold(raw: Dict[str, Path], out: Path) -> int:
    c = _new_db(out)
    rows = []
    with open(raw["manifest"], encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            q = _canon(r.get("sequence") or "")
            if not q or not r.get("a3m_path"):
                continue
            data = {"path": r["a3m_path"], "bytes": int(r.get("a3m_bytes") or 0), "entry": r.get("entry", "")}
            rows.append((seq_sha256(q), r["acc"], q, json.dumps(data)))
    c.executemany("INSERT INTO entries VALUES (?,?,?,?)", rows)
    return _finish(c)


def _build_openproteinset(raw: Dict[str, Path], out: Path) -> int:
    chain_sha: Dict[str, str] = {}
    seqs: Dict[str, str] = {}
    with open(raw["chain_cache"], encoding="utf-8") as fh:
        for chain, rec in iter_json_object(fh):
            s = rec.get("seq") if isinstance(rec, dict) else None
            q = _canon(s) if isinstance(s, str) else None
            if q:
                sha = seq_sha256(q)
                chain_sha[chain] = sha
                seqs.setdefault(sha, q)
    c = _new_db(out)
    done: set = set()
    rows = []
    with open(raw["alignment_index"], encoding="utf-8") as fh:
        for chain, rec in iter_json_object(fh):
            got = chain_sha.get(chain)
            if got is None or got in done or not isinstance(rec, dict):
                continue
            sha = got
            files = [f for f in rec.get("files", []) if isinstance(f, list) and len(f) == 3
                     and str(f[0]).endswith(".a3m") and int(f[2]) > 0]
            if not files or not isinstance(rec.get("db"), str):
                continue
            done.add(sha)
            rows.append((sha, chain, seqs[sha], json.dumps({"db": rec["db"], "files": files})))
    c.executemany("INSERT INTO entries VALUES (?,?,?,?)", rows)
    return _finish(c)


_BUILDERS = {"openfold3": _build_openfold3, "litefold": _build_litefold, "openproteinset": _build_openproteinset}


def iter_json_object(fh: "io.TextIOBase | Any", chunk: int = 1 << 20) -> Iterator[Tuple[str, Any]]:
    """(key, value) of a top-level JSON object, read a chunk at a time, so a
    file of several hundred MB never has to be parsed in one piece."""
    dec = json.JSONDecoder()
    buf = fh.read(chunk)
    eof = not buf
    pos = 0

    def more() -> bool:
        nonlocal buf, pos, eof
        if eof:
            return False
        data = fh.read(chunk)
        if not data:
            eof = True
            return False
        buf = buf[pos:] + data
        pos = 0
        return True

    def skip_ws() -> None:
        nonlocal pos
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n":
                pos += 1
            if pos < len(buf) or not more():
                return

    def value() -> Any:
        nonlocal pos
        while True:
            try:
                v, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if more():
                    continue
                raise
            if end >= len(buf) and more():
                continue  # a number may go on in the next chunk
            pos = end
            return v

    skip_ws()
    if pos >= len(buf) or buf[pos] != "{":
        raise ValueError("not a JSON object")
    pos += 1
    while True:
        skip_ws()
        if pos < len(buf) and buf[pos] == ",":
            pos += 1
            skip_ws()
        if pos < len(buf) and buf[pos] == "}":
            return
        if pos >= len(buf):
            raise ValueError("JSON object ends early")
        k = value()
        skip_ws()
        if pos >= len(buf) or buf[pos] != ":":
            raise ValueError("JSON object key without ':'")
        pos += 1
        skip_ws()
        yield str(k), value()
        if pos > chunk:
            buf = buf[pos:]
            pos = 0
