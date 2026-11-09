"""Find and read every published open alignment of a sequence.

For one sequence, :meth:`OpenData.collect`:

1. looks the sha256 of the sequence up in the local copies of the
   publishers' tables (OpenFold3, LiteFold and, opt in, OpenProteinSet).
   Nothing is sent for this step;
2. when ``uniparc`` is on, asks UniParc which UniProtKB entries have
   exactly this sequence, sending the MD5 of the sequence (UniProt's own
   sequence checksum). The answer names the organisms that have this
   sequence, and its accessions are tried as OpenProteinSet UniClust30
   entries;
3. optionally (``near=True``, needs ``mmseqs`` on PATH) searches the table
   sequences on this machine for entries at 95% identity or more covering
   90% of the query;
4. reads each matched entry's alignment files from where the publisher put
   them, keeps them in the local cache, and returns their rows on the
   query's columns, each row marked with where it came from.

What leaves the machine: requests for the tables, requests for the public
files of the entries that matched (which tell their hosts which public
entries were asked for), and with ``uniparc`` the MD5 to rest.uniprot.org
and UniProt accessions to openfold.s3.amazonaws.com. The sequence itself is
never sent. An MD5 is not a secret for a sequence close to a public one: a
point mutant of a known protein can be found from its MD5 by trying the
variants, so the UniParc lookup is off unless asked for in open-only mode.
Every request is recorded in ``OpenResult.sent`` before it is made, so a
request that fails is named too.
"""
from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import re
import sqlite3
import time
import zlib
from urllib.parse import urlsplit
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .. import http as _http
from ..a3m import Row
from ..cache import FileLock, atomic_write, private_dirs
from ..seqkey import canonical_sequence, seq_sha256
from . import net
from .anchor import identity_map, map_from_alignment, reanchor
from .parse import Parsed, SourceMismatch, parse_a3m_lines, parse_sto_lines
from .sources import BASES, LITEFOLD_COMMIT, LITEFOLD_REPO, ROLE_DB, SOURCES
from .tables import Tables

__all__ = ["OpenData", "OpenResult", "Prov", "species_name", "PARSER_VERSION"]

#: Changes when parsing changes, so cached objects are read again.
PARSER_VERSION = 1
#: Per file: stop reading after this many rows or bytes (rows are ordered
#: best hit first). OpenFold3's uniprot_hits.sto files reach 2.9 GB.
MAX_ROWS = 50_000
MAX_MB = 256
#: At most this many UniProtKB accessions of the sequence are tried as
#: UniClust30 entries.
MAX_UNICLUST_PROBES = 25
#: A UniParc answer is reused for this long.
UNIPARC_DAYS = 30

OF3_FILES = (("cfdb_uniref30", "a3m", "cfdb_uniref30.a3m"), ("uniref90", "sto", "uniref90_hits.sto"),
             ("mgnify", "sto", "mgnify_hits.sto"), ("uniprot", "sto", "uniprot_hits.sto"))
OPS_ROLES = {"uniref90_hits.a3m": "uniref90", "bfd_uniclust_hits.a3m": "bfd_uniclust", "mgnify_hits.a3m": "mgnify"}
#: Organisms that are not organisms.
NOT_ORGANISMS = {"32630", "32644"}  # synthetic construct, unidentified


@dataclass(frozen=True)
class Prov:
    """Where one row came from."""

    source: str        # openfold3, litefold, openproteinset, uniclust30, colabfold
    role: str          # file kind: uniprot, uniref90, mgnify, cfdb_uniref30, colabfold, uniref, env, ...
    entry: str         # the publisher's id of the entry (PDB chain, UniProt accession), or the ticket
    url: str           # the object read
    record: int        # 0-based record in that object
    anchor: str = "exact"


@dataclass
class FileRef:
    role: str
    fmt: str
    url: str
    offset: int = 0
    length: Optional[int] = None


@dataclass
class Entry:
    source: str
    entry: str
    seq: str
    files: List[FileRef]
    anchor: str = "exact"
    cmap: Optional[List[Optional[int]]] = None


@dataclass
class OpenResult:
    """The open rows of one sequence. ``rows`` exclude the query; each has
    ``extra['prov']``. ``organisms`` maps an evidence kind to the species
    names that have exactly this sequence."""

    query: str
    rows: List[Row] = field(default_factory=list)
    entries: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    organisms: Dict[str, Set[str]] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    uniparc: Optional[Dict[str, Any]] = None
    sent: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: What could not be searched or read (a table, an answer, a file), as
    #: opposed to not being in any table. Each is also in ``notes``.
    failed: List[str] = field(default_factory=list)
    #: Entries a table lists whose files are not published (HTTP 403/404).
    unpublished: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)
        self.notes.append(msg)

    def species(self) -> Set[str]:
        return set().union(*self.organisms.values()) if self.organisms else set()


_OS = re.compile(r"\bOS=(.+?)(?:\s+OX=|\s+GN=|\s+PE=|\s+SV=|$)")
_OX = re.compile(r"\bOX=(\d+)")


def species_name(name: str) -> Optional[str]:
    """The species part of an organism name: genus and epithet ("Escherichia
    coli" for "Escherichia coli (strain K12)"), three words after
    "Candidatus", and the whole name for viruses and phages, whose names are
    not binomials. None for names that are not an organism."""
    n = re.sub(r"\([^)]*\)", " ", name or "").strip()
    words = n.split()
    if not words or n.lower() in ("synthetic construct", "unidentified") or "metagenome" in n.lower():
        return None
    if re.search(r"virus|phage|viroid", n, re.I):
        return " ".join(words)
    if words[0] == "Candidatus":
        return " ".join(words[:3])
    if len(words) >= 2 and words[1] in ("sp.", "cf.", "aff."):
        return " ".join(words)
    return " ".join(words[:2])


class OpenData:
    """Open alignment lookups, with everything kept under ``cache_base/open``.

    ``openproteinset``: also use the OpenProteinSet PDB set (about 610 MB of
    tables the first time). ``near``: also take entries at 95% identity or
    more (needs ``mmseqs``). ``max_rows`` / ``max_mb``: per-file caps.
    ``uniparc``: look the sequence's MD5 up in UniParc (organisms and
    UniClust30 entries). ``work``: where the files read, UniParc answers
    and near-match searches are kept, when not under ``cache_base/open``
    (a folder of its own per object without the cache); ``persistent``:
    False when ``cache_base`` is a temporary folder.

    One object can be used from several threads.
    """

    def __init__(self, cache_base: Path, *, openproteinset: bool = False, near: bool = False,
                 uniparc: bool = True, max_rows: int = MAX_ROWS, max_mb: int = MAX_MB,
                 say: Callable[[str], None] = lambda m: None, work: Optional[Path] = None,
                 persistent: bool = True):
        self.base = Path(cache_base)
        self.root = self.base / "open"
        self.work = Path(work) if work is not None else self.root
        self.openproteinset = openproteinset
        self.near = near
        self.uniparc = uniparc
        self.max_rows = max_rows
        self.max_bytes = max_mb * (1 << 20)
        self.say = say
        self.tables = Tables(self.root / "tables", self.base, say, persistent=persistent)

    @property
    def table_ids(self) -> List[str]:
        return ["openfold3", "litefold"] + (["openproteinset"] if self.openproteinset else [])

    # ------------------------------------------------------------ entries
    def _table(self, tid: str, sha: str, res: OpenResult) -> List[Dict[str, Any]]:
        try:
            got = self.tables.lookup(tid, sha, res.sent)
        except (net.OpenDataError, OSError, ValueError, sqlite3.Error) as e:
            msg = f"the {tid} table could not be read ({e})"
            if msg not in res.failed:
                res.fail(msg)
            return []
        return got

    def _exact(self, q: str, sha: str, res: OpenResult) -> List[Entry]:
        out: List[Entry] = []
        for e in self._table("openfold3", sha, res)[:1]:
            base = f"{BASES['of3']}/msas/pdb_msas_completed/{e['name']}/"
            out.append(Entry("openfold3", e["name"], q, [FileRef(r, f, base + n) for r, f, n in OF3_FILES]))
        for e in self._table("litefold", sha, res)[:1]:
            # Read whole, not by the manifest's byte count: a count smaller
            # than the file would cut the alignment without a sign.
            url = f"{BASES['hf']}/datasets/{LITEFOLD_REPO}/resolve/{LITEFOLD_COMMIT}/{e['path']}"
            out.append(Entry("litefold", e["name"], q, [FileRef("colabfold", "a3m", url)]))
        if self.openproteinset:
            for e in self._table("openproteinset", sha, res)[:1]:
                url = f"{BASES['ops']}/alignment_db/{e['db']}"
                files = [FileRef(OPS_ROLES[n], "a3m", url, int(a), int(b)) for n, a, b in e["files"] if n in OPS_ROLES]
                out.append(Entry("openproteinset", e["name"], q, files))
        return out

    # ------------------------------------------------------------ UniParc
    def _uniparc(self, q: str, res: OpenResult) -> Optional[Dict[str, Any]]:
        md5 = hashlib.md5(q.encode()).hexdigest().upper()
        d = self.work / "uniparc"
        path = d / f"{md5}.json"
        got: Optional[Dict[str, Any]]
        try:
            got = json.loads(path.read_text())
            if not isinstance(got, dict):
                raise ValueError("not an answer")
            if time.time() - float(got.get("checked_at", 0)) < UNIPARC_DAYS * 86400 or _http.remote_disabled():
                return dict(got)
        except (OSError, ValueError, TypeError):
            got = None
        if _http.remote_disabled():
            return None
        url = (f"{BASES['uniprot']}/uniparc/search?query=checksum:{md5}"
               "&fields=upi,accession,organism&format=json&size=10")
        # Recorded before it is sent: a request that fails was still sent.
        msg = "the MD5 of the sequence to rest.uniprot.org (UniParc)"
        if msg not in res.sent:
            res.sent.append(msg)
        try:
            body, h = net.get(url, timeout=30)
            data = json.loads(body)
            out = _uniparc_answer(data, md5)
        except (net.OpenDataError, ValueError) as e:
            res.fail(f"the UniParc lookup failed ({e}); organisms and UniClust30 entries are not known")
            return got
        out["release"] = h.get("x-uniprot-release")
        private_dirs(d, self.base)
        atomic_write(path, json.dumps(out).encode(), mode=0o600)
        return out

    def _uniclust30(self, q: str, up: Dict[str, Any], res: OpenResult) -> List[Entry]:
        found = up.get("uniclust30")
        if found is None:
            if _http.remote_disabled():
                return []
            found = []
            accs = sorted(up.get("accessions") or [], key=lambda a: (not re.match(r"^[OPQ]", a), a))
            asked = 0
            try:
                for acc in accs[:MAX_UNICLUST_PROBES]:
                    url = f"{BASES['ops']}/uniclust30/{acc}/a3m/uniclust30.a3m"
                    asked += 1
                    try:
                        h = net.head(url)
                    except net.OpenDataError as e:
                        res.fail(f"the UniClust30 entries could not be checked ({e})")
                        return []  # unknown: do not cache a negative answer
                    if h.status == 200:
                        found.append(acc)
            finally:
                if asked:
                    host = urlsplit(BASES["ops"]).netloc
                    res.sent.append(f"{asked} UniProt accession{'s' if asked != 1 else ''} of the sequence "
                                    f"to {host} (UniClust30 entries)")
            up["uniclust30"] = found
            d = self.work / "uniparc"
            private_dirs(d, self.base)
            atomic_write(d / f"{up['md5']}.json", json.dumps(up).encode(), mode=0o600)
        return [Entry("uniclust30", acc, q,
                      [FileRef("uniclust30", "a3m", f"{BASES['ops']}/uniclust30/{acc}/a3m/uniclust30.a3m")])
                for acc in found[:1]]

    # ------------------------------------------------------------ near
    def _near(self, q: str, sha: str, res: OpenResult) -> List[Entry]:
        from . import near

        try:
            hits = near.search(q, self.tables, [t for t in self.table_ids], self.work / "near", self.base)
        except near.NearUnavailable as e:
            res.notes.append(str(e))
            return []
        out: List[Entry] = []
        for h in hits:
            if h.entry_sha == sha:
                continue
            cmap = map_from_alignment(h.qaln, h.taln, h.qstart, h.tstart, len(h.entry_seq))
            tag = f"near:{h.entry_sha[:12]}:id={h.fident:.3f}:qcov={h.qcov:.3f}"
            for ent in self._exact(h.entry_seq, h.entry_sha, res):
                ent.anchor, ent.cmap = tag, cmap
                out.append(ent)
        return out

    # ------------------------------------------------------------ collect
    def collect(self, sequence: str) -> OpenResult:
        q = canonical_sequence(sequence)
        sha = seq_sha256(q)
        res = OpenResult(query=q)
        entries = self._exact(q, sha, res)
        for tid in self.table_ids:
            t = self.tables.info(tid)
            if t is not None:
                res.tables.append(t.public())
            elif not any(f.startswith(f"the {tid} table") for f in res.failed):
                res.fail(f"the {tid} table is not in the cache, and SCIGANTIC_MSA_NO_REMOTE is set")
        up = self._uniparc(q, res) if self.uniparc else None
        res.uniparc = None if up is None else {k: up.get(k) for k in ("upis", "release", "accessions")}
        if up:
            names = {species_name(o["name"]) for o in up.get("organisms", [])} - {None}
            if names:
                res.organisms["uniparc"] = names  # type: ignore[assignment]
            entries += self._uniclust30(q, up, res)
        if self.near:
            entries += self._near(q, sha, res)
        for e in entries:
            if e.source == "litefold":
                res.organisms.setdefault("litefold", set()).add("Homo sapiens")
            res.entries.append({"source": e.source, "entry": e.entry, "anchor": e.anchor})
            n = len(res.objects)
            for f in e.files:
                self._read(e, f, res)
            mine = res.objects[n:]
            if mine and all(o.get("error") == "not published" for o in mine):
                res.unpublished.append(f"the {e.source} table lists {e.entry}, but none of its files is published")
        return res

    def _read(self, e: Entry, f: FileRef, res: OpenResult) -> None:
        rec: Dict[str, Any] = {"source": e.source, "entry": e.entry, "role": f.role, "url": f.url,
                               "anchor": e.anchor}
        if f.offset or f.length is not None:
            rec["range"] = [f.offset, f.length]
        def sending() -> None:
            # Recorded before the request, so one that fails is named too.
            msg = f"requests for the public files of {e.source} entry {e.entry} to {urlsplit(f.url).netloc}"
            if msg not in res.sent:
                res.sent.append(msg)

        try:
            parsed, meta = self.fetch(f, e.seq, on_send=sending)
        except net.NotFound:
            rec["error"] = "not published"
            res.objects.append(rec)
            return
        except SourceMismatch as ex:
            rec["error"] = str(ex)
            res.objects.append(rec)
            res.notes.append(f"{e.source} {e.entry} {f.role}: {ex}")
            return
        except (net.OpenDataError, _http.RemoteNotAllowed) as ex:
            rec["error"] = str(ex)
            res.objects.append(rec)
            res.fail(f"{e.source} {e.entry} {f.role}: {ex}")
            return
        rec.update({k: meta.get(k) for k in ("etag", "last_modified", "bytes_read", "truncated", "fetched_at",
                                             "from_cache")})
        rec["rows"] = len(parsed.rows) - 1
        rec["skipped"] = dict(parsed.skipped)
        cmap = e.cmap if e.cmap is not None else identity_map(len(e.seq))
        n = 0
        outside = 0
        for header, row, record in parsed.rows[1:]:
            new = row if e.cmap is None else reanchor(row, cmap, len(res.query))
            if new is None:
                outside += 1
                continue
            r = Row(header, new, ROLE_DB.get(f.role, "other"))
            r.extra["prov"] = Prov(e.source, f.role, e.entry, f.url, record, e.anchor)
            res.rows.append(r)
            n += 1
        if outside:
            rec["skipped"]["outside_query_after_reanchor"] = outside
        rec["rows_used"] = n
        res.objects.append(rec)
        if f.role == "uniprot" and e.cmap is None:
            names = set()
            for h in parsed.query_copies:
                m = _OS.search(h)
                s = species_name(m.group(1)) if m else None
                if s:
                    names.add(s)
            if names:
                res.organisms.setdefault("uniprot_hits", set()).update(names)

    # ------------------------------------------------------------ objects
    def _object_path(self, f: FileRef, seq: str) -> Path:
        k = hashlib.sha256(f"{PARSER_VERSION}|{f.url}|{f.offset}|{f.length}|{f.fmt}|{self.max_rows}|"
                           f"{self.max_bytes}|{seq_sha256(seq)}".encode()).hexdigest()
        return self.work / "objects" / k[:2] / f"{k}.a3m.gz"

    def _cached(self, path: Path, seq: str) -> Optional[Tuple[Parsed, Dict[str, Any]]]:
        try:
            p, m = _load(path, seq)
            return p, dict(m, from_cache=True)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError, zlib.error, UnicodeDecodeError):
            # A damaged copy is a miss, and is removed so it is read again.
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def fetch(self, f: FileRef, seq: str, on_send: Optional[Callable[[], None]] = None) -> Tuple[Parsed, Dict[str, Any]]:
        """The parsed rows of one file, from the cache or the network.
        ``on_send`` is called before the first request is made. Processes
        that want the same file wait for the first to read it."""
        path = self._object_path(f, seq)
        hit = self._cached(path, seq)
        if hit is not None:
            return hit
        if _http.remote_disabled():
            raise _http.RemoteNotAllowed(_http.NO_REMOTE_MESSAGE + "; this open file is not in the cache")
        lock = FileLock(path, timeout=3600, make_dir=lambda: private_dirs(path.parent, self.base))
        lock.acquire(on_wait=lambda: self.say(f"waiting for another process to read {_http._short(f.url)}"))
        try:
            hit = self._cached(path, seq)
            if hit is not None:
                return hit
            if on_send is not None:
                on_send()
            parsed, meta = self._download(f, seq)
            _store(path, parsed, meta, self.base)
        finally:
            lock.release()
        return parsed, dict(meta, from_cache=False)

    def _download(self, f: FileRef, seq: str) -> Tuple[Parsed, Dict[str, Any]]:
        headers = {}
        if f.offset or f.length is not None:
            end = "" if f.length is None else str(f.offset + f.length - 1)
            headers["Range"] = f"bytes={f.offset}-{end}"
        cap = self.max_bytes
        for attempt in range(net.ATTEMPTS):
            bodies: List[_Body] = []

            def reopen(bodies: List[_Body] = bodies) -> Iterable[bytes]:
                with net.stream(f.url, headers=headers, timeout=120) as (r2, h2):
                    b2 = _Body(r2, _want(h2, f))
                    bodies.append(b2)
                    yield from b2.lines()

            try:
                with net.stream(f.url, headers=headers, timeout=120) as (r, h):
                    body = _Body(r, _want(h, f))
                    bodies.append(body)
                    if f.fmt == "sto":
                        clen = h.get("content-length")
                        parsed = parse_sto_lines(body.lines(), seq, max_rows=self.max_rows, max_bytes=cap,
                                                 reopen=reopen, size=int(clen) if clen and clen.isdigit() else None)
                    else:
                        parsed = parse_a3m_lines(body.lines(), seq, max_rows=self.max_rows, max_bytes=cap)
                    last = bodies[-1]
                    if f.fmt == "sto" and last.eof and not last.closed:
                        # Read to the end, yet no '//': the file was cut.
                        raise ConnectionError("the Stockholm file ends without its closing '//'")
                    meta = {"url": f.url, "etag": net.etag_of(h), "last_modified": h.get("last-modified"),
                            "bytes_read": parsed.bytes_read, "truncated": parsed.truncated,
                            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    if f.length is not None and not parsed.truncated and parsed.bytes_read != f.length:
                        raise ConnectionError(f"read {parsed.bytes_read} of {f.length} bytes")
                return parsed, meta
            except (ConnectionError, TimeoutError, OSError, http.client.HTTPException) as e:
                if isinstance(e, net.OpenDataError) or attempt == net.ATTEMPTS - 1:
                    raise net.OpenDataError(f"could not read {f.url}: {e}") from None
                _http._sleep(2.0 * (attempt + 1))
        raise net.OpenDataError(f.url)  # pragma: no cover


def _want(h: Dict[str, str], f: FileRef) -> Optional[int]:
    """How many bytes the body must have: the Content-Length, else the
    length of the range asked for."""
    v = h.get("content-length")
    if v and v.isdigit():
        return int(v)
    return f.length


class _Body:
    """The lines of a response, checked at its end: a body shorter than the
    server said (HTTPResponse.readline gives b"" on an early end, with no
    error) raises ConnectionError. ``closed`` is set once a Stockholm
    ``//`` line has gone by."""

    def __init__(self, r: Any, want: Optional[int]):
        self.r = r
        self.want = want
        self.n = 0
        self.eof = False
        self.closed = False

    def lines(self) -> Iterator[bytes]:
        while True:
            line = self.r.readline(1 << 24)
            if not line:
                self.eof = True
                if self.want is not None and self.n < self.want:
                    raise ConnectionError(f"got {self.n} of {self.want} bytes")
                return
            self.n += len(line)
            if line.startswith(b"//"):
                self.closed = True
            yield line


def _uniparc_answer(data: Any, md5: str) -> Dict[str, Any]:
    """The parts of a UniParc search answer used here. ValueError for an
    answer of another shape."""
    if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
        raise ValueError("UniParc answered in an unexpected form")
    upis: List[str] = []
    accs: List[str] = []
    orgs: List[Dict[str, str]] = []
    for r in data.get("results", []):
        if not isinstance(r, dict):
            raise ValueError("UniParc answered in an unexpected form")
        if isinstance(r.get("uniParcId"), str):
            upis.append(r["uniParcId"])
        a = r.get("uniProtKBAccessions", [])
        o = r.get("organisms", [])
        if not isinstance(a, list) or not isinstance(o, list):
            raise ValueError("UniParc answered in an unexpected form")
        accs += [x.split(".")[0] for x in a if isinstance(x, str) and x]
        for org in o:
            if not isinstance(org, dict):
                continue
            tx = str(org.get("taxonId", ""))
            if tx and tx not in NOT_ORGANISMS:
                orgs.append({"taxid": tx, "name": str(org.get("scientificName", ""))})
    return {"md5": md5, "upis": upis, "accessions": sorted(set(accs)), "organisms": orgs,
            "checked_at": time.time(), "uniclust30": None}


def _store(path: Path, p: Parsed, meta: Dict[str, Any], base: Path) -> None:
    meta = dict(meta, skipped=dict(p.skipped), query_copies=p.query_copies, rows=len(p.rows))
    text = "#scigantic-msa-open " + json.dumps(meta, separators=(",", ":")) + "\n"
    text += "".join(f">{rec} {h}\n{s}\n" for h, s, rec in p.rows)
    private_dirs(path.parent, base)
    atomic_write(path, gzip.compress(text.encode("utf-8"), 4, mtime=0), mode=0o600)


def _load(path: Path, seq: str) -> Tuple[Parsed, Dict[str, Any]]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    first, _, body = text.partition("\n")
    if not first.startswith("#scigantic-msa-open "):
        raise ValueError("not a cached object")
    meta = json.loads(first[len("#scigantic-msa-open "):])
    rows: List[Tuple[str, str, int]] = []
    lines = body.split("\n")
    for i in range(0, len(lines) - 1, 2):
        rec, _, h = lines[i][1:].partition(" ")
        rows.append((h, lines[i + 1], int(rec)))
    if len(rows) != meta.get("rows") or not rows or rows[0][1] != seq:
        raise ValueError("cached object is damaged")
    p = Parsed(rows, Counter(meta.get("skipped") or {}), list(meta.get("query_copies") or []),
               bool(meta.get("truncated")), int(meta.get("bytes_read") or 0))
    return p, meta


def source_title(sid: str) -> str:
    return SOURCES.get(sid, {}).get("title", sid)


def describe_sent(results: Sequence[OpenResult]) -> List[str]:
    out: List[str] = []
    for r in results:
        for s in r.sent:
            if s not in out:
                out.append(s)
    return out
