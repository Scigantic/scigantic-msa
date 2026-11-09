"""The one-shot tool: get an alignment for a sequence, or read one you have.

    from scigantic_msa import MSA
    result = MSA().get(sequence, remote=True)        # open data + the ColabFold server
    result = MSA(server=None).get(sequence)          # open data only; the sequence is not sent
    result = MSA().from_a3m("my_alignment.a3m")      # no network

``get`` sends the sequence to a third-party search server, so it refuses to
run unless you pass ``remote=True``. It also looks the sequence up in open
data: the published alignments of OpenFold3, LiteFold and OpenProteinSet,
read where their publishers put them. That lookup never sends the
sequence. It downloads the tables, and asks for the public files of the
entries that matched, so those hosts learn which public entries you asked
for. With a search server it also sends the sequence's MD5 to UniProt
(``uniparc``); that is off for open data only, since the MD5 of a close
variant of a public protein can be reversed by trying the variants. The
server's rows and the open rows are merged into one alignment per chain,
with every row's source kept.

It checks the local cache first, so running the same sequence again does
not submit it again. Nothing here talks to a Scigantic server. There is no
account, no login and no upload.
"""
from __future__ import annotations

import atexit
import gzip
import hashlib
import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
import urllib.parse
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

from . import http
from .http import RemoteNotAllowed
from .a3m import A3MError, Row, parse_a3m, parse_stockholm
from .alignment import _UNIPROT, Alignment, species_key
from .cache import AlignmentCache, BlobCache, LockTimeout, cache_dir as default_cache_dir
from .pairing import ComplexPairing, pair_by_species
from .remote import BACKENDS, announce, make_backend, remote_disabled
from .seqkey import InvalidSequence, canonical_sequence, seq_sha256

log = logging.getLogger("scigantic_msa")

__all__ = ["MSA", "RemoteNotAllowed", "SERVERS", "OPEN_PAIRS"]

#: What each server is, for the messages that say where a sequence goes.
SERVERS: Dict[str, str] = {
    "colabfold": "the ColabFold MMseqs2 server, run by the ColabFold team. Results there can be "
                 "downloaded by anyone who knows the sequence",
    "nvidia": "NVIDIA's hosted MSA Search (needs NVIDIA_API_KEY), under NVIDIA's API trial terms",
}

#: How extra pairs by species from open data are added to a complex:
#: auto (only when every chain's organism is known and they share one),
#: always, or never.
OPEN_PAIRS = ("auto", "always", "never")


#: The cache warning is said once per process, not once per MSA object.
_cache_warned = False

#: Without the cache, open data goes to one temporary folder per process.
_temp: Dict[str, Path] = {}
_temp_lock = threading.Lock()


def _temp_base() -> Path:
    """The temporary folder for open data without the cache, one per
    process, so the tables are downloaded once per process and not once per
    MSA object. Removed at exit, and on SIGTERM when nothing else handles
    that signal."""
    with _temp_lock:
        p = _temp.get("base")
        if p is None or not p.is_dir():
            p = Path(tempfile.mkdtemp(prefix="scigantic-msa-"))
            _temp["base"] = p
            atexit.register(shutil.rmtree, str(p), True)
            _remove_on_sigterm()
        return p


def _remove_on_sigterm() -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
            return  # someone else handles it
    except (AttributeError, ValueError):
        return

    def handler(signum: int, frame: Any) -> None:
        p = _temp.get("base")
        if p is not None:
            shutil.rmtree(str(p), True)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, handler)
    except (OSError, ValueError):
        pass


class MSA:
    """Get an alignment for a protein sequence, once.

    server: "colabfold" (default), "nvidia", or None for open data only (the
        sequence is sent to no search server).
    host: another URL for that server, for example your own ColabFold
        MMseqs2 server. Default: api.colabfold.com, or
        ``SCIGANTIC_MSA_COLABFOLD_URL`` / ``SCIGANTIC_MSA_NVIDIA_URL``.
    open_data: merge the published open alignments of the sequence (default
        on). False gives the server's result alone.
    openproteinset: also use OpenProteinSet's PDB alignments, which needs
        about 610 MB of tables the first time (default off).
    near: also use entries at 95% identity or more over 90% of the query,
        found with ``mmseqs`` on this machine (default off).
    open_pairs: for a complex, when to add pairs by species from open rows
        to the server's own pairs: "auto" (default; only when the chains
        share an organism), "always" or "never".
    uniparc: send the sequence's MD5 to UniProt's UniParc to find the
        organisms that have exactly this sequence (used to pair a complex)
        and its UniClust30 entries. Default: on with a search server, off
        with ``server=None``, because the MD5 of a close variant of a public
        protein can be reversed by trying the variants.
    cache: keep results in a local cache (default on) so the same sequence
        is not submitted twice. ``cache_dir`` picks the directory. Without
        it, open data goes to a temporary folder: the tables are shared by
        the MSA objects of one process, and each object's files are removed
        by ``close()`` (or a ``with`` block), when the object is collected,
        and at exit.
    quiet: do not print progress lines to stderr.
    backend: a ready client object, mainly for tests.
    """

    def __init__(
        self,
        server: Optional[str] = "colabfold",
        *,
        host: Optional[str] = None,
        cache: bool = True,
        cache_dir: Union[str, Path, None] = None,
        quiet: bool = False,
        backend: Any = None,
        open_data: bool = True,
        openproteinset: bool = False,
        near: bool = False,
        open_pairs: str = "auto",
        uniparc: Optional[bool] = None,
        depth_summary: bool = True,
    ):
        if open_pairs not in OPEN_PAIRS:
            raise ValueError(f"open_pairs must be one of {', '.join(OPEN_PAIRS)}, not {open_pairs!r}")
        if backend is None and server is not None:
            if server not in BACKENDS:
                raise ValueError(f"unknown server {server!r}; choose one of {', '.join(BACKENDS)}, or None for open data only")
            kw: Dict[str, Any] = {}
            if host:
                check_host(host)
                kw["host" if server == "colabfold" else "url"] = host.strip()
            else:
                env = _HOST_ENV[server]
                if os.environ.get(env):
                    check_host(os.environ[env], env)
            backend = make_backend(server, **kw)
        if backend is None and not open_data:
            raise ValueError("with server=None there is nothing to search: keep open_data=True")
        self.backend = backend
        self.server = backend.name if backend is not None else None
        self.quiet = quiet
        self.open_pairs = open_pairs
        self.depth_summary = depth_summary
        base = Path(cache_dir).expanduser() if cache_dir else None
        self._cache = AlignmentCache(base) if cache else None
        self._pairs = BlobCache(base, "pairs") if cache else None
        self.open: Any = None
        self._finalizer: Any = None
        if open_data:
            from .opendata.lookup import OpenData

            if uniparc is None:
                uniparc = backend is not None
            work = None
            if cache:
                obase = base or default_cache_dir()
            else:
                obase = _temp_base()
                work = Path(tempfile.mkdtemp(prefix="run-", dir=str(obase)))
                self._finalizer = weakref.finalize(self, shutil.rmtree, str(work), True)
            self.open = OpenData(obase, openproteinset=openproteinset, near=near, uniparc=bool(uniparc),
                                 say=self._say, work=work, persistent=cache)

    def close(self) -> None:
        """Close the open data tables and, without the cache, remove the
        files this object read. The object can still be used after."""
        if self.open is not None:
            self.open.tables.close()
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self) -> "MSA":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ info
    @property
    def server_name(self) -> str:
        """The host that receives the sequence, e.g. api.colabfold.com ("" for open data only)."""
        return str(self.backend.server_name) if self.backend is not None else ""

    def describe_server(self) -> str:
        if self.backend is None:
            return "no search server (open data only)"
        if not _is_default_host(self.backend):
            return f"{self.server_name} (a {self.server} server you chose, not the public one)"
        return f"{self.server_name} ({SERVERS.get(str(self.server), str(self.server))})"

    def describe_open(self) -> str:
        """What the open data lookup sends, for the messages before a run.
        It names the same requests as the record of what was sent after."""
        if self.open is None:
            return ""
        tids = ", ".join(self.open.table_ids)
        if remote_disabled():
            return (f"open data from the cache only: the tables of {tids} and the files already read there "
                    "(SCIGANTIC_MSA_NO_REMOTE is set, so nothing is sent)")
        when = "once, then checked once a day" if self.open.tables.persistent else "for this process only"
        what = (f"open data: the sequence is looked up on this machine in the published tables of {tids} "
                f"(downloaded from their publishers {when}). It sends requests for those tables, and "
                "requests for the public files of matching entries to their publishers")
        if self.open.uniparc:
            what += ("; the MD5 of the sequence to rest.uniprot.org (UniParc), to find the organisms that have "
                     "it; and up to 25 UniProt accessions of it to openfold.s3.amazonaws.com (UniClust30 entries)")
        return what + ". The sequence itself is not sent"

    @property
    def cache_dir(self) -> Optional[Path]:
        return self._cache.root.parent if self._cache is not None else None

    def cached(self, sequence: str) -> Optional[Alignment]:
        """The cached server result for this sequence and server, or None. Sends nothing."""
        if self._cache is None or self.backend is None:
            return None
        q = canonical_sequence(sequence)
        return self._cache_get(seq_sha256(q))

    def pair_pending(self, sequences: List[str]) -> bool:
        """True when a complex of these chains needs a paired search the
        cache does not hold (the server is sent every chain once more)."""
        qs = list(dict.fromkeys(canonical_sequence(s) for s in sequences))
        if len(qs) < 2 or self.backend is None or not hasattr(self.backend, "pair_search"):
            return False
        return self._pair_cached(qs) is None

    def _cache_get(self, sha: str) -> Optional[Alignment]:
        if self._cache is None or self.backend is None:
            return None
        try:
            return self._cache.get(sha, self.backend.name, self.backend.dbkey)
        except OSError as e:
            self._drop_cache(e)
            return None

    def _drop_cache(self, e: OSError) -> None:
        """Caching is optional: a cache directory that cannot be used (read
        only, or a path that is a file) must not stop a search. Say so once
        and carry on without it."""
        if self._cache is None:
            return
        where = self._cache.root.parent
        self._cache = None
        self._pairs = None
        global _cache_warned
        if _cache_warned:
            return
        _cache_warned = True
        msg = (f"cannot use the cache at {where} ({e.strerror or type(e).__name__}); continuing without it. "
               "Pass --no-cache (cache=False in Python) to skip it.")
        log.info(msg)
        print(f"scigantic-msa: warning: {msg}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------ get
    @overload
    def get(self, sequence: str, *, remote: bool = False) -> Alignment: ...

    @overload
    def get(self, sequence: Union[List[str], Tuple[str, ...]], *, remote: bool = False) -> List[Alignment]: ...

    def get(self, sequence: Union[str, List[str], Tuple[str, ...]], *,
            remote: bool = False) -> Union[Alignment, List[Alignment]]:
        """The alignment for one sequence, or a list of alignments for the
        chains of a complex (pass a list; identical chains are searched once,
        and the distinct chains are paired).

        With a search server, remote must be True: the sequence is sent to
        the server named in the error otherwise. A cached result is used
        without sending anything. With ``server=None`` the sequence does not
        leave the machine, so ``remote`` is not needed; requests for public
        tables and files still do (see ``describe_open``).
        """
        single = isinstance(sequence, str)
        if not single and (isinstance(sequence, (bytes, bytearray)) or not isinstance(sequence, (list, tuple))):
            raise TypeError(f"pass the sequence as a str, or a list of str for a complex, "
                            f"not {type(sequence).__name__}")
        seqs: List[str] = [sequence] if isinstance(sequence, str) else list(sequence)
        if not seqs:
            raise ValueError("no sequence given")
        queries = [protein_sequence(s, i) for i, s in enumerate(seqs, 1)]
        distinct = list(dict.fromkeys(queries))
        if self.backend is not None:
            if remote is not True:
                raise RemoteNotAllowed(
                    f"MSA().get() sends the sequence to {self.describe_server()}. "
                    "Pass remote=True to allow that, use MSA(server=None) for open data only, or "
                    "MSA().from_a3m() with an alignment you already have."
                )
            if remote_disabled():
                # A cached result sends nothing, so the kill switch allows it.
                if any(self._cache_get(seq_sha256(q)) is None for q in distinct):
                    raise RemoteNotAllowed(http.NO_REMOTE_MESSAGE)
        names = _distinct_names(queries)
        server: Dict[str, Alignment] = {}
        out: Dict[str, Alignment] = {}
        opened: Dict[str, Any] = {}
        for i, q in enumerate(distinct):
            what = f"chain {names[i]}" if len(distinct) > 1 or len(queries) > 1 else "sequence"
            if self.backend is not None:
                server[q] = self._one(q, what)
            if self.open is not None:
                opened[q] = self.open.collect(q)
                for n in opened[q].notes:
                    self._say(f"{what}: open data: {n}")
            out[q] = self._merge(q, server.get(q), opened.get(q), what)
        if len(distinct) > 1:
            self._pair(distinct, names, out, server, opened)
        res = [out[q] for q in queries]
        return res[0] if single else res

    def _one(self, q: str, what: str) -> Alignment:
        sha = seq_sha256(q)
        b = self.backend
        hit = self._cache_get(sha)
        if hit is not None:
            self._say(f"{what}: {hit.summary()} (nothing sent)")
            return hit
        lock = self._cache.lock(sha, b.name, b.dbkey) if self._cache is not None else None
        if lock is not None:
            try:
                lock.acquire()
            except LockTimeout:
                raise
            except OSError as e:
                lock = None
                self._drop_cache(e)
        try:
            if lock is not None:
                again = self._cache_get(sha)
                if again is not None:
                    self._say(f"{what}: computed by another process while waiting")
                    return again
            if remote_disabled():
                raise RemoteNotAllowed(http.NO_REMOTE_MESSAGE)
            # A search that cannot start (no API key) must not say it is sending.
            check = getattr(b, "check_ready", None)
            if check is not None:
                check()
            if not self.quiet:
                announce(b)
            res = b.search(q)
            m = Alignment(query=q, rows=res.rows, source="remote", provenance=dict(res.provenance),
                          templates=res.templates, attribution=[_attribution(b)])
            if self._cache is not None:
                try:
                    m.cache_path = str(self._cache.put(m, b.name, b.dbkey))
                except OSError as e:
                    self._drop_cache(e)
            self._say(f"{what}: {m.summary()}")
            return m
        finally:
            if lock is not None:
                lock.release()

    # ------------------------------------------------------------ open data
    def _merge(self, q: str, srv: Optional[Alignment], od: Any, what: str) -> Alignment:
        if od is None or (srv is not None and not od.rows):
            # Open data had nothing for this sequence: the server's result as it is.
            assert srv is not None
            if od is not None:
                srv.provenance = dict(srv.provenance, open=_open_record(od),
                                      organisms={k: sorted(v) for k, v in od.organisms.items()})
            return srv
        from .opendata.lookup import Prov
        from .opendata.merge import merge_rows

        server_rows: List[Row] = []
        if srv is not None:
            tid = str(srv.provenance.get("ticket") or srv.provenance.get("server") or "")
            for i, r in enumerate(srv.rows[1:], 1):
                r.extra["prov"] = Prov(self.server or "server", r.db, tid, "", i)
                server_rows.append(r)
        rows, counts = merge_rows(q, server_rows, od.rows)
        qrow = srv.rows[0] if srv is not None else Row("query", q, "query")
        prov: Dict[str, Any] = dict(srv.provenance) if srv is not None else {}
        prov["from"] = "merged" if srv is not None else "open"
        prov["open"] = _open_record(od)
        prov["merge"] = counts.as_dict()
        prov["organisms"] = {k: sorted(v) for k, v in od.organisms.items()}
        att = list(srv.attribution) if srv is not None else []
        att += _open_attribution(od, counts)
        m = Alignment(query=q, rows=[qrow] + rows, source="merged" if srv is not None else "open",
                      provenance=prov, attribution=att, templates=srv.templates if srv is not None else None)
        m.cache_path = srv.cache_path if srv is not None else None
        if self.depth_summary:
            m.provenance["depth"] = depth_record(m, server_rows if srv is not None else None)
        return m

    # ------------------------------------------------------------ complexes
    def _pair_key(self, qs: List[str]) -> str:
        b = self.backend
        return hashlib.sha256(("|".join(seq_sha256(q) for q in qs) + "|" + b.name + "|" + b.pair_dbkey).encode()).hexdigest()

    def _pair_cached(self, qs: List[str]) -> Optional[bytes]:
        if self._pairs is None:
            return None
        return self._pairs.get(self._pair_key(qs), "pair.tar.gz")

    def _server_pairs(self, qs: List[str], names: List[str]) -> Any:
        """The server's paired search for these chains, or None when the
        server has none or it is not allowed now."""
        from .remote.colabfold import parse_pair_result

        b = self.backend
        if b is None or not hasattr(b, "pair_search"):
            return None
        raw = self._pair_cached(qs)
        if raw is not None:
            self._say(f"paired search of chains {', '.join(names)}: from the cache (nothing sent)")
            return parse_pair_result(raw, qs, "cache", getattr(b, "host", ""))
        if remote_disabled():
            self._say("paired search: not in the cache and SCIGANTIC_MSA_NO_REMOTE is set, so the chains "
                      "are paired by species instead")
            return None
        if not self.quiet:
            announce(b)
        import io
        import tarfile

        from .remote.colabfold import RemoteError

        try:
            res = b.pair_search(qs)
        except RemoteNotAllowed:
            raise
        except RemoteError as e:
            print(f"scigantic-msa: warning: the paired search failed ({e}); the chains are paired by species "
                  "from the server's rows instead", file=sys.stderr, flush=True)
            return None
        if self._pairs is not None:
            # Keep pair.a3m only, as a small tar of its own.
            text = "\x00".join(
                f">{101 + c}\n{q}\n" + "".join(f">{r.header}\n{r.seq}\n" if r is not None else f">gap\n{'-' * len(q)}\n"
                                              for r in res.rows[c])
                for c, q in enumerate(qs)).encode()
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                info = tarfile.TarInfo("pair.a3m")
                info.size = len(text)
                tf.addfile(info, io.BytesIO(text))
            try:
                self._pairs.put(self._pair_key(qs), "pair.tar.gz", buf.getvalue())
            except OSError as e:
                self._drop_cache(e)
        self._say(f"paired search of chains {', '.join(names)}: {len(res.rows[0])} paired rows")
        return res

    def _pair(self, qs: List[str], names: List[str], out: Dict[str, Alignment], server: Dict[str, Alignment],
              opened: Dict[str, Any]) -> None:
        from .opendata.merge import combine_pairs, organism_rule
        from .opendata.sources import SOURCES as OPEN_SETS

        sp = self._server_pairs(qs, names)
        if sp is not None:
            from .opendata.lookup import Prov

            for c in range(len(qs)):
                for k, r in enumerate(sp.rows[c], 1):
                    if r is not None:
                        r.extra["prov"] = Prov(self.server or "server", "paired", "", "", k)
        base = None
        if sp is None and server:
            base = pair_by_species([server[q] for q in qs], names=names)
        add_open = False
        rule = ""
        if opened:
            if self.open_pairs == "never":
                rule = "extra species pairs from open data are turned off"
            elif self.open_pairs == "always":
                add_open, rule = True, "extra species pairs from open data are on for any organisms"
            else:
                add_open, rule = organism_rule([opened[q].species() for q in qs], names)
                if add_open:
                    rule = "extra species pairs from open data are added because " + rule
        # The pool for open pairs is the open rows the merge kept, so a pair
        # never brings back a copy the merge dropped.
        open_rows = [[Row("query", q, "query")]
                     + [r for r in out[q].rows[1:] if getattr(r.extra.get("prov"), "source", None) in OPEN_SETS]
                     for q in qs]
        plan = combine_pairs(qs, [out[q].rows for q in qs], open_rows, names,
                             server_pairs=sp.rows if sp is not None else None, base=base,
                             add_open=add_open and bool(opened), rule=rule)
        cp = ComplexPairing(list(qs), plan.pairing, plan.server_pairs, plan.open_pairs, plan.rule)
        self._say(plan.pairing.message)
        for q in qs:
            m = out[q]
            m.complex_pairing = cp
            m.provenance = dict(m.provenance, pairing={"server_pairs": plan.server_pairs,
                                                       "open_pairs": plan.open_pairs, "rule": plan.rule,
                                                       "paired_search": ({k: v for k, v in sp.provenance.items() if k != "ticket"}
                                                                         if sp is not None else None)})

    # ------------------------------------------------------------ files
    def from_a3m(self, a3m: Union[str, bytes, os.PathLike], sequence: Optional[str] = None, *,
                 uniprot: Union[str, bytes, os.PathLike, None] = None) -> Alignment:
        """Read an alignment you already have: a path to an ``.a3m`` (or
        ``.a3m.gz``, or Stockholm ``.sto``) file, or the A3M text itself.
        Nothing is sent anywhere.

        With ``sequence``, the first record must be exactly that sequence.
        The whole file is checked: allowed characters only, every row as
        wide as the query, and no complex header (pass one file per chain).

        ``uniprot`` is an optional second alignment of the same chain against
        UniProt (such as AlphaFold 3's or OpenFold3's ``uniprot_hits``), with
        headers like ``tr|ACC|NAME_SPECIES``. Its rows are used to pair the
        chains of a complex: AlphaFold 3 gets them as ``pairedMsa``,
        OpenFold3 as ``uniprot_hits.a3m`` and Protenix as ``pairing.a3m``.
        """
        return read_alignment(a3m, sequence, uniprot=uniprot)

    def _say(self, msg: str) -> None:
        log.info(msg)
        if not self.quiet:
            print(f"scigantic-msa: {msg}", file=sys.stderr, flush=True)


def _distinct_names(queries: List[str]) -> List[str]:
    """For each distinct sequence, the chain ids that have it ("B, D")."""
    from .adapters import chain_id

    ids: Dict[str, List[str]] = {}
    for i, q in enumerate(queries):
        ids.setdefault(q, []).append(chain_id(i))
    return [", ".join(v) for v in ids.values()]


def _open_record(od: Any) -> Dict[str, Any]:
    """What the attribution file says about the open lookup of one chain."""
    objs = []
    for o in od.objects:
        objs.append({k: o.get(k) for k in ("source", "entry", "role", "url", "range", "anchor", "etag",
                                            "last_modified", "rows", "rows_used", "truncated", "consensus_identity", "error")
                     if o.get(k) is not None})
    return {"entries": od.entries, "objects": objs, "tables": od.tables, "uniparc": od.uniparc,
            "notes": od.notes, "sent": od.sent, "failed": od.failed, "unpublished": od.unpublished}


def _open_attribution(od: Any, counts: Any) -> List[Dict[str, Any]]:
    from .opendata.sources import SOURCES, licence_flags

    used = sorted({o["source"] for o in od.objects if not o.get("error") and o.get("rows_used")})
    out: List[Dict[str, Any]] = []
    tables = {t["table"]: t for t in od.tables}
    for sid in used:
        s = SOURCES[sid]
        a = {"id": sid, "title": s["title"], "license": s["license"], "license_url": s["license_url"],
             "attribution": s["attribution"], "citation": s["citation"], "where": s["where"],
             "databases": s["databases"]}
        tid = {"openfold3": "openfold3", "litefold": "litefold", "openproteinset": "openproteinset"}.get(sid)
        if tid in tables:
            t = tables[tid]
            a["table"] = {"entries": t["entries"], "files": t["files"]}
        flags = licence_flags([sid])
        if flags:
            a["licence_flags"] = flags
        out.append(a)
    if od.uniparc is not None:
        out.append({"id": "uniprot-uniparc", "title": "UniProt UniParc (organisms of the exact sequence, by MD5)",
                    "license": "CC-BY-4.0", "release": od.uniparc.get("release"),
                    "citation": "The UniProt Consortium, UniProt: the Universal Protein Knowledgebase, "
                                "Nucleic Acids Research"})
    return out


def depth_record(m: Alignment, server_rows: Optional[List[Row]] = None) -> Dict[str, Any]:
    """Rows, an estimated Neff at 80% identity and species, for the whole
    alignment and for the server's own result alone."""
    from .opendata.merge import neff_estimate

    srv = list(server_rows or [])
    kept_srv = sum(1 for r in m.rows[1:] if getattr(r.extra.get("prov"), "source", None) in ("colabfold", "nvidia"))
    out: Dict[str, Any] = {"rows": len(m.rows) - 1, "server_rows": len(srv), "open_rows": len(m.rows) - 1 - kept_srv,
                           "neff80": round(neff_estimate(m.query, m.rows[1:]), 1),
                           "species": len({k for k in (species_key(r) for r in m.rows[1:]) if k})}
    if server_rows is not None:
        out["server_neff80"] = round(neff_estimate(m.query, srv), 1)
        out["server_species"] = len({k for k in (species_key(r) for r in srv) if k})
    return out



#: The environment variable that sets each server's URL.
_HOST_ENV = {"colabfold": "SCIGANTIC_MSA_COLABFOLD_URL", "nvidia": "SCIGANTIC_MSA_NVIDIA_URL"}


def check_host(host: str, what: str = "--host") -> None:
    """Refuse a server URL that is not http(s)://host. A missing scheme
    ("localhost:8080") gets an example of the right form. ``what`` names
    where the URL came from in the error. A URL with a user name or
    password is refused first, on the string as given, so no error repeats
    it."""
    http.refuse_credentials(host, what)
    host = host.strip()
    http.require_ascii(host, what)
    parts = urllib.parse.urlsplit(host)
    if parts.scheme.lower() in ("http", "https"):
        if not parts.netloc:
            raise ValueError(f"{what} {host!r} has no host name")
        if any(c.isspace() for c in host):
            raise ValueError(f"{what} {host!r} has a space in it")
        if parts.query or parts.fragment or "?" in host or "#" in host:
            raise ValueError(f"{what} {host!r} has a query or fragment; give only the server's URL")
        return
    if "://" in host:
        raise ValueError(f"{what} must start with http:// or https://, not {parts.scheme}://")
    raise ValueError(f"{what} needs http:// or https:// in front, for example http://{host}")


def _is_default_host(backend: Any) -> bool:
    from .remote import colabfold, nvidia

    if backend.name == "colabfold" and hasattr(backend, "host"):
        return colabfold.public_host(str(backend.host))
    if backend.name == "nvidia" and hasattr(backend, "url"):
        return str(backend.url) == nvidia.DEFAULT_URL
    return True


def protein_sequence(seq: str, i: int) -> str:
    """Canonical sequence, refusing anything that is not letters (a mistyped
    file name, say) before it can be sent. The error names only the
    offending characters."""
    try:
        q = canonical_sequence(seq)
    except InvalidSequence as e:
        raise InvalidSequence(f"sequence {i}: {e}") from None
    bad = sorted({c for c in q if not ("A" <= c <= "Z")})
    if bad:
        hint = ". For a complex, pass a list with one sequence per chain" if ":" in bad else ""
        raise InvalidSequence(f"sequence {i} has characters that are not amino acid letters: {''.join(bad)!r}{hint}")
    return q


def read_alignment(a3m: Union[str, bytes, os.PathLike], sequence: Optional[str] = None, *,
                   uniprot: Union[str, bytes, os.PathLike, None] = None) -> Alignment:
    name = None
    data: Union[str, bytes]
    if isinstance(a3m, os.PathLike) or (isinstance(a3m, str) and "\n" not in a3m and "\r" not in a3m
                                      and not a3m.lstrip("\ufeff \t").startswith(">")):
        p = Path(a3m)
        if not p.is_file():
            # A str reaches here when it has no newline and does not start
            # with ">", which is the only way to tell "a path" from "A3M
            # text". Say which branch was taken and why, because the obvious
            # reading of a FileNotFoundError is that the caller mistyped a
            # path, and for a short or empty string the caller passed content.
            # Path("") is ".", so the old message claimed the working
            # directory did not exist.
            what = f"{a3m!r}" if isinstance(a3m, str) else str(p)
            if isinstance(a3m, str) and not a3m.strip():
                raise A3MError(
                    "from_a3m was given an empty string. Pass A3M text "
                    "(starting with '>') or a path to a file."
                )
            if p.is_dir():
                raise A3MError(f"{p} is a directory, not an alignment file")
            raise FileNotFoundError(
                f"{what} was read as a file path, because it has no newline "
                "and does not start with '>', and no such file exists. If "
                "that was meant to be A3M text, it is missing its '>' header."
            )
        name = p.name
        raw = p.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError) as e:
                raise A3MError(f"{p.name}: not a readable gzip file ({e})") from None
        if not raw.strip():
            raise A3MError(f"{p.name} is empty")
        data = raw
    else:
        data = a3m
    what = name or "a3m"
    if isinstance(data, bytes) and data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    elif isinstance(data, str):
        data = data.lstrip("\ufeff")
    head = data[:20] if isinstance(data, bytes) else data[:20].encode()
    if head.lstrip().startswith(b"# STOCKHOLM"):
        try:
            text = data.decode("utf-8") if isinstance(data, bytes) else data
        except UnicodeDecodeError as e:
            raise A3MError(f"{what}: not UTF-8 text ({e})") from None
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        rows = parse_stockholm(text.split("\n"), sequence, what=what)
    else:
        rows = parse_a3m(data, sequence, what=what)
    prov = {"from": "file"}
    if name:
        prov["file"] = name
    uniprot_rows = None
    if uniprot is not None:
        uniprot_rows, uni_name = _uniprot_rows(uniprot, rows[0].seq, what)
        if uni_name:
            prov["uniprot_file"] = uni_name
    return Alignment(query=rows[0].seq, rows=rows, source="file", provenance=prov, uniprot_rows=uniprot_rows)


def _uniprot_rows(uniprot: Union[str, bytes, os.PathLike], query: str, what: str) -> "tuple[list, Optional[str]]":
    """The rows of a UniProt alignment for the chain whose query is
    ``query``, marked as UniProt rows. At least one row must carry a UniProt
    species code in its header, the only species AlphaFold 3 pairs on."""
    try:
        uni = read_alignment(uniprot, query)
    except A3MError as e:
        raise A3MError(f"UniProt alignment for {what}: {e}") from None
    if not any(_UNIPROT.match(r.header.split(None, 1)[0].partition("/")[0]) for r in uni.rows[1:] if r.header.strip()):
        where = uni.provenance.get("file") or "the UniProt alignment"
        raise A3MError(f"{where} has no row with a UniProt header such as tr|ACC|NAME_SPECIES, "
                       "so no model could pair on it")
    for r in uni.rows[1:]:
        r.db = "uniprot"
    return uni.rows, uni.provenance.get("file")


def _attribution(backend: Any) -> Dict[str, Any]:
    if backend.name == "colabfold":
        return {"id": "colabfold-api", "title": "ColabFold MMseqs2 server", "server": backend.server_name,
                "license": "see the server's terms",
                "citation": "Mirdita et al., ColabFold: making protein folding accessible to all, Nature Methods 2022"}
    return {"id": "nvidia-msa-search", "title": "NVIDIA MSA Search", "server": backend.server_name,
            "license": "NVIDIA API trial terms"}
