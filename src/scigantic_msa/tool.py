"""The one-shot tool: get an alignment for a sequence, or read one you have.

    from scigantic_msa import MSA
    result = MSA().get(sequence, remote=True)        # asks the ColabFold server
    result = MSA().from_a3m("my_alignment.a3m")      # no network

``get`` sends the sequence to a third-party search server, so it refuses to
run unless you pass ``remote=True``. It checks the local cache first, so
running the same sequence again does not submit it again.

Nothing here talks to any other host. There is no account, no login and no
upload.
"""
from __future__ import annotations

import gzip
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .a3m import A3MError, parse_a3m, parse_stockholm
from .alignment import Alignment
from .cache import AlignmentCache
from .remote import BACKENDS, announce, make_backend, remote_disabled
from .seqkey import canonical_sequence, seq_sha256

log = logging.getLogger("scigantic_msa")

__all__ = ["MSA", "RemoteNotAllowed", "SERVERS"]

#: What each server is, for the messages that say where a sequence goes.
SERVERS: Dict[str, str] = {
    "colabfold": "the ColabFold MMseqs2 server, run by the ColabFold team. Results there can be "
                 "downloaded by anyone who knows the sequence",
    "nvidia": "NVIDIA's hosted MSA Search (needs NVIDIA_API_KEY), under NVIDIA's API trial terms",
}


class RemoteNotAllowed(PermissionError):
    """get() was called without remote=True, or SCIGANTIC_MSA_NO_REMOTE is set."""


class MSA:
    """Get an alignment for a protein sequence, once.

    server: "colabfold" (default) or "nvidia".
    host: another URL for that server, for example your own ColabFold
        MMseqs2 server. Default: api.colabfold.com, or
        ``SCIGANTIC_MSA_COLABFOLD_URL`` / ``SCIGANTIC_MSA_NVIDIA_URL``.
    cache: keep results in a local cache (default on) so the same sequence
        is not submitted twice. ``cache_dir`` picks the directory.
    quiet: do not print progress lines to stderr.
    backend: a ready client object, mainly for tests.
    """

    def __init__(
        self,
        server: str = "colabfold",
        *,
        host: Optional[str] = None,
        cache: bool = True,
        cache_dir: Union[str, Path, None] = None,
        quiet: bool = False,
        backend: Any = None,
    ):
        if backend is None:
            if server not in BACKENDS:
                raise ValueError(f"unknown server {server!r}; choose one of {', '.join(BACKENDS)}")
            kw: Dict[str, Any] = {}
            if host:
                kw["host" if server == "colabfold" else "url"] = host
            backend = make_backend(server, **kw)
        self.backend = backend
        self.server = backend.name
        self.quiet = quiet
        self._cache = AlignmentCache(Path(cache_dir).expanduser() if cache_dir else None) if cache else None

    # ------------------------------------------------------------ info
    @property
    def server_name(self) -> str:
        """The host that receives the sequence, e.g. api.colabfold.com."""
        return str(self.backend.server_name)

    def describe_server(self) -> str:
        return f"{self.server_name} ({SERVERS.get(self.server, self.server)})"

    @property
    def cache_dir(self) -> Optional[Path]:
        return self._cache.root.parent if self._cache is not None else None

    def cached(self, sequence: str) -> Optional[Alignment]:
        """The cached result for this sequence and server, or None. Sends nothing."""
        if self._cache is None:
            return None
        q = canonical_sequence(sequence)
        return self._cache.get(seq_sha256(q), self.backend.name, self.backend.dbkey)

    # ------------------------------------------------------------ get
    def get(self, sequence: Union[str, Sequence[str]], *, remote: bool = False) -> Any:
        """The alignment for one sequence, or a list of alignments for the
        chains of a complex (pass a list; identical chains are searched once).

        remote must be True: the sequence is sent to the server named in the
        error otherwise. A cached result is used without sending anything.
        """
        single = isinstance(sequence, str)
        seqs: List[str] = [sequence] if isinstance(sequence, str) else list(sequence)
        if not seqs:
            raise ValueError("no sequence given")
        queries = [canonical_sequence(s) for s in seqs]
        if remote is not True:
            raise RemoteNotAllowed(
                f"MSA().get() sends the sequence to {self.describe_server()}. "
                "Pass remote=True to allow that, or use MSA().from_a3m() with an alignment you already have."
            )
        if remote_disabled():
            raise RemoteNotAllowed("SCIGANTIC_MSA_NO_REMOTE is set, so no sequence is sent to any server")
        done: Dict[str, Alignment] = {}
        for i, q in enumerate(queries, 1):
            if q not in done:
                done[q] = self._one(q, f"chain {i}" if not single else "sequence")
        out = [done[q] for q in queries]
        return out[0] if single else out

    def _one(self, q: str, what: str) -> Alignment:
        sha = seq_sha256(q)
        b = self.backend
        if self._cache is not None:
            hit = self._cache.get(sha, b.name, b.dbkey)
            if hit is not None:
                self._say(f"{what}: {hit.summary()} (nothing sent)")
                return hit
        lock = self._cache.lock(sha, b.name, b.dbkey) if self._cache is not None else None
        if lock is not None:
            lock.acquire()
        try:
            if self._cache is not None:
                again = self._cache.get(sha, b.name, b.dbkey)
                if again is not None:
                    self._say(f"{what}: computed by another process while waiting")
                    return again
            if not self.quiet:
                announce(b)
            res = b.search(q)
            m = Alignment(query=q, rows=res.rows, source="remote", provenance=dict(res.provenance),
                          templates=res.templates, attribution=[_attribution(b)])
            if self._cache is not None:
                try:
                    m.cache_path = str(self._cache.put(m, b.name, b.dbkey))
                except OSError as e:
                    log.warning("could not write the cache: %s", e)
            self._say(f"{what}: {m.summary()}")
            return m
        finally:
            if lock is not None:
                lock.release()

    # ------------------------------------------------------------ files
    def from_a3m(self, a3m: Union[str, bytes, os.PathLike], sequence: Optional[str] = None) -> Alignment:
        """Read an alignment you already have: a path to an ``.a3m`` (or
        ``.a3m.gz``, or Stockholm ``.sto``) file, or the A3M text itself.
        Nothing is sent anywhere.

        With ``sequence``, the first record must be exactly that sequence.
        The whole file is checked: allowed characters only, every row as
        wide as the query, and no complex header (pass one file per chain).
        """
        return read_alignment(a3m, sequence)

    def _say(self, msg: str) -> None:
        log.info(msg)
        if not self.quiet:
            print(f"scigantic-msa: {msg}", file=sys.stderr, flush=True)


def read_alignment(a3m: Union[str, bytes, os.PathLike], sequence: Optional[str] = None) -> Alignment:
    name = None
    data: Union[str, bytes]
    if isinstance(a3m, os.PathLike) or (isinstance(a3m, str) and "\n" not in a3m and not a3m.lstrip().startswith(">")):
        p = Path(a3m)
        if not p.is_file():
            raise FileNotFoundError(f"{p} does not exist")
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
    head = data[:20] if isinstance(data, bytes) else data[:20].encode()
    if head.lstrip().startswith(b"# STOCKHOLM"):
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        rows = parse_stockholm(text.split("\n"), sequence, what=what)
    else:
        rows = parse_a3m(data, sequence, what=what)
    prov = {"from": "file"}
    if name:
        prov["file"] = name
    return Alignment(query=rows[0].seq, rows=rows, source="file", provenance=prov)


def _attribution(backend: Any) -> Dict[str, Any]:
    if backend.name == "colabfold":
        return {"id": "colabfold-api", "title": "ColabFold MMseqs2 server", "server": backend.server_name,
                "license": "see the server's terms",
                "citation": "Mirdita et al., ColabFold: making protein folding accessible to all, Nature Methods 2022"}
    return {"id": "nvidia-msa-search", "title": "NVIDIA MSA Search", "server": backend.server_name,
            "license": "NVIDIA API trial terms"}
