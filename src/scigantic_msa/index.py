"""Client for the sequence-keyed index of published MSAs (format version 1).

The index does not hold alignments. Each entry says where the alignment for
one sequence already lives (an https URL, a byte offset and a length) and
which source and license it comes from.

Layout under the base URL::

    v1/manifest.json                         points at one immutable build
    v1/builds/<build_id>/SHA256SUMS          sha256 of every file in the build
    v1/builds/<build_id>/shards/<abc>.jsonl.gz
    v1/builds/<build_id>/by_length/<L>.fa.gz and by_length/index.json
    v1/builds/<build_id>/sequences.fasta.gz

A lookup reads the manifest (cached for an hour), SHA256SUMS (cached per
build) and one shard (cached per build). Only the first three hex digits of
the hash reach the server.

The base URL is ``SCIGANTIC_MSA_INDEX_URL`` or ``DEFAULT_INDEX_URL``. It can
also be a local directory or a mounted path.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import http
from .cache import FileCache, atomic_write, cache_dir
from .seqkey import is_seq_sha256

__all__ = [
    "DEFAULT_INDEX_URL",
    "Index",
    "IndexEntry",
    "FileRef",
    "IndexIntegrityError",
    "IndexUnavailable",
    "index_url",
]

#: Default index location. Change it here, or set SCIGANTIC_MSA_INDEX_URL.
DEFAULT_INDEX_URL = "https://msa-index.scigantic.com/"

FORMAT = "scigantic-msa-index"
FORMAT_VERSION = 1
MANIFEST_TTL = 3600.0
ROLES = frozenset({
    "colabfold", "af3_json", "uniref90", "mgnify", "bfd_uniclust", "cfdb_uniref30",
    "uniprot", "uniclust30", "templates_pdb70", "templates_hmm",
})
FORMATS = frozenset({"a3m", "sto", "hhr", "json"})


class IndexIntegrityError(Exception):
    """An index file is missing, truncated, or does not match its checksum.
    This is never treated as 'no hit'."""


class IndexUnavailable(Exception):
    """The index cannot be reached and nothing usable is cached."""


def index_url() -> str:
    return os.environ.get("SCIGANTIC_MSA_INDEX_URL") or DEFAULT_INDEX_URL


@dataclass
class FileRef:
    role: str
    format: str
    url: str
    offset: int
    length: int
    compression: str = "none"
    uncompressed_length: Optional[int] = None

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "FileRef":
        try:
            f = cls(
                role=str(d["role"]),
                format=str(d["format"]),
                url=str(d["url"]),
                offset=int(d["offset"]),
                length=int(d["length"]),
                compression=str(d.get("compression", "none")),
                uncompressed_length=(int(d["uncompressed_length"]) if d.get("uncompressed_length") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise IndexIntegrityError(f"bad file entry {d!r}: {e}") from None
        if f.compression not in ("none", "deflate-raw"):
            raise IndexIntegrityError(f"unknown compression {f.compression!r}")
        if f.compression == "deflate-raw" and f.uncompressed_length is None:
            raise IndexIntegrityError("deflate-raw file entry without uncompressed_length")
        if f.offset < 0 or f.length <= 0:
            raise IndexIntegrityError(f"bad offset/length in {d!r}")
        return f


@dataclass
class IndexEntry:
    seq_sha256: str
    length: int
    source: str
    source_id: str
    license: str
    files: List[FileRef]
    n_aliases: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def file(self, role: str) -> Optional[FileRef]:
        for f in self.files:
            if f.role == role:
                return f
        return None

    def roles(self) -> List[str]:
        return [f.role for f in self.files]

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "IndexEntry":
        try:
            e = cls(
                seq_sha256=str(d["seq_sha256"]),
                length=int(d["length"]),
                source=str(d["source"]),
                source_id=str(d["source_id"]),
                license=str(d.get("license", "")),
                files=[FileRef.from_json(f) for f in d["files"]],
                n_aliases=d.get("n_aliases"),
                raw=d,
            )
        except (KeyError, TypeError, ValueError) as ex:
            raise IndexIntegrityError(f"bad index entry: {ex}") from None
        if not is_seq_sha256(e.seq_sha256):
            raise IndexIntegrityError(f"bad seq_sha256 {e.seq_sha256!r}")
        return e


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Index:
    """Read the index. ``base`` is an https URL, a file:// URL or a directory.

    ``offline=True`` never touches the network: everything must already be in
    the cache (see :meth:`download_all`) or under a local ``base``.
    """

    def __init__(
        self,
        base: Optional[str] = None,
        *,
        cache_root: Optional[Path] = None,
        offline: bool = False,
        manifest_ttl: float = MANIFEST_TTL,
        mounts: Optional[Dict[str, str]] = None,
    ):
        b = base or index_url()
        if not http.is_local(b) and not b.endswith("/"):
            b += "/"
        self.base = b
        self.local = http.is_local(b)
        self.offline = offline or os.environ.get("SCIGANTIC_MSA_OFFLINE") == "1"
        self.manifest_ttl = manifest_ttl
        root = cache_root or cache_dir()
        tag = hashlib.sha256(self.base.encode()).hexdigest()[:12]
        self.cache = root / "index" / tag
        self.files = FileCache(root)
        self.mounts = mounts if mounts is not None else _mounts_from_env()
        self._manifest: Optional[Dict[str, Any]] = None
        self._sums: Optional[Dict[str, str]] = None
        self._shards: Dict[str, List[IndexEntry]] = {}

    # ----------------------------------------------------------- raw access
    def _url(self, rel: str) -> str:
        if self.local:
            return str(http.local_path(self.base) / rel)
        return self.base + rel

    def _fetch(self, rel: str) -> bytes:
        if self.offline and not self.local:
            raise IndexUnavailable(f"offline and {rel} is not cached")
        try:
            return http.get_bytes(self._url(rel))
        except http.NotFound:
            raise IndexIntegrityError(f"index file {rel} is missing") from None
        except http.HTTPError as e:
            raise IndexUnavailable(f"could not read index file {rel}: {e}") from None

    # ----------------------------------------------------------- manifest
    def manifest(self) -> Dict[str, Any]:
        if self._manifest is not None:
            return self._manifest
        cached = self.cache / "manifest.json"
        data: Optional[bytes] = None
        if not self.local and cached.exists():
            age = time.time() - cached.stat().st_mtime
            if age < self.manifest_ttl or self.offline:
                data = cached.read_bytes()
        if data is None:
            try:
                data = self._fetch("v1/manifest.json")
            except IndexUnavailable:
                if cached.exists():
                    data = cached.read_bytes()
                else:
                    raise
            else:
                if not self.local:
                    self._check_manifest(data)
                    atomic_write(cached, data)
        m = self._check_manifest(data)
        self._manifest = m
        return m

    @staticmethod
    def _check_manifest(data: bytes) -> Dict[str, Any]:
        if not data.strip():
            raise IndexIntegrityError("index manifest is empty")
        try:
            m = json.loads(data)
        except ValueError:
            raise IndexIntegrityError("index manifest is not JSON") from None
        if not isinstance(m, dict) or m.get("format") != FORMAT:
            raise IndexIntegrityError("not a scigantic-msa index manifest")
        if m.get("format_version") != FORMAT_VERSION:
            raise IndexIntegrityError(
                f"index format_version {m.get('format_version')} is not supported by this version "
                f"of scigantic-msa (needs {FORMAT_VERSION}); upgrade the package"
            )
        for k in ("build_id", "build_root", "shard_path", "sha256sums", "sha256sums_sha256"):
            if not m.get(k):
                raise IndexIntegrityError(f"index manifest lacks {k}")
        if int(m.get("shard_hex_len", 3)) != 3:
            raise IndexIntegrityError("unsupported shard_hex_len")
        return m

    @property
    def build_id(self) -> str:
        return str(self.manifest()["build_id"])

    def _build_rel(self, rel: str) -> str:
        root = str(self.manifest()["build_root"])
        if not root.endswith("/"):
            root += "/"
        return root + rel

    def _build_cache(self) -> Path:
        return self.cache / "builds" / self.build_id

    # ----------------------------------------------------------- checksums
    def sha256sums(self) -> Dict[str, str]:
        if self._sums is not None:
            return self._sums
        m = self.manifest()
        rel = str(m["sha256sums"])
        cached = self._build_cache() / "SHA256SUMS"
        data = None
        if cached.exists():
            data = cached.read_bytes()
            if _sha(data) != m["sha256sums_sha256"]:
                data = None
        if data is None:
            data = self._fetch(self._build_rel(rel))
            if _sha(data) != m["sha256sums_sha256"]:
                raise IndexIntegrityError("SHA256SUMS does not match the manifest checksum")
            if not self.local:
                atomic_write(cached, data)
        text = gzip.decompress(data).decode() if rel.endswith(".gz") else data.decode()
        sums: Dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise IndexIntegrityError(f"bad SHA256SUMS line {line[:80]!r}")
            sums[parts[1].strip().lstrip("*")] = parts[0].lower()
        if not sums:
            raise IndexIntegrityError("SHA256SUMS is empty")
        self._sums = sums
        return sums

    def _verified(self, rel: str) -> bytes:
        """A build file, checked against SHA256SUMS, cached on disk."""
        want = self.sha256sums().get(rel)
        if want is None:
            raise IndexIntegrityError(f"{rel} is not listed in SHA256SUMS")
        cached = self._build_cache() / rel
        if not self.local and cached.exists():
            data = cached.read_bytes()
            if _sha(data) == want:
                return data
        data = self._fetch(self._build_rel(rel))
        if _sha(data) != want:
            raise IndexIntegrityError(f"{rel} does not match its checksum (got {len(data)} bytes)")
        if not self.local:
            atomic_write(cached, data)
        return data

    # ----------------------------------------------------------- shards
    def shard(self, prefix: str) -> List[IndexEntry]:
        prefix = prefix.lower()
        if len(prefix) != 3 or any(c not in "0123456789abcdef" for c in prefix):
            raise ValueError(f"bad shard prefix {prefix!r}")
        if prefix in self._shards:
            return self._shards[prefix]
        rel = str(self.manifest()["shard_path"]).format(prefix=prefix)
        data = self._verified(rel)
        entries = self._parse_shard(prefix, data)
        self._shards[prefix] = entries
        return entries

    def _parse_shard(self, prefix: str, data: bytes) -> List[IndexEntry]:
        try:
            text = gzip.decompress(data).decode("utf-8")
        except (OSError, EOFError, UnicodeDecodeError) as e:
            raise IndexIntegrityError(f"shard {prefix} is not valid gzip text: {e}") from None
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            raise IndexIntegrityError(f"shard {prefix} is empty (no header line)")
        try:
            head = json.loads(lines[0])
        except ValueError:
            raise IndexIntegrityError(f"shard {prefix} header is not JSON") from None
        if not isinstance(head, dict) or head.get("_shard") != prefix:
            raise IndexIntegrityError(f"shard {prefix} header names {head.get('_shard') if isinstance(head, dict) else head!r}")
        if head.get("build_id") != self.build_id:
            raise IndexIntegrityError(f"shard {prefix} is from build {head.get('build_id')}, manifest says {self.build_id}")
        n = head.get("entries")
        if not isinstance(n, int) or n != len(lines) - 1:
            raise IndexIntegrityError(f"shard {prefix} header says {n} entries, file has {len(lines) - 1}")
        out: List[IndexEntry] = []
        for line in lines[1:]:
            try:
                d = json.loads(line)
            except ValueError:
                raise IndexIntegrityError(f"shard {prefix} has a line that is not JSON") from None
            entry = IndexEntry.from_json(d)
            if not entry.seq_sha256.startswith(prefix):
                raise IndexIntegrityError(f"shard {prefix} holds {entry.seq_sha256[:8]}..., wrong shard")
            out.append(entry)
        return out

    def lookup(self, sha: str) -> List[IndexEntry]:
        """All entries for one seq_sha256, in the manifest's source priority."""
        if not is_seq_sha256(sha):
            raise ValueError("lookup needs a seq_sha256 (64 lowercase hex)")
        hits = [e for e in self.shard(sha[:3]) if e.seq_sha256 == sha]
        prio = {s: i for i, s in enumerate(self.manifest().get("source_priority", []))}
        hits.sort(key=lambda e: prio.get(e.source, len(prio)))
        return hits

    # ----------------------------------------------------------- sources
    def source(self, source_id: str) -> Dict[str, Any]:
        for s in self.manifest().get("sources", []):
            if s.get("id") == source_id:
                return s
        return {"id": source_id}

    def attribution(self, source_id: str) -> Dict[str, Any]:
        s = self.source(source_id)
        return {k: s[k] for k in ("id", "title", "license", "attribution", "citation", "version") if k in s}

    # ----------------------------------------------------------- near search data
    def by_length(self, length: int) -> List[Tuple[str, str]]:
        """(seq_sha256, sequence) for every indexed sequence of this length."""
        near = self.manifest().get("near") or {}
        idx_rel = near.get("by_length_index")
        if not idx_rel:
            raise IndexUnavailable("this index build has no near-identical search data")
        index = json.loads(self._verified(idx_rel))
        if str(length) not in index:
            return []
        rel = str(near["by_length"]).format(length=length)
        data = self._verified(rel)
        want = index[str(length)].get("sha256")
        if want and _sha(data) != want:
            raise IndexIntegrityError(f"{rel} does not match by_length/index.json")
        out = list(_read_fasta(gzip.decompress(data).decode()))
        n = index[str(length)].get("n")
        if n is not None and n != len(out):
            raise IndexIntegrityError(f"{rel} has {len(out)} sequences, index says {n}")
        return out

    def sequences_fasta(self) -> Path:
        """Local path of sequences.fasta.gz (downloaded once per build)."""
        near = self.manifest().get("near") or {}
        rel = near.get("sequences_fasta")
        if not rel:
            raise IndexUnavailable("this index build has no sequences.fasta.gz")
        if self.local:
            p = Path(self._url(self._build_rel(rel)))
            want = self.sha256sums().get(rel)
            if want is None or _file_sha(p) != want:
                raise IndexIntegrityError(f"{rel} does not match its checksum")
            return p
        self._verified(rel)
        return self._build_cache() / rel

    def download_all(self, progress: Any = None) -> int:
        """Fetch every shard and the near-search files, for offline use."""
        n = 0
        for i in range(4096):
            self.shard(f"{i:03x}")
            n += 1
            if progress:
                progress(n)
        near = self.manifest().get("near") or {}
        if near.get("sequences_fasta"):
            self.sequences_fasta()
        if near.get("by_length_index"):
            index = json.loads(self._verified(near["by_length_index"]))
            for L in index:
                self._verified(str(near["by_length"]).format(length=int(L)))
        return n

    # ----------------------------------------------------------- alignment bytes
    def resolve(self, url: str) -> str:
        """A local mounted path for a source URL when one is configured and
        exists, else the URL."""
        if not self.mounts:
            return url
        for s in self.manifest().get("sources", []):
            for https_prefix, s3_prefix in (s.get("url_prefixes") or {}).items():
                if url.startswith(https_prefix):
                    s3 = s3_prefix + url[len(https_prefix):]
                    for mp, local in self.mounts.items():
                        if s3.startswith(mp):
                            p = Path(local) / s3[len(mp):]
                            if p.exists():
                                return str(p)
        return url

    def fetch(self, f: FileRef, *, use_cache: bool = True) -> bytes:
        """The bytes of one alignment file, decompressed, with the length checked."""
        data = self.files.get(f.url, f.offset, f.length) if use_cache else None
        if data is None:
            src = self.resolve(f.url)
            try:
                data = http.get_bytes(src, offset=f.offset, length=f.length)
            except http.HTTPError as e:
                raise IndexUnavailable(f"could not read {f.url}: {e}") from None
            if use_cache:
                self.files.put(f.url, f.offset, f.length, data)
        if f.compression == "deflate-raw":
            try:
                data = zlib.decompress(data, -15)
            except zlib.error as e:
                raise IndexIntegrityError(f"{f.url}: bad deflate data: {e}") from None
            if len(data) != f.uncompressed_length:
                raise IndexIntegrityError(f"{f.url}: decompressed to {len(data)} bytes, index says {f.uncompressed_length}")
        return data

    def stream(self, f: FileRef):
        """An open binary stream of a big uncompressed file (Stockholm)."""
        if f.compression != "none":
            raise ValueError("streaming is only for uncompressed files")
        return http.open_stream(self.resolve(f.url), offset=f.offset)


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_fasta(text: str) -> Iterator[Tuple[str, str]]:
    name = None
    parts: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(parts)
            name = line[1:].strip().split()[0]
            parts = []
        elif line.strip():
            parts.append(line.strip())
    if name is not None:
        yield name, "".join(parts)


def _mounts_from_env() -> Dict[str, str]:
    """SCIGANTIC_MSA_MOUNTS='s3://openfold/=/mnt/openfold;s3://other/=/mnt/other'"""
    raw = os.environ.get("SCIGANTIC_MSA_MOUNTS", "")
    out = {}
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                out[k if k.endswith("/") else k + "/"] = v
    return out
