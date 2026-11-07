"""The MSA result object."""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .a3m import A3MError, Row, classify_header, format_a3m, parse_a3m, ungapped
from .seqkey import canonical_sequence, seq_sha256

__all__ = ["MSA", "SOURCES", "classify_header", "species_key"]

#: Where a result came from. Every MSA has exactly one of these in ``.source``.
SOURCES = ("exact", "near", "remote", "cache")

_TAXID = re.compile(r"(?:\bTaxID=|\bOX=)(\d+)")
_UNIREF_N = re.compile(r"\bn=(\d+)\b")
_UNIPROT = re.compile(r"^(?:tr|sp)\|[A-Za-z0-9]{6,10}(?:_\d)?\|[A-Za-z0-9]+_([A-Za-z0-9]{1,5})(?:_\d+)?$")


_TAXIDS_LINE = "#scigantic-msa-taxids "


def species_key(row: Row) -> Optional[str]:
    """A key that is equal for rows from the same species, or None.

    UniRef90 rows carry the cluster's common taxon; it is only a species when
    the cluster has one member (n=1), so larger clusters give None.
    """
    taxid = row.taxid
    if taxid is None:
        m = _TAXID.search(row.header)
        taxid = m.group(1) if m else None
    if taxid is not None:
        if row.header.startswith("UniRef90_"):
            n = _UNIREF_N.search(row.header)
            if not n or n.group(1) != "1":
                return None
        return "tx:" + taxid
    first = row.header.split(None, 1)[0].partition("/")[0] if row.header.strip() else ""
    m = _UNIPROT.match(first)
    if m:
        return "mn:" + m.group(1)
    return None


@dataclass
class MSA:
    """An alignment for one protein chain.

    rows: unpaired alignment rows, query first.
    uniprot_rows: optional UniProt hits (query first) whose headers carry the
        species, used only for pairing chains of a complex.
    source: one of exact, near, remote, cache.
    provenance: what produced it (index entry, server, mode, databases, dates).
    attribution: sources and licenses to credit.
    derived: set when the rows were copied from a near-identical parent
        sequence: parent seq_sha256, identity and the differences.
    """

    query: str
    rows: List[Row]
    source: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    attribution: List[Dict[str, Any]] = field(default_factory=list)
    uniprot_rows: Optional[List[Row]] = None
    derived: Optional[Dict[str, Any]] = None
    templates: Optional[Dict[str, Any]] = None
    #: Local cache file this result was written to, if any.
    cache_path: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.query = canonical_sequence(self.query)
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if not self.rows or self.rows[0].seq != self.query:
            raise A3MError("the first row of an MSA must be the query")
        if self.uniprot_rows is not None and (not self.uniprot_rows or self.uniprot_rows[0].seq != self.query):
            raise A3MError("the first UniProt row must be the query")

    # ------------------------------------------------------------------ basics
    @property
    def seq_sha256(self) -> str:
        return seq_sha256(self.query)

    @property
    def depth(self) -> int:
        """Number of rows, query included. 1 means no MSA beyond the query."""
        return len(self.rows)

    @property
    def is_derived(self) -> bool:
        return self.derived is not None

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        extra = ""
        if self.derived:
            extra = f", derived from {self.derived['parent_seq_sha256'][:12]} ({self.derived['identity']:.3f} identity)"
        return (
            f"MSA(source={self.source!r}, length={len(self.query)}, depth={self.depth}, "
            f"seq_sha256={self.seq_sha256[:12]}...{extra})"
        )

    def summary(self) -> str:
        """One line that says where this came from, for logs."""
        p = self.provenance
        where = p.get("index_source") or p.get("backend") or p.get("from") or ""
        lic = ", ".join(sorted({a.get("license", "") for a in self.attribution if a.get("license")}))
        s = f"{self.source}: {where} depth {self.depth}"
        if lic:
            s += f", license {lic}"
        if self.derived:
            d = self.derived
            s += (
                f"; rows copied from a near-identical sequence {d['parent_seq_sha256'][:12]} "
                f"({d['identity'] * 100:.1f}% identity, {len(d.get('mismatches', []))} substitutions, "
                f"{d.get('insertions', 0)} insertions, {d.get('deletions', 0)} deletions)"
            )
        return s

    # ------------------------------------------------------------------ text
    def a3m(self, max_rows: Optional[int] = None, dedupe: bool = False) -> str:
        rows = self.rows
        if dedupe:
            rows = dedupe_rows(rows)
        if max_rows is not None:
            rows = rows[:max_rows]
        return format_a3m(rows)

    def uniprot_a3m(self) -> str:
        return format_a3m(self.uniprot_rows or [])

    # ------------------------------------------------------------------ writers
    def to_a3m(self, path: "str | Path | None" = None, *, max_rows: Optional[int] = None) -> str:
        from .writers import write_a3m

        return write_a3m(self, path, max_rows=max_rows)

    def to_colabfold(self, path: "str | Path") -> Path:
        from .writers import write_colabfold

        return write_colabfold([self], path)

    def to_af3_json(self, path: "str | Path | None" = None, **kw: Any) -> dict:
        from .writers import write_af3_json

        return write_af3_json([self], path, **kw)

    def to_boltz(self, directory: "str | Path", **kw: Any) -> str:
        from .writers import write_boltz

        return write_boltz([self], directory, **kw)[0]

    def to_chai(self, directory: "str | Path") -> Path:
        from .writers import write_chai

        return write_chai([self], directory)[0]

    def to_openfold3(self, directory: "str | Path") -> dict:
        from .writers import write_openfold3

        return write_openfold3([self], directory)[0]

    # ------------------------------------------------------------------ storage form
    def to_bundle(self) -> bytes:
        """The form used for the local cache: gzip of an
        A3M whose first line is a '#scigantic-msa' JSON line."""
        meta = {
            "v": 1,
            "seq_sha256": self.seq_sha256,
            "provenance": self.provenance,
            "attribution": self.attribution,
            "derived": self.derived,
            "n_rows": len(self.rows),
            "n_uniprot": len(self.uniprot_rows) if self.uniprot_rows else 0,
        }
        taxids = {str(i): r.taxid for i, r in enumerate(self.rows) if r.taxid}
        text = "#scigantic-msa " + json.dumps(meta, separators=(",", ":"), sort_keys=True) + "\n"
        text += format_a3m(self.rows)
        if self.uniprot_rows:
            text += format_a3m(self.uniprot_rows[1:])
        # The taxid map goes last, after the rows.
        if taxids:
            text += _TAXIDS_LINE + json.dumps(taxids, separators=(",", ":"), sort_keys=True) + "\n"
        return gzip.compress(text.encode("utf-8"), 6, mtime=0)

    @classmethod
    def from_bundle(cls, data: bytes, source: str, query: Optional[str] = None) -> "MSA":
        try:
            text = gzip.decompress(data).decode("utf-8")
        except (OSError, EOFError, UnicodeDecodeError) as e:
            raise A3MError(f"stored alignment is not gzip UTF-8 text: {e}") from None
        meta: Dict[str, Any] = {}
        if text.startswith("#scigantic-msa "):
            first, _, text = text.partition("\n")
            try:
                meta = json.loads(first[len("#scigantic-msa ") :])
            except ValueError:
                raise A3MError("stored alignment has a broken metadata line") from None
        body, sep, last = text.rstrip("\n").rpartition("\n")
        if sep and last.startswith(_TAXIDS_LINE):
            try:
                taxids = json.loads(last[len(_TAXIDS_LINE) :])
            except ValueError:
                raise A3MError("stored alignment has a broken taxid line") from None
            if not isinstance(taxids, dict):
                raise A3MError("stored alignment has a broken taxid line")
            meta = dict(meta, taxids=taxids)
            text = body + "\n"
        return cls.from_a3m(text, source=source, query=query, meta=meta)

    @classmethod
    def from_a3m(
        cls,
        text: "str | bytes",
        *,
        source: str,
        query: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> "MSA":
        meta = meta or {}
        rows = parse_a3m(text, query, db="auto")
        n_uni = int(meta.get("n_uniprot") or 0)
        n_rows = meta.get("n_rows")
        if n_rows is not None and len(rows) != int(n_rows) + max(n_uni - 1, 0):
            raise A3MError(
                f"stored alignment has {len(rows)} rows, its metadata says {int(n_rows) + max(n_uni - 1, 0)}"
            )
        uniprot_rows = None
        if n_uni:
            split = len(rows) - (n_uni - 1)
            uniprot_rows = [rows[0]] + rows[split:]
            rows = rows[:split]
        for k, v in (meta.get("taxids") or {}).items():
            i = int(k)
            if 0 <= i < len(rows):
                rows[i].taxid = v
        want = meta.get("seq_sha256")
        msa = cls(
            query=rows[0].seq,
            rows=rows,
            source=source,
            provenance=dict(provenance or meta.get("provenance") or {}),
            attribution=list(meta.get("attribution") or []),
            uniprot_rows=uniprot_rows,
            derived=meta.get("derived"),
        )
        if want and want != msa.seq_sha256:
            raise A3MError("stored alignment's metadata names a different sequence")
        return msa


def dedupe_rows(rows: Sequence[Row]) -> List[Row]:
    """Drop rows whose residues (gaps removed, uppercased) repeat an earlier row.
    The query stays first."""
    seen = set()
    out: List[Row] = []
    for r in rows:
        k = ungapped(r.seq)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out
