"""The alignment result object."""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .a3m import A3MError, Row, classify_header, format_a3m, parse_a3m, ungapped
from .seqkey import canonical_sequence, seq_sha256

__all__ = ["Alignment", "SOURCES", "classify_header", "species_key", "NOT_SPECIES_TAXIDS"]

#: Where a result came from: a search server just now, this machine's cache
#: of an earlier search, an alignment file you passed in, a server search
#: merged with open data, or open data alone.
SOURCES = ("remote", "cache", "file", "merged", "open")

_TAXID = re.compile(r"(?:\bTaxID=|\bOX=)(\d+)")
_UNIREF_N = re.compile(r"\bn=(\d+)\b")
_UNIREF_ROW = re.compile(r"^UniRef(?:100|90|50)_")
_UNIPROT = re.compile(r"^(?:tr|sp)\|[A-Za-z0-9]{6,10}(?:_\d)?\|[A-Za-z0-9]+_([A-Za-z0-9]{1,5})(?:_\d+)?$")
_TAX_NAME = re.compile(r"\b(?:Tax|OS)=(.+?)(?:\s+(?:TaxID|OX|GN|PE|SV|RepID)=|$)")

#: NCBI taxa that are never one species: root, cellular organisms, the
#: three domains, viruses, unclassified and other sequences, synthetic
#: construct and unidentified.
NOT_SPECIES_TAXIDS = frozenset({"1", "131567", "2", "2157", "2759", "10239", "12908", "28384", "32630", "32644"})


_TAXIDS_LINE = "#scigantic-msa-taxids "


def _not_a_species_name(header: str) -> bool:
    m = _TAX_NAME.search(header)
    if not m:
        return False
    name = m.group(1).strip().lower()
    return name.startswith("unclassified") or "environmental samples" in name or "metagenome" in name


def species_key(row: Row) -> Optional[str]:
    """A key that is equal for rows from the same species, or None.

    Only species-level keys count. A UniRef row (UniRef100, 90 or 50) names
    the common taxon of its cluster, which is one organism only when the
    cluster has one member (n=1), so larger clusters give None. Taxa that
    group many organisms (the domains, root, viruses, "unclassified ..."
    names, metagenomes) give None, and so do UniProt's virtual species
    codes, which all start with 9 (``_9BACT``, ``_9GAMM``) and stand for a
    group of organisms, not one.
    """
    m = _TAXID.search(row.header)
    taxid = row.taxid or (m.group(1) if m else None)
    if m and _UNIREF_ROW.match(row.header):
        # The header's taxon is the cluster's; one member makes it a species.
        n = _UNIREF_N.search(row.header)
        if not n or n.group(1) != "1":
            return None
    if taxid is not None:
        if taxid in NOT_SPECIES_TAXIDS or _not_a_species_name(row.header):
            return None
        return "tx:" + taxid
    first = row.header.split(None, 1)[0].partition("/")[0] if row.header.strip() else ""
    m = _UNIPROT.match(first)
    if m and not m.group(1).startswith("9"):
        return "mn:" + m.group(1)
    return None


@dataclass
class Alignment:
    """An alignment for one protein chain.

    rows: alignment rows, query first.
    uniprot_rows: optional UniProt hits (query first) whose headers carry the
        species, used only for pairing chains of a complex.
    source: remote, cache or file.
    provenance: what produced it (server, mode, dates, file name).
    attribution: sources and licenses to credit.
    """

    query: str
    rows: List[Row]
    source: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    attribution: List[Dict[str, Any]] = field(default_factory=list)
    uniprot_rows: Optional[List[Row]] = None
    templates: Optional[Dict[str, Any]] = None
    #: Local cache file this result was written to, if any.
    cache_path: Optional[str] = field(default=None, repr=False, compare=False)
    #: For a chain of a complex from ``MSA().get([...])``: the pairs made
    #: for that complex (a :class:`scigantic_msa.pairing.ComplexPairing`),
    #: which every writer uses instead of pairing by species itself.
    complex_pairing: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.query = canonical_sequence(self.query)
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if not self.rows or self.rows[0].seq != self.query:
            raise A3MError("the first row of an alignment must be the query")
        if self.uniprot_rows is not None and (not self.uniprot_rows or self.uniprot_rows[0].seq != self.query):
            raise A3MError("the first UniProt row must be the query")

    @property
    def sequence(self) -> str:
        return self.query

    @property
    def seq_sha256(self) -> str:
        return seq_sha256(self.query)

    @property
    def depth(self) -> int:
        """Number of rows, query included. 1 means no hits beyond the query."""
        return len(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return (
            f"Alignment(source={self.source!r}, length={len(self.query)}, depth={self.depth}, "
            f"seq_sha256={self.seq_sha256[:12]}...)"
        )

    def summary(self) -> str:
        """One line that says where this came from, for logs. It never
        contains the sequence."""
        p = self.provenance
        where = p.get("server") or p.get("file") or p.get("backend") or ""
        s = f"{self.source}: {where}, length {len(self.query)}, depth {self.depth}".replace(": ,", ":")
        lic = ", ".join(sorted({a.get("license", "") for a in self.attribution if a.get("license")}))
        if lic:
            s += f", terms: {lic}"
        return s

    def a3m(self, max_rows: Optional[int] = None) -> str:
        """The alignment as A3M text, query first."""
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        return format_a3m(rows)

    # ------------------------------------------------------------------ storage form
    def to_bundle(self) -> bytes:
        """The form used for the local cache: gzip of an
        A3M whose first line is a '#scigantic-msa' JSON line."""
        meta = {
            "v": 1,
            "seq_sha256": self.seq_sha256,
            "provenance": self.provenance,
            "attribution": self.attribution,
            "n_rows": len(self.rows),
            "n_uniprot": len(self.uniprot_rows) if self.uniprot_rows else 0,
        }
        # Each row's database label where it is not what its header alone
        # gives, so a result read back from the cache labels its rows (and so
        # Chai-1's source_database) exactly as the fresh result did: a
        # ColabFold environmental row is "env" even when its header is MGYP.
        stored = self.rows + (self.uniprot_rows[1:] if self.uniprot_rows else [])
        meta["dbs"] = {str(i): r.db for i, r in enumerate(stored) if i and r.db != classify_header(r.header)}
        if self.templates is not None:
            meta["templates"] = self.templates
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
    def from_bundle(cls, data: bytes, source: str, query: Optional[str] = None) -> "Alignment":
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
    ) -> "Alignment":
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
        prov = dict(provenance or meta.get("provenance") or {})
        dbs = meta.get("dbs")
        if isinstance(dbs, dict):
            stored = rows + (uniprot_rows[1:] if uniprot_rows else [])
            for k, v in dbs.items():
                i = int(k)
                if 0 < i < len(stored) and isinstance(v, str):
                    stored[i].db = v
        elif prov.get("backend") == "colabfold":
            # Written by 0.2.7 or earlier, without labels: a fresh ColabFold
            # result labels every row of the environmental file "env" unless
            # its header is UniRef, and uniref.a3m holds only UniRef rows.
            for r in rows[1:]:
                if not r.header.startswith("UniRef"):
                    r.db = "env"
        for k, v in (meta.get("taxids") or {}).items():
            i = int(k)
            if 0 <= i < len(rows):
                rows[i].taxid = v
        want = meta.get("seq_sha256")
        templates = meta.get("templates")
        msa = cls(
            query=rows[0].seq,
            rows=rows,
            source=source,
            provenance=prov,
            attribution=list(meta.get("attribution") or []),
            uniprot_rows=uniprot_rows,
            templates=templates if isinstance(templates, dict) else None,
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
