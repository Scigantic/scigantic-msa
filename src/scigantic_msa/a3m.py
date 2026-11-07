"""Strict A3M and Stockholm parsing.

An alignment here is a list of :class:`Row`. Row 0 is the query: no lowercase,
no gaps, and its residues equal the canonical chain sequence. Every row has
the same number of match columns (uppercase letters plus ``-``) as the query
has residues. Lowercase letters are insertions relative to the query.

Rules (from the AlphaFold 3, Boltz and Chai parsers and the ColabFold server
output):

- NUL bytes are removed. ColabFold ends each query block with one.
- A leading ``#`` line is only accepted as the ColabFold header
  ``#<len>\\t<cardinality>``. A header naming several chains is a complex
  file and is refused here (use :func:`parse_colabfold_complex`).
  Other ``#`` lines are refused unless ``skip_comments=True``.
- Records whose name ends in ``_consensus`` are dropped (UniClust30 files).
- Wrapped sequence lines are joined and whitespace inside them is removed.
- Allowed characters in a row: ``A-Z``, ``a-z`` and ``-``.
- Rows with no match residue (all ``-``) are dropped.
- A row identical to the query row after the first one (ColabFold repeats the
  query at the start of the env block) is dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .seqkey import canonical_sequence

__all__ = [
    "A3MError",
    "Row",
    "parse_a3m",
    "parse_stockholm",
    "format_a3m",
    "match_length",
    "ungapped",
    "a3m_query",
    "parse_colabfold_complex",
    "classify_header",
]

_ALLOWED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-")
_CF_HEADER = re.compile(r"^#(\d+(?:,\d+)*)\t(\d+(?:,\d+)*)\s*$")
_TAXID = re.compile(r"(?:\bTaxID=|\bOX=)(\d+)")


class A3MError(ValueError):
    """The alignment text does not follow the rules above."""


@dataclass
class Row:
    """One alignment row.

    header: text after ``>``, kept verbatim.
    seq: the A3M row (uppercase and ``-`` are match columns, lowercase are insertions).
    db: where the row came from: query, uniref, env, uniref90, mgnify,
        bfd_uniclust, cfdb_uniref30, uniprot, uniclust30 or other.
    taxid: NCBI taxonomy id as a string when known, else None.
    """

    header: str
    seq: str
    db: str = "other"
    taxid: Optional[str] = None
    extra: dict = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> dict:
        d = {"header": self.header, "seq": self.seq, "db": self.db}
        if self.taxid is not None:
            d["taxid"] = self.taxid
        return d


def match_length(row: str) -> int:
    return sum(1 for c in row if c == "-" or "A" <= c <= "Z")


def ungapped(row: str) -> str:
    """Residues of a row with gaps removed, uppercased."""
    return "".join(c for c in row if c != "-").upper()


def _match_residues(row: str) -> str:
    return "".join(c for c in row if c == "-" or "A" <= c <= "Z")


def _taxid_from_header(header: str) -> Optional[str]:
    m = _TAXID.search(header)
    return m.group(1) if m else None


def _records(text: str, *, skip_comments: bool, what: str) -> Iterator[Tuple[str, str]]:
    name: Optional[str] = None
    parts: List[str] = []
    seen = False
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith("#"):
            m = _CF_HEADER.match(line)
            if m and not seen:
                seen = True
                lens = m.group(1).split(",")
                if len(lens) > 1:
                    raise A3MError(
                        f"{what}: line {lineno} is a ColabFold complex header ({line.strip()!r}); "
                        "read it with parse_colabfold_complex"
                    )
                yield ("#len", lens[0])
                continue
            seen = True
            if skip_comments:
                continue
            raise A3MError(f"{what}: line {lineno} is a '#' line that is not a ColabFold header: {line[:60]!r}")
        seen = True
        if line.startswith(">"):
            if name is not None:
                yield (name, "".join(parts))
            name = line[1:]
            parts = []
            continue
        if name is None:
            raise A3MError(f"{what}: line {lineno} has sequence data before any '>' header")
        parts.append("".join(line.split()))
    if name is not None:
        yield (name, "".join(parts))


def _is_consensus(header: str) -> bool:
    first = header.lstrip(">").split()
    return bool(first) and first[0].endswith("_consensus")


def parse_a3m(
    text: "str | bytes",
    query: Optional[str] = None,
    *,
    db: str = "auto",
    skip_comments: bool = False,
    what: str = "a3m",
) -> List[Row]:
    """Parse and validate an A3M. Returns rows with the query first.

    ``query``, when given, must equal the first record's residues after
    canonicalisation. ``db`` labels the hit rows; ``auto`` reads the label
    from each header (see :func:`classify_header`).
    """
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8")
        except UnicodeDecodeError as e:
            raise A3MError(f"{what}: not UTF-8 text ({e})") from None
    text = text.replace("\x00", "")
    rows: List[Row] = []
    declared_len: Optional[int] = None
    qrow: Optional[str] = None
    for i, (header, seq) in enumerate(_records(text, skip_comments=skip_comments, what=what)):
        if header == "#len":
            declared_len = int(seq)
            continue
        if _is_consensus(header):
            continue
        if not seq:
            raise A3MError(f"{what}: record {header[:40]!r} has no sequence")
        bad = set(seq) - _ALLOWED
        if bad:
            raise A3MError(
                f"{what}: record {header[:40]!r} has characters outside A-Z, a-z and '-': "
                f"{''.join(sorted(bad))!r}"
            )
        if qrow is None:
            if any(c.islower() for c in seq) or "-" in seq:
                raise A3MError(
                    f"{what}: the first record {header[:40]!r} has gaps or lowercase letters, "
                    "so it is not a usable query row"
                )
            qrow = seq
            if query is not None:
                want = canonical_sequence(query)
                if seq != want:
                    raise A3MError(
                        f"{what}: the first record does not match the query "
                        f"(length {len(seq)} vs {len(want)}; first difference at "
                        f"position {_first_diff(seq, want) + 1})"
                    )
            if declared_len is not None and declared_len != len(seq):
                raise A3MError(f"{what}: header says length {declared_len} but the query has {len(seq)} residues")
            rows.append(Row(header, seq, "query"))
            continue
        n = match_length(seq)
        if n != len(qrow):
            raise A3MError(
                f"{what}: record {header[:40]!r} has {n} match columns, the query has {len(qrow)}"
            )
        if not any("A" <= c <= "Z" for c in seq):
            continue
        if seq == qrow:
            # ColabFold repeats the query record at the start of the env block.
            if header == rows[0].header or header.split()[0:1] == rows[0].header.split()[0:1]:
                continue
        rdb = classify_header(header) if db == "auto" else db
        rows.append(Row(header, seq, rdb, _taxid_from_header(header)))
    if qrow is None:
        raise A3MError(f"{what}: no records")
    return rows


def classify_header(header: str) -> str:
    """Database label for a hit row, from its header."""
    first = header.split(None, 1)[0] if header.strip() else ""
    if first.startswith("UniRef90_"):
        return "uniref90"
    if first.startswith("UniRef"):
        return "uniref"
    if first.startswith(("tr|", "sp|")):
        return "uniprot"
    if first.startswith("MGYP"):
        return "mgnify"
    return "env"


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def format_a3m(rows: Sequence[Row]) -> str:
    return "".join(f">{r.header}\n{r.seq}\n" for r in rows)


def a3m_query(text: str, expect_id: Optional[str] = None) -> Optional[str]:
    """The index builder's query rule: first record that is not a ``_consensus``
    record, skipping ``#`` lines. Returns None when it has gaps or lowercase,
    or when ``expect_id`` is given and is not in the record name."""
    name: Optional[str] = None
    seq: List[str] = []
    consensus = False
    for line in text.replace("\x00", "").split("\n"):
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if name is not None and not consensus:
                break
            name = line[1:].lstrip(">").strip()
            consensus = name.split()[0].endswith("_consensus") if name else False
            seq = []
            continue
        if name is not None:
            seq.append(line.strip())
    if name is None or consensus:
        return None
    q = "".join(seq)
    if not q or "-" in q or any(c.islower() for c in q):
        return None
    if expect_id is not None and expect_id not in name.split()[0]:
        return None
    return q


# --------------------------------------------------------------------------- Stockholm


def parse_stockholm(
    lines: Iterable["str | bytes"],
    query: Optional[str] = None,
    *,
    db: str = "auto",
    max_rows: Optional[int] = None,
    what: str = "stockholm",
) -> List[Row]:
    """Convert a Stockholm alignment to A3M rows.

    Columns where the first sequence has a gap become insertions: residues
    there are lowercased and gaps dropped. ``max_rows`` stops reading once that
    many rows (query included) are complete; this only works for single-block
    files, which is what OpenFold3 and jackhmmer write. A file cut short in
    the middle of an interleaved block fails the query check instead of
    returning a short alignment.
    """
    order: List[str] = []
    seqs: dict = {}
    desc: dict = {}
    started = False
    blocks = 0
    in_block = False
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.rstrip("\r\n")
        if not started:
            if not line.strip():
                continue
            if not line.startswith("# STOCKHOLM"):
                raise A3MError(f"{what}: does not start with '# STOCKHOLM 1.0'")
            started = True
            continue
        if line.startswith("//"):
            break
        if not line.strip():
            in_block = False
            continue
        if line.startswith("#=GS "):
            parts = line[5:].split(None, 2)
            if len(parts) == 3 and parts[1] == "DE":
                desc[parts[0]] = parts[2]
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise A3MError(f"{what}: unexpected line {line[:60]!r}")
        name, aln = parts
        if not in_block:
            in_block = True
            blocks += 1
        if name in seqs:
            if blocks == 1:
                raise A3MError(f"{what}: sequence name {name!r} appears twice in one block")
            seqs[name].append(aln)
        else:
            if blocks > 1:
                raise A3MError(f"{what}: name {name!r} first appears in block {blocks}")
            if max_rows is not None and len(order) >= max_rows:
                break
            order.append(name)
            seqs[name] = [aln]
    if not started:
        raise A3MError(f"{what}: empty file")
    if not order:
        raise A3MError(f"{what}: no sequences")
    aligned = ["".join(seqs[n]) for n in order]
    width = len(aligned[0])
    qaln = aligned[0]
    insert = [c in "-." for c in qaln]
    rows: List[Row] = []
    for name, aln in zip(order, aligned):
        if len(aln) != width:
            raise A3MError(f"{what}: row {name!r} is {len(aln)} columns wide, the first row is {width}")
        out = []
        for c, ins in zip(aln, insert):
            if c in "-.":
                if not ins:
                    out.append("-")
            elif ins:
                out.append(c.lower())
            else:
                out.append(c.upper())
        header = name + (" " + desc[name] if name in desc else "")
        rows.append(Row(header, "".join(out)))
    return parse_a3m(format_a3m(rows), query, db=db, what=what)


# --------------------------------------------------------------------------- ColabFold complex


def parse_colabfold_complex(text: str) -> Tuple[List[int], List[int], List[str], List[Tuple[str, str]]]:
    """Read a ColabFold complex A3M. Returns (lengths, cardinalities, query
    sequences per chain, records) where records are (header, row) pairs of the
    full concatenated width."""
    text = text.replace("\x00", "")
    lines = [l.rstrip("\r") for l in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise A3MError("complex a3m: empty")
    m = _CF_HEADER.match(lines[0])
    if not m:
        raise A3MError("complex a3m: first line is not '#<lens>\\t<cardinalities>'")
    lens = [int(x) for x in m.group(1).split(",")]
    cards = [int(x) for x in m.group(2).split(",")]
    if len(lens) != len(cards):
        raise A3MError("complex a3m: lengths and cardinalities differ in count")
    records = list(_records("\n".join(lines[1:]), skip_comments=False, what="complex a3m"))
    if not records:
        raise A3MError("complex a3m: no records")
    q = records[0][1]
    if len(q) != sum(lens) or "-" in q:
        raise A3MError("complex a3m: first record is not the concatenated query")
    for h, s in records:
        if match_length(s) != sum(lens):
            raise A3MError(f"complex a3m: record {h[:40]!r} is not {sum(lens)} columns wide")
    queries = []
    pos = 0
    for n in lens:
        queries.append(q[pos : pos + n])
        pos += n
    return lens, cards, queries, records
