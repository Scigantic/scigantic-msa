"""Read published alignment files as they stream in.

Unlike :mod:`scigantic_msa.a3m`, which refuses a file with one bad record,
these parsers drop a bad row and count it: published files are large and
now and then hold a broken record, and losing one row is better than losing
the file. The query row is still checked strictly: the first record that is
not a ``_consensus`` record must be exactly the expected sequence, or the
whole file is refused.

Each parser returns a :class:`Parsed`: rows as (header, A3M row, record)
with the query first, where record is the 0-based position in the file
(headers are not always unique), plus counts of what was skipped and why.
Rows that are copies of the query are not kept, but their headers are, since
they name organisms that have this exact sequence.

Files are read line by line and can be cut off after ``max_rows`` rows or
``max_bytes`` bytes. Rows in these files are ordered best hit first, so a
cut drops the weakest hits.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

__all__ = ["Parsed", "SourceMismatch", "parse_a3m_lines", "parse_sto_lines"]

ALLOWED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-")


class SourceMismatch(ValueError):
    """The file's query row is not the expected sequence."""


@dataclass
class Parsed:
    rows: List[Tuple[str, str, int]]
    skipped: Counter = field(default_factory=Counter)
    query_copies: List[str] = field(default_factory=list)
    truncated: bool = False
    bytes_read: int = 0


def match_length(row: str) -> int:
    return sum(1 for c in row if c == "-" or "A" <= c <= "Z")


def _ungapped(row: str) -> str:
    return row.replace("-", "").upper()


def _is_consensus(header: str) -> bool:
    first = header.lstrip(">").split()
    return bool(first) and first[0].endswith("_consensus")


class _Counter:
    """Wraps a line iterator of bytes, counting bytes and stopping past a cap."""

    def __init__(self, lines: Iterable[bytes], max_bytes: Optional[int]):
        self.lines = lines
        self.max_bytes = max_bytes
        self.n = 0
        self.cut = False

    def __iter__(self):
        for raw in self.lines:
            self.n += len(raw)
            if self.max_bytes is not None and self.n > self.max_bytes:
                self.cut = True
                return
            yield raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw


def _accept(p: Parsed, header: str, seq: str, query: str, rec: int) -> None:
    if not seq:
        p.skipped["empty_record"] += 1
    elif set(seq) - ALLOWED:
        p.skipped["bad_characters"] += 1
    elif match_length(seq) != len(query):
        p.skipped["match_columns_differ_from_query"] += 1
    elif not any("A" <= c <= "Z" for c in seq):
        p.skipped["no_match_residue"] += 1
    elif _ungapped(seq) == query:
        p.skipped["copy_of_query"] += 1
        if len(p.query_copies) < 200:
            p.query_copies.append(header)
    else:
        p.rows.append((header, seq, rec))


def parse_a3m_lines(lines: Iterable[bytes], query: str, *, max_rows: Optional[int] = None,
                    max_bytes: Optional[int] = None) -> Parsed:
    """A3M, one or several blocks (ColabFold writes NUL between blocks)."""
    src = _Counter(lines, max_bytes)
    p = Parsed([])
    first = True
    rec = -1
    name: Optional[str] = None
    parts: List[str] = []

    def flush() -> bool:
        nonlocal first
        assert name is not None
        if _is_consensus(name):
            p.skipped["consensus_record"] += 1
            return True
        seq = "".join(parts)
        if first:
            if seq != query:
                raise SourceMismatch(f"first record {name[:40]!r} is not the expected query "
                                     f"(length {len(seq)} vs {len(query)})")
            p.rows.append((name, seq, rec))
            first = False
            return True
        _accept(p, name, seq, query, rec)
        return max_rows is None or len(p.rows) - 1 < max_rows

    for raw in src:
        line = raw.replace("\x00", "").rstrip("\r\n")
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith(">"):
            if name is not None and not flush():
                p.truncated = True
                break
            rec += 1
            name, parts = line[1:], []
        elif name is not None:
            parts.append("".join(line.split()))
    else:
        if name is not None and not src.cut:
            flush()
    p.truncated = p.truncated or src.cut
    p.bytes_read = src.n
    if first:
        raise SourceMismatch("no query record")
    return p


class _Interleaved(Exception):
    pass


def _sto_convert(aln: str, insert: List[bool]) -> str:
    out = []
    for c, ins in zip(aln, insert):
        if c in "-.":
            if not ins:
                out.append("-")
        elif ins:
            out.append(c.lower())
        else:
            out.append(c.upper())
    return "".join(out)


def parse_sto_lines(lines: Iterable[bytes], query: str, *, max_rows: Optional[int] = None,
                    max_bytes: Optional[int] = None,
                    reopen: Optional[Callable[[], Iterable[bytes]]] = None) -> Parsed:
    """Stockholm, converted line by line when it has one block (all
    ``#=GS`` lines, then one line per sequence). A file with several blocks
    (OpenFold3 wraps wide alignments at 200 columns) is read again through
    ``reopen`` and joined in memory; then ``max_rows`` still applies but
    ``max_bytes`` does not, since every block is needed for every row."""
    try:
        return _sto_stream(lines, query, max_rows, max_bytes)
    except _Interleaved:
        if reopen is None:
            raise SourceMismatch("Stockholm file has several blocks") from None
        return _sto_blocks(reopen(), query, max_rows)


def _sto_stream(lines: Iterable[bytes], query: str, max_rows: Optional[int], max_bytes: Optional[int]) -> Parsed:
    src = _Counter(lines, max_bytes)
    p = Parsed([])
    desc: Dict[str, str] = {}
    seen: set = set()
    insert: Optional[List[bool]] = None
    width = 0
    started = False
    blank_after_rows = False
    rec = -1
    for line in src:
        line = line.rstrip("\r\n")
        if not started:
            if not line.strip():
                continue
            if not line.lstrip("﻿").startswith("# STOCKHOLM"):
                raise SourceMismatch("does not start with '# STOCKHOLM'")
            started = True
            continue
        if line.startswith("//"):
            break
        if not line.strip():
            if insert is not None:
                blank_after_rows = True
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
            p.skipped["bad_line"] += 1
            continue
        name, aln = parts
        if name in seen or blank_after_rows:
            raise _Interleaved()
        seen.add(name)
        rec += 1
        header = name + (" " + desc[name] if name in desc else "")
        if insert is None:
            insert = [c in "-." for c in aln]
            width = len(aln)
            q = _sto_convert(aln, insert)
            if q != query and len(q) < len(query) and query.startswith(q):
                raise _Interleaved()
            if q != query:
                raise SourceMismatch(f"first Stockholm row {name[:40]!r} is not the expected query")
            p.rows.append((header, q, rec))
            continue
        if len(aln) != width:
            p.skipped["width_differs"] += 1
            continue
        _accept(p, header, _sto_convert(aln, insert), query, rec)
        if max_rows is not None and len(p.rows) - 1 >= max_rows:
            p.truncated = True
            break
    p.truncated = p.truncated or src.cut
    p.bytes_read = src.n
    if insert is None:
        if src.cut:
            raise SourceMismatch("the size cap was reached before the first sequence row")
        raise SourceMismatch("no sequences")
    return p


def _sto_blocks(lines: Iterable[bytes], query: str, max_rows: Optional[int]) -> Parsed:
    order: List[str] = []
    seqs: Dict[str, List[str]] = {}
    desc: Dict[str, str] = {}
    n = 0
    cut = False
    for raw in lines:
        n += len(raw)
        line = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw).rstrip("\r\n")
        if line.startswith("//"):
            break
        if line.startswith("#=GS "):
            parts = line[5:].split(None, 2)
            if len(parts) == 3 and parts[1] == "DE":
                desc[parts[0]] = parts[2]
            continue
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        if parts[0] not in seqs:
            if max_rows is not None and len(order) > max_rows:
                cut = True
                continue
            order.append(parts[0])
            seqs[parts[0]] = []
        seqs[parts[0]].append(parts[1])
    if not order:
        raise SourceMismatch("no sequences")
    text = ["# STOCKHOLM 1.0"] + [f"#=GS {k} DE {desc[k]}" for k in order if k in desc]
    text += [f"{k} {''.join(seqs[k])}" for k in order]
    p = _sto_stream([t.encode() for t in text], query, max_rows, None)
    p.bytes_read = n
    p.truncated = p.truncated or cut
    return p
