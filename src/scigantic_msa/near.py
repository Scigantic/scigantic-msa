"""Reuse the MSA of a near-identical indexed sequence (a point mutant, or a
construct a few residues longer or shorter).

Everything runs on this machine. The user's sequence never leaves it:

- ``by_length``: download the index file of every sequence with length
  L-k..L+k (only the length is visible to the index host), and compare.
- ``mmseqs``: search sequences.fasta.gz (downloaded once, about 90 MB) with a
  local MMseqs2 binary. Nothing about the query is visible to anyone.

The parent's rows are rebuilt against the user's query: substituted columns
keep the hit residues, columns the query lacks become insertions, and new
query columns are gaps in every hit. Row 0 becomes the user's query, which
AlphaFold 3, Boltz and Chai require. The result is labelled ``source='near'``
with the parent hash, identity and every difference.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from .a3m import Row, format_a3m, parse_a3m
from .cache import FileLock
from .seqkey import canonical_sequence

if TYPE_CHECKING:  # pragma: no cover
    from .index import Index

__all__ = ["NearHit", "align", "remap_rows", "find_near", "mmseqs_available"]

Pair = Tuple[Optional[int], Optional[int]]


@dataclass
class NearHit:
    parent_seq_sha256: str
    parent_seq: str
    identity: float
    path: List[Pair]
    mismatches: List[Tuple[int, str, str]] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    method: str = "by_length"

    def describe(self) -> dict:
        return {
            "parent_seq_sha256": self.parent_seq_sha256,
            "identity": round(self.identity, 4),
            "mismatches": [[p, a, b] for p, a, b in self.mismatches],
            "insertions": self.insertions,
            "deletions": self.deletions,
            "method": self.method,
            "note": "rows copied from the MSA of a near-identical sequence; row 1 is your query",
        }


def align(query: str, parent: str, max_edits: int) -> Optional[Tuple[int, List[Pair]]]:
    """Banded global alignment with unit costs (substitution, insertion,
    deletion). Returns (edits, path) or None if more than ``max_edits`` edits
    are needed. ``path`` pairs are (query index, parent index), None for a gap.
    """
    m, n = len(query), len(parent)
    if abs(m - n) > max_edits:
        return None
    if m == n:
        d = sum(1 for a, b in zip(query, parent) if a != b)
        if d <= max_edits:
            return d, [(i, i) for i in range(m)]
        # A same-length pair can still be closer with an indel pair; fall through.
    band = max_edits
    INF = max_edits + 1
    # dp[i][j - i + band] for |i - j| <= band
    width = 2 * band + 1
    prev = [INF] * width
    for j in range(0, min(n, band) + 1):
        prev[j + band] = j
    back: List[bytearray] = [bytearray(width)]
    for j in range(1, min(n, band) + 1):
        back[0][j + band] = 2  # from left (parent residue deleted in query)
    for i in range(1, m + 1):
        cur = [INF] * width
        bk = bytearray(width)
        lo = max(0, i - band)
        hi = min(n, i + band)
        row_min = INF
        for j in range(lo, hi + 1):
            k = j - i + band
            best = INF
            b = 0
            if j > 0:
                # diagonal: prev row, j-1 -> index (j-1) - (i-1) + band = k
                v = prev[k] + (0 if query[i - 1] == parent[j - 1] else 1)
                if v < best:
                    best, b = v, 0
            # up: prev row, same j -> index j - (i-1) + band = k + 1
            if k + 1 < width:
                v = prev[k + 1] + 1
                if v < best:
                    best, b = v, 1
            # left: cur row, j-1 -> k - 1
            if j > 0 and k - 1 >= 0:
                v = cur[k - 1] + 1
                if v < best:
                    best, b = v, 2
            cur[k] = min(best, INF)
            bk[k] = b
            row_min = min(row_min, cur[k])
        if row_min > max_edits:
            return None
        prev = cur
        back.append(bk)
    k = n - m + band
    if not (0 <= k < width) or prev[k] > max_edits:
        return None
    edits = prev[k]
    path: List[Pair] = []
    i, j = m, n
    while i > 0 or j > 0:
        k = j - i + band
        b = back[i][k] if i > 0 else 2
        if i > 0 and j > 0 and b == 0:
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and b == 1:
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    path.reverse()
    return edits, path


def _describe(query: str, parent: str, path: Sequence[Pair]) -> Tuple[float, List[Tuple[int, str, str]], int, int]:
    same = 0
    mism: List[Tuple[int, str, str]] = []
    ins = dels = 0
    for qi, pj in path:
        if qi is not None and pj is not None:
            if query[qi] == parent[pj]:
                same += 1
            else:
                mism.append((qi + 1, parent[pj], query[qi]))
        elif qi is not None:
            ins += 1
        else:
            dels += 1
    return same / max(len(query), len(parent)), mism, ins, dels


def _tokens(row: str, n: int) -> Tuple[List[str], List[str], str]:
    """Split an A3M row into per-column (insertion before, match char) plus
    the trailing insertion."""
    ins: List[str] = []
    match: List[str] = []
    buf: List[str] = []
    for c in row:
        if c.islower():
            buf.append(c)
        else:
            ins.append("".join(buf))
            buf = []
            match.append(c)
    if len(match) != n:
        raise ValueError("row width does not match the parent")
    return ins, match, "".join(buf)


def remap_rows(parent_rows: Sequence[Row], query: str, path: Sequence[Pair]) -> List[Row]:
    """Rebuild an A3M made for the parent so it fits ``query``."""
    parent = parent_rows[0].seq
    out = [Row(parent_rows[0].header, query, "query")]
    n = len(parent)
    for r in parent_rows[1:]:
        ins, match, tail = _tokens(r.seq, n)
        parts: List[str] = []
        for qi, pj in path:
            if pj is None:
                parts.append("-")
            elif qi is None:
                parts.append(ins[pj])
                if match[pj] != "-":
                    parts.append(match[pj].lower())
            else:
                parts.append(ins[pj])
                parts.append(match[pj])
        parts.append(tail)
        out.append(Row(r.header, "".join(parts), r.db, r.taxid))
    # Validate, and drop rows that became empty.
    return _revalidate(out, query)


def _revalidate(rows: List[Row], query: str) -> List[Row]:
    text = format_a3m(rows)
    parsed = parse_a3m(text, query)
    # parse_a3m relabels db from headers; keep the labels and taxids we had.
    keep = {(r.header, r.seq): r for r in rows}
    result = []
    for p in parsed:
        orig = keep.get((p.header, p.seq))
        result.append(orig if orig is not None else p)
    return result


def _composition_close(a: str, b: str, max_edits: int) -> bool:
    ca, cb = Counter(a), Counter(b)
    diff = sum(abs(ca[c] - cb[c]) for c in set(ca) | set(cb))
    # One substitution changes the composition distance by at most 2, an indel by 1.
    return diff <= 2 * max_edits


def mmseqs_available() -> Optional[str]:
    return shutil.which(os.environ.get("SCIGANTIC_MSA_MMSEQS", "mmseqs"))


def find_near(
    index: "Index",
    query: str,
    *,
    min_identity: float = 0.95,
    max_length_diff: int = 5,
    method: str = "auto",
    exclude: Sequence[str] = (),
    limit: int = 5,
) -> List[NearHit]:
    """Indexed sequences close to ``query``, best first.

    ``method``: by_length, mmseqs, or auto (mmseqs when the binary is on PATH
    and the sequence database is already downloaded, else by_length).
    """
    q = canonical_sequence(query)
    max_edits = int(len(q) * (1.0 - min_identity) + 1e-9)
    if max_edits < 1:
        return []
    if method == "auto":
        method = "mmseqs" if mmseqs_available() and _sequences_cached(index) else "by_length"
    if method == "by_length":
        cands = _candidates_by_length(index, q, max_edits, max_length_diff)
    elif method == "mmseqs":
        cands = _candidates_mmseqs(index, q, min_identity)
    else:
        raise ValueError(f"unknown near method {method!r}")
    hits: List[NearHit] = []
    for sha, seq in cands:
        if sha in exclude or seq == q:
            continue
        if abs(len(seq) - len(q)) > max_length_diff:
            continue
        if not _composition_close(q, seq, max_edits):
            continue
        res = align(q, seq, max_edits)
        if res is None:
            continue
        _, path = res
        ident, mism, ins, dels = _describe(q, seq, path)
        if ident < min_identity:
            continue
        hits.append(NearHit(sha, seq, ident, path, mism, ins, dels, method))
    hits.sort(key=lambda h: (-h.identity, h.insertions + h.deletions, h.parent_seq_sha256))
    return hits[:limit]


def _sequences_cached(index: "Index") -> bool:
    try:
        near = index.manifest().get("near") or {}
        rel = near.get("sequences_fasta")
        if not rel:
            return False
        if index.local:
            return True
        return (index._build_cache() / rel).exists()
    except Exception:  # noqa: BLE001 - no index means no cache
        return False


def _candidates_by_length(index: "Index", q: str, max_edits: int, max_length_diff: int) -> List[Tuple[str, str]]:
    k = min(max_edits, max_length_diff)
    out: List[Tuple[str, str]] = []
    for L in range(max(1, len(q) - k), len(q) + k + 1):
        out.extend(index.by_length(L))
    return out


def _candidates_mmseqs(index: "Index", q: str, min_identity: float) -> List[Tuple[str, str]]:
    exe = mmseqs_available()
    if not exe:
        raise RuntimeError("near search with method='mmseqs' needs the mmseqs binary on PATH")
    fasta = index.sequences_fasta()
    dbdir = Path(str(fasta) + ".mmseqs")
    db = dbdir / "seqdb"
    with FileLock(dbdir / "build", timeout=3600):
        if not (dbdir / "seqdb.done").exists():
            dbdir.mkdir(parents=True, exist_ok=True)
            subprocess.run([exe, "createdb", str(fasta), str(db), "-v", "1"], check=True,
                           stdout=subprocess.DEVNULL)
            (dbdir / "seqdb.done").write_text("ok\n")
    with tempfile.TemporaryDirectory(prefix="scigantic-msa-") as tmp:
        qf = Path(tmp) / "q.fa"
        qf.write_text(f">query\n{q}\n")
        res = Path(tmp) / "res.m8"
        subprocess.run(
            [exe, "easy-search", str(qf), str(db), str(res), str(Path(tmp) / "t"),
             "--min-seq-id", str(max(0.0, min_identity - 0.02)), "-c", "0.9", "--cov-mode", "0",
             "--max-seqs", "50", "--format-output", "target,fident,tseq", "-v", "1"],
            check=True, stdout=subprocess.DEVNULL,
        )
        out = []
        for line in res.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                out.append((parts[0], parts[2].strip().upper()))
        return out
