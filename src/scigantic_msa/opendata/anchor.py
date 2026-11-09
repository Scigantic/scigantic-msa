"""Move A3M rows from a published entry's query coordinates to ours, for
near matches (an entry whose sequence is close to, but not the same as, the
query).

A column map says, for each residue position of the source query, which
position of our query it is aligned to (or None). For an exact entry the map
is the identity and a row comes out unchanged. For a near entry the map comes
from a pairwise alignment of the two queries: residues in source columns that
have no partner in our query become insertions (lowercase), our positions with
no source partner become gaps, and whatever lies outside the aligned span is
cropped.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

ColMap = List[Optional[int]]


def identity_map(n: int) -> ColMap:
    return list(range(n))


def map_from_alignment(qaln: str, taln: str, qstart: int, tstart: int, tlen: int) -> ColMap:
    """Column map from an alignment of our query (qaln) to the source query
    (taln). qstart and tstart are 0-based offsets of the first aligned
    residue."""
    cmap: ColMap = [None] * tlen
    qi, ti = qstart, tstart
    for a, b in zip(qaln, taln):
        if a != "-" and b != "-":
            cmap[ti] = qi
            qi += 1
            ti += 1
        elif a == "-":
            ti += 1
        else:
            qi += 1
    return cmap


def tokenize(row: str) -> Tuple[List[str], List[str], str]:
    """(insertion before each match column, match column chars, trailing
    insertion)."""
    ins: List[str] = []
    cols: List[str] = []
    cur: List[str] = []
    for c in row:
        if c == "-" or "A" <= c <= "Z":
            ins.append("".join(cur))
            cols.append(c)
            cur = []
        else:
            cur.append(c)
    return ins, cols, "".join(cur)


def reanchor(row: str, cmap: Sequence[Optional[int]], qlen: int) -> Optional[str]:
    """``row`` in source coordinates to our coordinates, or None when no
    residue lands on our query."""
    ins, cols, tail = tokenize(row)
    if len(cols) != len(cmap):
        raise ValueError(f"row has {len(cols)} match columns, the map has {len(cmap)}")
    out: List[str] = []
    qnext = 0
    started = False
    pending = ""
    last_mapped = -1
    any_residue = False
    for i, (before, c) in enumerate(zip(ins, cols)):
        j = cmap[i]
        if j is None:
            if started:
                pending += before + ("" if c == "-" else c.lower())
            continue
        if j < qnext:
            raise ValueError("column map is not increasing")
        if started:
            out.append(pending + before)
        elif i == 0:
            out.append(before)  # nothing cropped yet: keep a leading insertion
        pending = ""
        while qnext < j:
            out.append("-")
            qnext += 1
        out.append(c)
        if c != "-":
            any_residue = True
        qnext = j + 1
        started = True
        last_mapped = i
    if not any_residue:
        return None
    while qnext < qlen:
        out.append("-")
        qnext += 1
    if last_mapped == len(cols) - 1:
        out.append(tail)
    return "".join(out)
