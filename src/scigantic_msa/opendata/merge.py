"""Merge a server search with open rows, and pair the chains of a complex.

Merge rules, in order:

1. A row with the query's residues is dropped.
2. Rows with the same residues (gaps removed) are one row. The first copy
   is kept: ColabFold rows come first, then open rows in source order. The
   kept row keeps its own species, or none: an environmental or
   metagenomic row never takes the species of a copy from UniProt.
3. Rows of the same sequence entry (same accession) that cover the same
   part of it are one row, and the copy with the most aligned residues is
   kept. "The same part" means the stated spans of the entry (``/12-200``
   in Stockholm names, the target start and end in ColabFold headers)
   overlap by at least half of the shorter one; when a row states no span,
   its span on the query is used. Rows of one entry that cover different
   parts of it (repeats, several domains) all stay. Exact-string dedup
   alone left such copies in and inflated Neff by up to 38% on the
   benchmark this was measured on.

Order: the ColabFold rows first, in the server's order, then the open rows,
taking one row from each open file in turn (each file in its own order,
best hits first). Models that read only the first N rows therefore see the
whole server result before any open row, and open rows matter most where
the server result is short.
"""
from __future__ import annotations

import dataclasses
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..a3m import Row, ungapped
from ..alignment import species_key
from ..pairing import Pairing, pair_by_species, _chains

__all__ = ["merge_rows", "MergeCounts", "neff_estimate", "accession", "combine_pairs", "PairPlan",
           "organism_rule"]

_UNIPROT = re.compile(r"^(?:tr|sp)\|([^|]+)\|")
_UNIREF = re.compile(r"^UniRef(?:100|90|50)_([^\s/|]+)")
_SPAN = re.compile(r"/(\d+)-(\d+)$")


def accession(header: str) -> str:
    """The id of the sequence entry a row is from, without a span."""
    first = header.split(None, 1)[0] if header.strip() else ""
    first = first.split("\t", 1)[0]
    m = _UNIPROT.match(first)
    if m:
        return m.group(1)
    m = _UNIREF.match(first)
    if m:
        return m.group(1)
    return _SPAN.sub("", first)


def _source_span(header: str) -> Optional[Tuple[int, int]]:
    first = header.split(None, 1)[0] if header.strip() else ""
    m = _SPAN.search(first.split("\t", 1)[0])
    if m:
        return int(m.group(1)), int(m.group(2))
    parts = header.split("\t")
    if len(parts) >= 10 and parts[7].strip().isdigit() and parts[8].strip().isdigit():
        return int(parts[7]), int(parts[8])
    return None


def _query_span(seq: str) -> Tuple[int, int, int]:
    """(first, last, residues) over the match columns of a row."""
    first = last = -1
    n = 0
    i = 0
    for c in seq:
        if c == "-":
            i += 1
        elif "A" <= c <= "Z":
            if first < 0:
                first = i
            last = i
            n += 1
            i += 1
    return first, last, n


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return False
    return (hi - lo + 1) * 2 >= min(a[1] - a[0] + 1, b[1] - b[0] + 1)


@dataclass
class MergeCounts:
    """What the merge did, for the summary and the attribution file."""

    server_rows: int = 0
    open_rows_offered: int = 0
    same_residues: int = 0
    same_entry_span: int = 0
    kept_server: int = 0
    kept_open: Counter = field(default_factory=Counter)
    query_copies: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"server_rows": self.server_rows, "open_rows_offered": self.open_rows_offered,
                "dropped_same_residues": self.same_residues, "dropped_same_entry_and_span": self.same_entry_span,
                "dropped_query_copies": self.query_copies, "kept_server_rows": self.kept_server,
                "kept_open_rows": dict(sorted(self.kept_open.items())), "kept_open_total": sum(self.kept_open.values())}


def _interleave(rows: Sequence[Row]) -> List[Row]:
    files: Dict[Tuple[str, str, str, str], List[Row]] = {}
    for r in rows:
        p = r.extra.get("prov")
        k = (p.source, p.role, p.entry, p.anchor) if p else ("", "", "", "")
        files.setdefault(k, []).append(r)
    lists = list(files.values())
    out: List[Row] = []
    i = 0
    while lists:
        nxt = []
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
                nxt.append(lst)
        lists = [lst for lst in nxt if i + 1 < len(lst)]
        i += 1
    return out


def _also(kept: Row, dropped: Row) -> None:
    """Note on the kept copy (merge's own) where the dropped copy came from."""
    p = dropped.extra.get("prov")
    if p is not None:
        kept.extra["also"] = set(kept.extra.get("also", ())) | {f"{p.source}:{p.role}"}


def merge_rows(query: str, server: Sequence[Row], open_rows: Sequence[Row]) -> Tuple[List[Row], MergeCounts]:
    """The merged rows (query first) and what happened. ``server`` and
    ``open_rows`` exclude the query; every row should carry
    ``extra['prov']``. The rows returned are copies; the input rows are
    not changed."""
    cnt = MergeCounts(server_rows=len(server), open_rows_offered=len(open_rows))
    by_res: Dict[str, Row] = {}
    cands: List[Row] = []
    for r in list(server) + _interleave(open_rows):
        k = ungapped(r.seq)
        if k == query:
            cnt.query_copies += 1
            continue
        old = by_res.get(k)
        if old is not None:
            cnt.same_residues += 1
            _also(old, r)
            continue
        r = dataclasses.replace(r, extra=dict(r.extra))
        by_res[k] = r
        cands.append(r)
    # Same entry, same part of it: keep the copy with the most residues.
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(cands):
        groups.setdefault(accession(r.header), []).append(i)
    drop: Set[int] = set()
    for acc, idx in groups.items():
        if len(idx) < 2 or not acc:
            continue
        info = []
        for i in idx:
            f, la, n = _query_span(cands[i].seq)
            info.append((i, _source_span(cands[i].header), (f, la), n))
        info.sort(key=lambda t: (-t[3], t[0]))
        kept: List[Tuple[int, Optional[Tuple[int, int]], Tuple[int, int], int]] = []
        for t in info:
            same = False
            for kk in kept:
                if t[1] is not None and kk[1] is not None:
                    same = _overlap(t[1], kk[1])
                else:
                    same = _overlap(t[2], kk[2])
                if same:
                    _also(cands[kk[0]], cands[t[0]])
                    break
            if same:
                drop.add(t[0])
            else:
                kept.append(t)
    cnt.same_entry_span = len(drop)
    out = [r for i, r in enumerate(cands) if i not in drop]
    for r in out:
        p = r.extra.get("prov")
        if p is None or p.source in ("colabfold", "nvidia"):
            cnt.kept_server += 1
        else:
            cnt.kept_open[p.source] += 1
    return out, cnt


# --------------------------------------------------------------------------- depth


def _codes(query: str, rows: Sequence[Row]) -> List[int]:
    out = []
    for s in [query] + [r.seq for r in rows]:
        cols = "".join(c for c in s if c == "-" or "A" <= c <= "Z")
        out.append(int.from_bytes(cols.encode("ascii", "replace"), "big"))
    return out


def neff_estimate(query: str, rows: Sequence[Row], *, sample: int = 128, against: int = 8192, seed: int = 0,
                  threshold: float = 0.8) -> float:
    """Neff at 80% identity: the sum over rows (and the query) of 1 / the
    number of rows at 80% identity or more to it, identity counted over all
    query columns with gap matching gap. Exact up to ``sample`` rows. Above
    that it is estimated from a fixed random sample of rows, each compared
    with at most ``against`` other rows, so it takes a second or two at any
    depth and can be off by some percent. Use it to compare alignments, not
    to report to three digits."""
    L = len(query)
    codes = _codes(query, rows)
    n = len(codes)
    need = threshold * L
    rng = random.Random(seed)
    idx = list(range(n)) if n <= sample else rng.sample(range(n), sample)
    pool_idx = list(range(n)) if n <= against + 1 else rng.sample(range(n), against + 1)
    in_pool = set(pool_idx)
    total = 0.0
    for i in idx:
        a = codes[i]
        others = 0
        for j in pool_idx:
            if j != i and (a ^ codes[j]).to_bytes(L, "big").count(0) >= need:
                others += 1
        size = len(pool_idx) - (1 if i in in_pool else 0)
        total += 1.0 / (1.0 + others * (n - 1) / max(size, 1))
    return total * (n / len(idx))


# --------------------------------------------------------------------------- pairing


def organism_rule(organisms: Sequence[Set[str]], names: Sequence[str]) -> Tuple[bool, str]:
    """Whether open species pairs may be added: only when every chain's own
    organism is known and some species has the exact sequence of every
    chain. (True, why) or (False, why)."""
    unknown = [n for n, o in zip(names, organisms) if not o]
    if unknown:
        many = sum(len(u.split(", ")) for u in unknown) > 1
        return False, (f"the organism of {_label(unknown)} is not known, so no species pairs from open data "
                       f"are added ({'their' if many else 'its'} exact sequence is in no UniProt entry or "
                       "OpenFold3 UniProt hit)")
    common = set.intersection(*[set(o) for o in organisms])
    if not common:
        what = "; ".join(f"{n}: {', '.join(sorted(o)[:3])}{' and more' if len(o) > 3 else ''}"
                         for n, o in zip(names, organisms))
        return False, ("the chains come from different organisms, so no species pairs from open data are "
                       f"added (pairs by species would not be coevolved partners): {what}")
    return True, f"all chains are found in {sorted(common)[0]}{' and others' if len(common) > 1 else ''}"


def _label(names: Sequence[str]) -> str:
    flat = [x for n in names for x in n.split(", ")]
    return ("chain " if len(flat) == 1 else "chains ") + ", ".join(flat)


@dataclass
class PairPlan:
    """The pairs of one complex: the search server's own pairs first, then
    extra pairs by species from open rows."""

    pairing: Pairing
    server_pairs: int
    open_pairs: int
    rule: str


def _acc_keys(m_rows: Sequence[Row]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in m_rows[1:]:
        k = species_key(r)
        if k:
            out.setdefault(accession(r.header), k)
    return out


def combine_pairs(queries: Sequence[str], merged: Sequence[Sequence[Row]], open_only: Sequence[Sequence[Row]],
                  names: Sequence[str], *, server_pairs: Optional[List[List[Optional[Row]]]] = None,
                  base: Optional[Pairing] = None, add_open: bool = True, rule: str = "",
                  max_pairs: Optional[int] = None) -> PairPlan:
    """Pairs for the distinct chains ``queries``.

    ``server_pairs``: rows of the server's paired search, per chain, all the
    same length (None where a chain has no row). ``base``: pairs made
    another way when the server has no paired search. ``merged``: each
    chain's merged rows (query first), used to find the species of the
    server's pairs. ``open_only``: each chain's open rows as the merge kept
    them (query first), the pool for extra pairs, so open pairs are rows of
    the merged MSA. ``add_open``: the organism rule's answer."""
    nc = len(queries)
    rows: List[List[Optional[Row]]] = [[] for _ in range(nc)]
    keys: List[str] = []
    msg_parts: List[str] = []
    n_server = 0
    if server_pairs is not None:
        # The server's pairs are kept as the server made them. Only a pair
        # that repeats an earlier pair in every chain, or the query pair
        # itself, is left out. A row that repeats in one chain stays:
        # AlphaFold 3, ColabFold, OpenFold3, Protenix and Chai-1 keep it, and
        # Boltz, which drops it, only loses that pair.
        seen_pairs: Set[Tuple[Optional[str], ...]] = {tuple(queries)}
        total = len(server_pairs[0]) if server_pairs else 0
        for k in range(total):
            picked = [server_pairs[c][k] for c in range(nc)]
            res = tuple(ungapped(r.seq) if r is not None else None for r in picked)
            if all(x is None or x == q for x, q in zip(res, queries)) or res in seen_pairs:
                continue
            seen_pairs.add(res)
            for c in range(nc):
                rows[c].append(picked[c])
        n_server = len(rows[0])
        keys = [f"server:{k + 1}" for k in range(n_server)]
        msg_parts.append(f"{n_server} rows from the server's paired search"
                         + (f" ({total - n_server} more repeat another pair or the query and are left out)"
                            if total > n_server else ""))
    elif base is not None and base.n_paired:
        n_server = base.n_paired
        rows = [list(r) for r in base.rows]
        keys = list(base.keys)
        msg_parts.append(f"{n_server} rows paired by species from the server's rows")
    # Species and entries the server pairs already cover.
    acc_key = [_acc_keys(m) for m in merged]
    have_species: Set[str] = set()
    have_res: List[Set[str]] = [set() for _ in range(nc)]
    have_acc: List[Set[str]] = [set() for _ in range(nc)]
    for k in range(len(keys)):
        found = keys[k] if not keys[k].startswith("server:") else None
        for c in range(nc):
            r = rows[c][k]
            if r is None:
                continue
            have_res[c].add(ungapped(r.seq))
            a = accession(r.header)
            have_acc[c].add(a)
            found = found or species_key(r) or acc_key[c].get(a)
        if found:
            have_species.add(found)
    n_open = 0
    if add_open:
        from ..alignment import Alignment

        pools = [Alignment(query=q, rows=list(rs), source="file") for q, rs in zip(queries, open_only)]
        extra = pair_by_species(pools, max_pairs=None, names=list(names))
        for k, key in enumerate(extra.keys):
            if key in have_species:
                continue
            picked = [extra.rows[c][k] for c in range(nc)]
            if any(r is not None and (ungapped(r.seq) in have_res[c] or accession(r.header) in have_acc[c])
                   for c, r in enumerate(picked)):
                continue
            for c in range(nc):
                rows[c].append(picked[c])
            keys.append(key)
            n_open += 1
        if n_open:
            msg_parts.append(f"{n_open} more by species from open data")
        else:
            msg_parts.append("no further species pairs from open data")
    if max_pairs is not None and len(keys) > max_pairs:
        keys = keys[:max_pairs]
        rows = [r[:max_pairs] for r in rows]
    lonely = [c for c in range(nc) if not any(r is not None for r in rows[c])]
    who = _chains(list(names), [c for c in range(nc) if c not in lonely]) if len(lonely) < nc else ""
    if keys:
        msg = f"paired {len(keys)} rows across {who}: " + " and ".join(msg_parts)
        if server_pairs is not None and n_server:
            msg += ". The server's paired rows are written in addition to each chain's own rows"
    else:
        msg = "no paired rows; the chains get unpaired MSAs"
    if rule:
        msg += f". {rule[0].upper()}{rule[1:]}"
    return PairPlan(Pairing(rows, keys, lonely if keys else list(range(nc)), msg), n_server, n_open, rule)
