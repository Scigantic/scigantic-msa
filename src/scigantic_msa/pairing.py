"""Pair the rows of several chains by species, only where the data says the
species.

A row's species comes from an NCBI taxid (ColabFold taxonomy output,
``TaxID=`` or ``OX=`` in the header) or a UniProt species mnemonic. UniRef90
rows for clusters with more than one member carry the cluster's common taxon,
not a species, and are never used. Environmental and metagenomic rows have no
species and stay unpaired.

Every species found in two or more chains is paired: each chain that has
it gives its best (first) row of that species, and a chain without it gets
``None`` in that paired row. Rows shared by more chains come first, then by
the sum of their ranks. So a chain from another organism, or one with no
species information at all, never costs the others their pairing. Each
writer fills in ``None`` the way its model expects (gap rows, no key, or the
chain's next unpaired row).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .a3m import Row, ungapped
from .alignment import Alignment, species_key

__all__ = ["Pairing", "pair_by_species"]


@dataclass
class Pairing:
    """Result of pairing. ``rows[c]`` are chain c's paired rows (no query),
    all the same length; a chain in ``chains_without_species`` has ``None``
    in every row. ``message`` says what was and was not possible."""

    rows: List[List[Optional[Row]]]
    keys: List[str]
    chains_without_species: List[int] = field(default_factory=list)
    message: str = ""

    @property
    def n_paired(self) -> int:
        return len(self.keys)


def _pool(msa: Alignment) -> List[Row]:
    """Candidate rows for pairing. Rows with the same residues as the query or
    as an earlier row are left out: they add nothing, and Boltz and AlphaFold 3
    drop repeated rows, which would silently break the pair."""
    src = msa.uniprot_rows if msa.uniprot_rows and len(msa.uniprot_rows) > 1 else msa.rows
    seen = {msa.query}
    out = []
    for r in src[1:]:
        k = ungapped(r.seq)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def pair_by_species(msas: Sequence[Alignment], max_pairs: Optional[int] = 8192,
                    names: Optional[Sequence[str]] = None) -> Pairing:
    """Pair ``msas`` (one per distinct chain) by species. ``names`` are what
    the message calls each one; the default is "1", "2", ..."""
    label = list(names) if names is not None else [str(i + 1) for i in range(len(msas))]
    if len(msas) < 2:
        return Pairing([[] for _ in msas], [], [], "one chain: nothing to pair")
    best: List[Dict[str, tuple]] = []
    without: List[int] = []
    for ci, m in enumerate(msas):
        d: Dict[str, tuple] = {}
        for rank, r in enumerate(_pool(m)):
            k = species_key(r)
            if k is None or k in d:
                continue
            d[k] = (rank, r)
        if not d:
            without.append(ci)
        best.append(d)
    with_species = [ci for ci in range(len(msas)) if ci not in without]
    no_species = ""
    if without:
        plural = _count(label, without) > 1
        no_species = (_chains(label, without) + (" have" if plural else " has")
                      + " no rows with species information (environmental or clustered hits only)")
    if len(with_species) < 2:
        return Pairing([[] for _ in msas], [], without,
                       f"no pairing: {no_species}. The chains get unpaired MSAs.")
    # Keys must be of the same kind (taxid vs mnemonic) to compare.
    holders: Dict[str, List[int]] = {}
    for ci in with_species:
        for k in best[ci]:
            holders.setdefault(k, []).append(ci)
    shared = {k: cs for k, cs in holders.items() if len(cs) > 1}
    keys = sorted(shared, key=lambda k: (-len(shared[k]), sum(best[ci][k][0] for ci in shared[k]), k))
    if max_pairs is not None:
        keys = keys[:max_pairs]
    rows: List[List[Optional[Row]]] = [
        [best[ci][k][1] if k in best[ci] else None for k in keys] for ci in range(len(msas))
    ]
    paired = sorted({ci for k in keys for ci in shared[k]})
    lonely = [ci for ci in with_species if ci not in paired]
    if not keys:
        what = ("no species is shared by any two chains" if not without
                else f"no species is shared by {'any two of ' if len(with_species) > 2 else ''}"
                     f"{_chains(label, with_species)}, and {no_species}")
        return Pairing(rows, keys, without, f"{what}; the chains get unpaired MSAs")
    if len(paired) == len(msas):
        msg = f"paired {len(keys)} rows by species across {_count(label, range(len(msas)))} chains"
    else:
        msg = f"paired {len(keys)} rows by species across {_chains(label, paired)}"
    groups: Dict[tuple, int] = {}
    for k in keys:
        groups[tuple(shared[k])] = groups.get(tuple(shared[k]), 0) + 1
    if len(groups) > 1:
        msg += " (" + ", ".join(f"{n} with {_chains(label, list(g))}" for g, n in groups.items()) + ")"
    left = []
    if lonely:
        many = _count(label, lonely) > 1
        left.append(f"{_chains(label, lonely)} {'share' if many else 'shares'} no species with another chain")
    if without:
        left.append(no_species)
    if left:
        many = _count(label, lonely + without) > 1
        msg += "; " + " and ".join(left) + f", so {'they are' if many else 'it is'} not paired"
    return Pairing(rows, keys, without, msg)


def _count(label: Sequence[str], idx: Sequence[int]) -> int:
    return sum(len(label[i].split(", ")) for i in idx)


def _chains(label: Sequence[str], idx: Sequence[int]) -> str:
    """"chain C" or "chains A, B" for a message. A label may name several
    chains that share a sequence ("B, D"); the chains are named in their
    own order, so A, B, A, C reads "chains A, B, C, D", not "A, C, B, D"."""
    names = sorted((n for i in idx for n in label[i].split(", ")), key=lambda n: (len(n), n))
    return ("chain " if len(names) == 1 else "chains ") + ", ".join(names)
