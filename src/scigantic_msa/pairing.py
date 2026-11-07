"""Pair the rows of several chains by species, only where the data says the
species.

A row's species comes from an NCBI taxid (ColabFold taxonomy output,
``TaxID=`` or ``OX=`` in the header) or a UniProt species mnemonic. UniRef90
rows for clusters with more than one member carry the cluster's common taxon,
not a species, and are never used. Environmental and metagenomic rows have no
species and stay unpaired.

For each species present in every chain that has species information, the
best (first) row of each of those chains is taken. Rows are ordered by the
sum of their ranks. A chain with no species information at all does not stop
the others pairing: it gets ``None`` in every paired row, and each writer
fills that in the way its model expects (gap rows, or no key).
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
    common = set(best[with_species[0]])
    for ci in with_species[1:]:
        common &= set(best[ci])
    keys = sorted(common, key=lambda k: (sum(best[ci][k][0] for ci in with_species), k))
    if max_pairs is not None:
        keys = keys[:max_pairs]
    rows: List[List[Optional[Row]]] = [
        [best[ci][k][1] for k in keys] if ci in with_species else [None] * len(keys)
        for ci in range(len(msas))
    ]
    if not without:
        msg = (f"paired {len(keys)} rows by species across {len(msas)} chains" if keys
               else "no species is shared by all chains; the chains get unpaired MSAs")
    elif not keys:
        msg = (f"no species is shared by {_chains(label, with_species)}, and {no_species}; "
               "the chains get unpaired MSAs")
    else:
        msg = (f"paired {len(keys)} rows by species across {_chains(label, with_species)}; {no_species}, "
               f"so {'they are' if plural else 'it is'} not paired")
    return Pairing(rows, keys, without, msg)


def _count(label: Sequence[str], idx: Sequence[int]) -> int:
    return sum(len(label[i].split(", ")) for i in idx)


def _chains(label: Sequence[str], idx: Sequence[int]) -> str:
    """"chain C" or "chains A, B" for a message. A label may name several
    chains that share a sequence ("B, D")."""
    return ("chain " if _count(label, idx) == 1 else "chains ") + ", ".join(label[i] for i in idx)
