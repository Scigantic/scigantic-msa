"""Pair the rows of several chains by species, only where the data says the
species.

A row's species comes from an NCBI taxid (ColabFold taxonomy output,
``TaxID=`` or ``OX=`` in the header) or a UniProt species mnemonic. UniRef90
rows for clusters with more than one member carry the cluster's common taxon,
not a species, and are never used. Environmental and metagenomic rows have no
species and stay unpaired.

For each species present in every chain, the best (first) row of each chain
is taken. Rows are ordered by the sum of their ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .a3m import Row, ungapped
from .msa import MSA, species_key

__all__ = ["Pairing", "pair_by_species"]


@dataclass
class Pairing:
    """Result of pairing. ``rows[c]`` are chain c's paired rows (no query),
    all the same length. ``message`` says what was and was not possible."""

    rows: List[List[Row]]
    keys: List[str]
    chains_without_species: List[int] = field(default_factory=list)
    message: str = ""

    @property
    def n_paired(self) -> int:
        return len(self.keys)


def _pool(msa: MSA) -> List[Row]:
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


def pair_by_species(msas: Sequence[MSA], max_pairs: Optional[int] = 8192) -> Pairing:
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
    if without:
        return Pairing(
            [[] for _ in msas], [], without,
            "no pairing: chain(s) " + ", ".join(str(i + 1) for i in without)
            + " have no rows with species information (environmental or clustered hits only). "
            "The chains get unpaired MSAs.",
        )
    # Keys must be of the same kind (taxid vs mnemonic) to compare.
    common = set(best[0])
    for d in best[1:]:
        common &= set(d)
    keys = sorted(common, key=lambda k: (sum(d[k][0] for d in best), k))
    if max_pairs is not None:
        keys = keys[:max_pairs]
    rows = [[d[k][1] for k in keys] for d in best]
    if keys:
        msg = f"paired {len(keys)} rows by species across {len(msas)} chains"
    else:
        msg = "no species is shared by all chains; the chains get unpaired MSAs"
    return Pairing(rows, keys, [], msg)
