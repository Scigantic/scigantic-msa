"""Defects found by the 0.2.6 stress test, all in the AlphaFold 3 order search.

AlphaFold 3 files are read the way AlphaFold 3 reads a user MSA
(parse_fasta, deduplicate=False): every record, gap-only rows included.
"""
import json
import random

from scigantic_msa import write_model
from scigantic_msa.a3m import Row
from scigantic_msa.alignment import Alignment
from scigantic_msa.pairing import pair_by_species
from scigantic_msa.writers import af3_layout

AA = "ACDEFGHIKLMNPQRSTVWY"


def build(spec, spare, L=30, seed=0):
    """One Alignment per chain: a row per taxid in spec[chain] (in order),
    then spare[chain] environmental rows with no species. Every row is a
    distinct 3-point mutant of the query."""
    rng = random.Random(seed)
    out = []
    for c, taxa in spec.items():
        q = "".join(rng.choice(AA) for _ in range(L))
        rows, seen = [Row("101", q, "query")], {q}

        def mk(q=q, seen=seen):
            while True:
                s = list(q)
                for _ in range(3):
                    s[rng.randrange(L)] = rng.choice(AA)
                s = "".join(s)
                if s not in seen:
                    seen.add(s)
                    return s

        for t in taxa:
            rows.append(Row(f"UniRef100_{c}{t}\t{t}", mk(), "uniref", str(t)))
        for k in range(spare.get(c, 0)):
            rows.append(Row(f"env_{c}{k}", mk(), "env"))
        out.append(Alignment(query=q, rows=rows, source="remote", provenance={}))
    return out


def af3_rows(text):
    seqs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            seqs.append("")
        elif line:
            seqs[-1] += line
    return seqs


def af3_gap_rows(tmp_path, chains):
    write_model("af3", chains, tmp_path, name="j", quiet=True)
    doc = json.loads((tmp_path / "af3" / "j.json").read_text())
    return {p["protein"]["id"] if isinstance(p["protein"]["id"], str) else ", ".join(p["protein"]["id"]):
            sum(1 for s in af3_rows(p["protein"]["unpairedMsa"]) if s and set(s) == {"-"})
            for p in doc["sequences"]}


# ---------------------------------------------------------------- defect 1


def test_af3_lonely_chain_does_not_force_gap_rows(tmp_path):
    """Defect 1 (medium). writers.af3_layout.group_cost charges a chain that
    holds no pair (one that shares no species, which is live) as if every
    group but the last came before its last pair: max() over an empty set
    gives last=-1 and perm[:-1] is every group but the last. cost() skips
    such chains, group_cost does not, so the search compares wrong numbers
    and keeps an order that gives gap rows a reorder would avoid.

    A+B share 20 species (Y), B+C share 100 (X). C has no rows of its own,
    A has 100. Putting X first costs nothing (A fills X with its own rows);
    Y first gives C 20 gap rows. Without chain D the library finds the
    zero-gap order. Add D, one row of a species no one else has (a binder or
    a chain from another organism), and C gets 20 gap rows."""
    Y = list(range(1, 21))
    X = list(range(100, 200))
    spec = {"A": Y, "B": Y + X, "C": X}
    spare = {"A": 100, "B": 0, "C": 0}
    without = build(spec, spare)
    assert sum(af3_gap_rows(tmp_path / "x", without).values()) == 0
    with_d = build(dict(spec, D=[999]), spare)
    gaps = af3_gap_rows(tmp_path / "y", with_d)
    assert sum(gaps.values()) == 0, f"gap rows per chain with a lonely chain D present: {gaps}"


# ---------------------------------------------------------------- defect 2


def test_af3_more_than_seven_groups_finds_the_zero_gap_order(tmp_path):
    """Defect 2 (medium-low). With more than 7 groups af3_layout tries only
    two orders: the given one, and groups sorted by their shallowest
    chain's spare rows (ties keep the given order). The changelog says the
    pairs are put in the order that needs the fewest gap rows.

    8 groups: 6 groups of 10 species over chains D-H (each holding D, which
    has 3 spare rows), 1 species shared by A+B, 1 by C+D. Order A+B, C+D,
    then the rest needs no gap rows (A, B lead; C needs 1 filler and has 5;
    D needs 1 and has 3). Both tried orders put C+D after the 60, so chain
    C gets 56 gap rows: 56 of its 67 rows are gaps in its profile."""
    big = [("D", "E", "F"), ("D", "E", "G"), ("D", "F", "G"), ("D", "E", "H"), ("D", "F", "H"), ("D", "G", "H")]
    spec = {c: [] for c in "ABCDEFGH"}
    t = 1
    for g in big:
        for _ in range(10):
            for c in g:
                spec[c].append(t)
            t += 1
    for c in "AB":
        spec[c].append(t)
    t += 1
    for c in "CD":
        spec[c].append(t)
    spare = {"A": 0, "B": 70, "C": 5, "D": 3, "E": 70, "F": 70, "G": 70, "H": 70}
    chains = build(spec, spare)
    pr = pair_by_species(chains)
    groups = {tuple(c for c in range(8) if pr.rows[c][k] is not None) for k in range(pr.n_paired)}
    assert len(groups) == 8
    gaps = af3_gap_rows(tmp_path, chains)
    assert sum(gaps.values()) == 0, f"gap rows per chain: {gaps}"


# ---------------------------------------------------------------- defect 3


def test_af3_gap_notice_counts_chains_not_sequences():
    """Defect 3 (cosmetic). AF3Layout.note picks singular or plural from the
    number of distinct sequences, as the pairing notice did before 0.2.6.
    A, B, A, C where A needs a gap row reads "this chain has too few rows
    of its own and gets gap rows instead (chains A, C: 1), which count in
    its profile". It also does not say the 1 is per chain."""
    a, b, c = build({"A": ["2"], "B": ["1", "2"], "C": ["1"]}, {}, seed=1)
    msas = [a, b, c]
    names = ["A, C", "B", "D"]
    lay = af3_layout(msas, pair_by_species(msas, names=names), names)
    note = lay.note()
    assert "chains A, C" in note, note
    assert "this chain has" not in note, note
