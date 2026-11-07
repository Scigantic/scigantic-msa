"""A chain with no species information (environmental or clustered hits
only) must not cost the other chains of a complex their pairing.

Each model's own convention for a chain that has no row for a paired species:

- AlphaFold 3 joins unpaired rows across chains by row index; the chain keeps
  its own unpaired MSA (``msa_pairing._align_species`` would gap it, but
  that only happens for ``pairedMsa``).
- Boltz ``construct_paired_msa`` fills a chain missing from a key with its
  next unpaired row; the chain gets key -1 everywhere.
- Chai-1 ``pair_and_merge_msas`` pairs a key only when it is in every chain
  with an MSA, so the chain gets one gap row per key.
- ColabFold treats a row as paired when two or more chains have residues in
  it (``unserialize_msa``); the chain is gaps in the paired rows.
- OpenFold3 needs the same number of paired rows in every chain
  (``create_paired_from_precomputed``); the chain gets query plus gap rows.
"""
import json

import pytest

from conftest import Q_7M22A, Q_7M22C, Q_LITEFOLD, real_alignment
from scigantic_msa.a3m import Row, parse_a3m, parse_colabfold_complex
from scigantic_msa.adapters import AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix
from scigantic_msa.alignment import Alignment, species_key
from scigantic_msa.pairing import pair_by_species
from scigantic_msa.writers import write_chai, write_colabfold


def synthetic(query, hits, taxids=None):
    rows = [Row("101", query, "query")]
    for i, s in enumerate(hits):
        rows.append(Row(f"UniRef100_H{i}", s, "uniref", (taxids or {}).get(i)))
    return Alignment(query=query, rows=rows, source="remote", provenance={"backend": "colabfold"})


@pytest.fixture
def trio():
    """A and B share species 9606 and 10090; C has environmental hits only."""
    a = synthetic("MKVLAGWERT", ["MRVLAGWERT", "MKILAGWERT", "MKVAAGWERT"], {0: "9606", 1: "10090", 2: "7227"})
    b = synthetic("GHSEQKLLPA", ["GHTEQKLLPA", "GASEQKLLPA"], {0: "10090", 1: "9606"})
    c = synthetic("PPWWGGYYST", ["PPWWGGYYSA", "PPWWGGYFST"])
    return a, b, c


@pytest.fixture
def real_trio():
    """Two real chains with UniProt rows, and a real ColabFold MSA whose 314
    rows carry no species."""
    c = real_alignment(Q_LITEFOLD)
    assert all(species_key(r) is None for r in c.rows)
    return real_alignment(Q_7M22A, with_uniprot=True), real_alignment(Q_7M22C, with_uniprot=True), c


def test_pairing_keeps_the_chains_that_have_species(trio):
    pr = pair_by_species(trio)
    assert pr.n_paired == 2 and pr.chains_without_species == [2]
    assert [species_key(r) for r in pr.rows[0]] == [species_key(r) for r in pr.rows[1]]
    assert pr.rows[2] == [None, None]
    assert "paired 2 rows by species" in pr.message and "3" in pr.message


def test_pairing_with_one_chain_of_species_left_still_says_so():
    a = synthetic("MKVLAGWERT", ["MRVLAGWERT"], {0: "9606"})
    b = synthetic("GHSEQKLLPA", ["GHTEQKLLPA"])
    c = synthetic("PPWWGGYYST", ["PPWWGGYYSA"])
    pr = pair_by_species([a, b, c])
    assert pr.n_paired == 0 and pr.chains_without_species == [1, 2]
    assert "no pairing" in pr.message


def test_real_trio_keeps_the_heterodimer_pairs(real_trio):
    a, b, c = real_trio
    assert pair_by_species([a, b, c]).n_paired == pair_by_species([a, b]).n_paired == 8


def _segments(row, lens):
    """Split a full-width row into one A3M segment per chain, as ColabFold's
    ``unserialize_msa`` does (lowercase letters are insertions)."""
    out, cur, n, k = [], "", 0, 0
    for ch in row:
        cur += ch
        if not ch.islower():
            n += 1
            if n == lens[k]:
                out.append(cur)
                cur, n, k = "", 0, k + 1
    return out


def test_colabfold_pairs_two_chains_and_gaps_the_third(trio, tmp_path):
    a, b, c = trio
    pr = pair_by_species(trio)
    p = write_colabfold(trio, tmp_path / "x.a3m")
    lens, _, queries, records = parse_colabfold_complex(p.read_text())
    assert queries == [a.query, b.query, c.query]
    # ColabFold treats a row as paired when two or more chains have residues.
    paired = [s for _, s in records[1:] if sum(any(x.isupper() for x in seg) for seg in _segments(s, lens)) > 1]
    assert len(paired) == pr.n_paired == 2
    for k, s in enumerate(paired):
        sa, sb, sc = _segments(s, lens)
        assert (sa, sb) == (pr.rows[0][k].seq, pr.rows[1][k].seq)
        assert sc == "-" * len(c.query)


def test_boltz_keys_pair_two_chains_and_leave_the_third_unpaired(trio, tmp_path):
    Boltz.prepare(list(trio), out_dir=tmp_path, quiet=True)
    keys = []
    for m in trio:
        lines = (tmp_path / f"{m.seq_sha256}.csv").read_text().splitlines()
        keys.append([l.split(",")[0] for l in lines[2:]])
        assert len(lines) - 1 == len(m.rows)
    assert set(keys[0]) - {"-1"} == set(keys[1]) - {"-1"} == {"1", "2"}
    assert set(keys[2]) == {"-1"}


def test_chai_gives_the_third_chain_gap_rows_for_each_key(trio, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    a, b, c = trio
    write_chai(trio, tmp_path)
    tabs = [pq.read_table(tmp_path / f"{m.seq_sha256}.aligned.pqt").to_pydict() for m in trio]
    ka, kb, kc = ({k for k in t["pairing_key"] if k} for t in tabs)
    assert ka == kb == kc == {"1", "2"}
    tc = tabs[2]
    gaps = [i for i, k in enumerate(tc["pairing_key"]) if k]
    assert all(tc["sequence"][i] == "-" * len(c.query) for i in gaps)
    # bfd_uniclust has no quota in Chai, so no gap row is cut before pairing.
    assert all(tc["source_database"][i] == "bfd_uniclust" for i in gaps)
    # The chain's own rows are all still there, unkeyed.
    own = [s for s, k in zip(tc["sequence"], tc["pairing_key"]) if not k]
    assert own == [r.seq for r in c.rows]


def test_af3_puts_paired_rows_first_for_two_chains(trio):
    a, b, c = trio
    pr = pair_by_species(trio)
    doc = AlphaFold3.prepare(list(trio), quiet=True)
    ups = [parse_a3m(s["protein"]["unpairedMsa"], s["protein"]["sequence"]) for s in doc["sequences"]]
    for ci in (0, 1):
        assert [r.seq for r in ups[ci][1:3]] == [r.seq for r in pr.rows[ci]]
    assert [r.seq for r in ups[2]] == [r.seq for r in c.rows]


def test_openfold3_writes_gap_rows_for_the_third_chain(trio, tmp_path):
    a, b, c = trio
    q = OpenFold3.prepare(list(trio), out_dir=tmp_path, quiet=True)
    chains = q["queries"]["query"]["chains"]
    # Read the records as OpenFold3's parse_a3m does: every record is a row.
    paired = [open(e["paired_msa_file_paths"] + "/colabfold_paired.a3m").read().splitlines()[1::2] for e in chains]
    assert [len(p) for p in paired] == [3, 3, 3]
    assert [p[0] for p in paired] == [a.query, b.query, c.query]
    assert paired[2][1:] == ["-" * len(c.query)] * 2
    assert all(any(x.isalpha() for x in s) for p in paired[:2] for s in p[1:])


@pytest.mark.parametrize("adapter", [AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix])
def test_every_adapter_names_the_unpaired_chain(adapter, trio, tmp_path, capsys):
    if adapter is Chai:
        pytest.importorskip("pyarrow")
    a, b, c = trio
    # Chains B and D share a sequence; the species-less chain is C.
    kw = {} if adapter is AlphaFold3 else {"out_dir": tmp_path}
    adapter.prepare([a, b, c, b], **kw)
    err = capsys.readouterr().err
    line = [l for l in err.splitlines() if "chain C" in l]
    assert len(line) == 1, err
    assert "no rows with species information" in line[0] or "no species" in line[0]
    assert "not paired" in line[0] or "no pairing.a3m" in line[0]


def test_attribution_notes_name_the_unpaired_chain(trio, tmp_path):
    Boltz.prepare(list(trio), out_dir=tmp_path, quiet=True)
    (entry,) = json.loads((tmp_path / "scigantic-msa-attribution.json").read_text())["outputs"]
    notes = entry["notes"]
    assert any("chain C" in n and "not paired" in n for n in notes)
