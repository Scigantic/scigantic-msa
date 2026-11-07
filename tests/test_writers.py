import csv
import json

import pytest

from conftest import Q_1HVW, Q_1PXQ, Q_6T33, Q_7M22A, Q_7M22C, Q_LITEFOLD, REAL, key
from scigantic_msa.a3m import Row, match_length, parse_a3m, parse_colabfold_complex
from scigantic_msa.assemble import load_entry
from scigantic_msa.alignment import Alignment
from scigantic_msa.pairing import pair_by_species
from scigantic_msa.writers import (
    ATTRIBUTION_FILE, chai_expected_basename, write_af3_json, write_boltz, write_chai, write_colabfold, write_openfold3,
)


@pytest.fixture
def msas(index_setup):
    idx, _, _ = index_setup

    def get(seq, **kw):
        return load_entry(idx, idx.lookup(key(seq))[0], seq, **kw)

    return get


def synthetic(query, hits, taxids=None):
    rows = [Row("q", query, "query")] + [Row(f"h{i} x", s, "uniref", (taxids or {}).get(i)) for i, s in enumerate(hits)]
    return Alignment(query=query, rows=rows, source="remote", provenance={"backend": "colabfold"})


def test_a3m_and_sidecar(msas, tmp_path):
    m = msas(Q_6T33)
    p = tmp_path / "out" / "x.a3m"
    text = m.to_a3m(p)
    assert p.read_text() == text
    assert parse_a3m(text, Q_6T33)[0].seq == Q_6T33
    side = json.loads((p.parent / ATTRIBUTION_FILE).read_text())
    assert side["chains"][0]["attribution"][0]["license"] == "CC-BY-4.0"
    assert side["chains"][0]["source"] == "exact"


def test_af3_new_input_monomer(msas, tmp_path):
    m = msas(Q_6T33, purpose="af3")
    doc = m.to_af3_json(tmp_path / "in.json")
    assert doc["dialect"] == "alphafold3" and doc["version"] == 2
    prot = doc["sequences"][0]["protein"]
    assert set(prot) <= {"id", "sequence", "unpairedMsa", "pairedMsa", "templates"}
    assert prot["pairedMsa"] == "" and prot["templates"] == []
    rows = parse_a3m(prot["unpairedMsa"], Q_6T33)
    assert rows[0].seq == Q_6T33
    assert json.loads((tmp_path / "in.json").read_text()) == doc


def test_af3_merge_into_existing_litefold_json(msas, tmp_path):
    existing = json.loads((REAL / "litefold-A0A096LP01/A0A096LP01.json").read_text())
    existing["sequences"][0]["protein"]["unpairedMsa"] = None
    existing["sequences"][0]["protein"]["pairedMsa"] = None
    existing["sequences"].append({"ligand": {"id": "L", "ccdCodes": ["ATP"]}})
    m = msas(Q_LITEFOLD)
    doc = write_af3_json([m], tmp_path / "merged.json", input=existing)
    prot = doc["sequences"][0]["protein"]
    assert prot["unpairedMsa"].startswith(">101\n")
    assert prot["pairedMsa"] == ""
    assert doc["sequences"][1] == {"ligand": {"id": "L", "ccdCodes": ["ATP"]}}
    assert doc["name"] == existing["name"]


def test_af3_rejects_unknown_keys_and_unmatched_chains(msas):
    base = {"dialect": "alphafold3", "version": 2, "name": "x", "modelSeeds": [1],
            "sequences": [{"protein": {"id": "A", "sequence": Q_6T33, "msa": "nope"}}]}
    with pytest.raises(ValueError, match="does not accept"):
        write_af3_json([msas(Q_6T33)], input=base)
    base["sequences"][0]["protein"].pop("msa")
    with pytest.raises(ValueError, match="match no protein"):
        write_af3_json([msas(Q_1HVW)], input=base)
    with pytest.raises(ValueError, match="dialect"):
        write_af3_json([msas(Q_6T33)], input={"sequences": []})


def test_af3_path_mode(msas, tmp_path):
    m = msas(Q_6T33)
    doc = m.to_af3_json(tmp_path / "in.json", inline=False)
    prot = doc["sequences"][0]["protein"]
    assert "unpairedMsa" not in prot and prot["unpairedMsaPath"] == f"{m.seq_sha256}.unpaired.a3m"
    assert (tmp_path / prot["unpairedMsaPath"]).exists() and prot["pairedMsa"] == ""


def test_af3_complex_with_uniprot_rows_uses_pairedMsa(msas, tmp_path):
    a, b = msas(Q_7M22A, want_uniprot=True), msas(Q_7M22C, want_uniprot=True)
    doc = write_af3_json([a, b, b], tmp_path / "c.json")
    prots = [s["protein"] for s in doc["sequences"]]
    assert prots[0]["id"] == "A" and prots[1]["id"] == ["B", "C"]
    for p, m in zip(prots, (a, b)):
        paired = parse_a3m(p["pairedMsa"], m.query)
        assert len(paired) == len(m.uniprot_rows)
        assert paired[1].header.startswith(("tr|", "sp|"))


def test_af3_complex_prepaired_by_taxid(tmp_path):
    a = synthetic("MKVL", ["MRVL", "MKIL", "MKVA"], {0: "9606", 1: "10090", 2: "7227"})
    b = synthetic("GHSE", ["GHTE", "GASE"], {0: "10090", 1: "9606"})
    doc = write_af3_json([a, b], tmp_path / "c.json")
    ua = parse_a3m(doc["sequences"][0]["protein"]["unpairedMsa"])
    ub = parse_a3m(doc["sequences"][1]["protein"]["unpairedMsa"])
    # Rows 1 and 2 are paired by taxid in both chains.
    assert [r.taxid for r in ua[1:3]] == [r.taxid for r in ub[1:3]]
    side = json.loads((tmp_path / ATTRIBUTION_FILE).read_text())
    assert any("--resolve_msa_overlaps=false" in n for n in side["notes"])


def test_boltz_monomer_and_complex(msas, tmp_path):
    m = msas(Q_6T33)
    line = m.to_boltz(tmp_path / "b1")
    p = tmp_path / "b1" / f"{m.seq_sha256}.a3m"
    assert line == f"msa: {p}" and p.read_text() == m.a3m()
    a, b = msas(Q_7M22A, want_uniprot=True), msas(Q_7M22C, want_uniprot=True)
    lines = write_boltz([a, b], tmp_path / "b2")
    assert all(l.endswith(".csv") for l in lines)
    pr = pair_by_species([a, b])
    assert pr.n_paired > 0
    for m2, l in zip((a, b), lines):
        rows = list(csv.DictReader(open(l.split("msa: ", 1)[1])))
        assert list(rows[0]) == ["key", "sequence"]
        assert rows[0] == {"key": "-1", "sequence": m2.query}
        keys = [r["key"] for r in rows[1:]]
        assert keys[: pr.n_paired] == [str(i) for i in range(1, pr.n_paired + 1)]
        assert all(k == "-1" for k in keys[pr.n_paired:])
        assert all(match_length(r["sequence"]) == len(m2.query) for r in rows)


def test_chai_parquet_schema(msas, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    m = msas(Q_LITEFOLD)
    p = m.to_chai(tmp_path / "chai")
    assert p.name == chai_expected_basename(Q_LITEFOLD) == f"{m.seq_sha256}.aligned.pqt"
    t = pq.read_table(p)
    assert t.column_names == ["sequence", "source_database", "pairing_key", "comment"]
    assert all(str(f.type) == "string" for f in t.schema)
    d = t.to_pydict()
    assert d["source_database"][0] == "query" and d["sequence"][0] == Q_LITEFOLD
    assert d["source_database"].count("query") == 1
    assert set(d["source_database"]) <= {"query", "uniprot", "uniref90", "bfd_uniclust", "mgnify"}
    assert "uniref90" in d["source_database"] and "bfd_uniclust" in d["source_database"]
    assert set(d["pairing_key"]) == {""}


def test_chai_complex_pairing_keys(msas, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    a, b = msas(Q_7M22A, want_uniprot=True), msas(Q_7M22C, want_uniprot=True)
    pa_, pb_ = write_chai([a, b], tmp_path / "c")
    ka = pq.read_table(pa_).to_pydict()["pairing_key"]
    kb = pq.read_table(pb_).to_pydict()["pairing_key"]
    shared = (set(ka) & set(kb)) - {""}
    assert shared and len(shared) == pair_by_species([a, b]).n_paired


def test_colabfold_monomer_homomer_and_complex(msas, tmp_path):
    m = msas(Q_6T33)
    p = write_colabfold([m], tmp_path / "m.a3m")
    assert p.read_text() == m.a3m()
    p2 = write_colabfold([m, m], tmp_path / "h.a3m")
    assert p2.read_text().startswith(f"#{len(Q_6T33)}\t2\n")
    a, b = msas(Q_7M22A, want_uniprot=True), msas(Q_7M22C, want_uniprot=True)
    p3 = write_colabfold([a, b], tmp_path / "c.a3m")
    lens, cards, queries, records = parse_colabfold_complex(p3.read_text())
    assert lens == [len(Q_7M22A), len(Q_7M22C)] and cards == [1, 1] and queries == [Q_7M22A, Q_7M22C]
    n_pair = pair_by_species([a, b]).n_paired
    assert len(records) == 1 + n_pair + a.depth + b.depth


def test_openfold3_dirs(msas, tmp_path):
    a, b = msas(Q_7M22A, want_uniprot=True), msas(Q_7M22C, want_uniprot=True)
    fields = write_openfold3([a, b], tmp_path / "of3")
    for f, m in zip(fields, (a, b)):
        d = tmp_path / "of3" / "main" / m.seq_sha256
        assert f["main_msa_file_paths"] == str(d)
        assert (d / "colabfold_main.a3m").read_text() == m.a3m()
        assert parse_a3m((d / "uniprot_hits.a3m").read_text(), m.query)[1].header.startswith(("tr|", "sp|"))
        assert "paired_msa_file_paths" not in f
    x = synthetic("MKVL", ["MRVL"], {0: "9606"})
    y = synthetic("GHSE", ["GHTE"], {0: "9606"})
    f2 = write_openfold3([x, y], tmp_path / "of3b")
    for f, m in zip(f2, (x, y)):
        pd = f["paired_msa_file_paths"]
        rows = parse_a3m(open(pd + "/colabfold_paired.a3m").read(), m.query)
        assert len(rows) == 2


def test_pairing_says_when_it_cannot(tmp_path):
    a = synthetic("MKVL", ["MRVL"])
    b = synthetic("GHSE", ["GHTE"], {0: "9606"})
    pr = pair_by_species([a, b])
    assert pr.n_paired == 0 and pr.chains_without_species == [0]
    assert "no rows with species information" in pr.message


def test_real_unrelated_chains_share_no_species(msas):
    a, b = msas(Q_1HVW, want_uniprot=True), msas(Q_1PXQ, want_uniprot=True)
    pr = pair_by_species([a, b])
    assert pr.n_paired == 0 and "no species is shared" in pr.message


def test_real_heterodimer_pairs_by_species(msas):
    a, b = msas(Q_7M22A, want_uniprot=True), msas(Q_7M22C, want_uniprot=True)
    pr = pair_by_species([a, b])
    # 10 species are shared; 2 of those rows are identical to the query and are left out.
    assert pr.n_paired == 8
    from scigantic_msa.alignment import species_key

    for ra, rb in zip(*pr.rows):
        assert species_key(ra) == species_key(rb) is not None
        assert ra.seq != a.query and rb.seq != b.query


def test_uniref90_cluster_taxids_are_not_species():
    from scigantic_msa.alignment import species_key

    assert species_key(Row("UniRef90_X n=9 Tax=Firmicutes TaxID=1239", "M")) is None
    assert species_key(Row("UniRef90_X n=1 Tax=Homo TaxID=9606", "M")) == "tx:9606"
    assert species_key(Row("tr|A0A146SKV9|A0A146SKV9_FUNHE/3-80", "M")) == "mn:FUNHE"
    assert species_key(Row("SRR123_4", "M")) is None
