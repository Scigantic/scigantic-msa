import json
import re
from pathlib import Path

import pytest

from conftest import Q_1HVW, Q_1PXQ, Q_6T33, Q_7M22A, Q_7M22C, Q_LITEFOLD, real_alignment
from scigantic_msa import MSA
from scigantic_msa.a3m import Row, parse_a3m, parse_colabfold_complex
from scigantic_msa.adapters import (
    PROTENIX_UNIREF, AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, PrepareError, Protenix,
    protenix_species,
)
from scigantic_msa.alignment import Alignment
from scigantic_msa.pairing import pair_by_species


def synthetic(query, hits, taxids=None, headers=None):
    rows = [Row("101", query, "query")]
    for i, s in enumerate(hits):
        h = (headers or {}).get(i, f"UniRef100_H{i}")
        rows.append(Row(h, s, "uniref", (taxids or {}).get(i)))
    return Alignment(query=query, rows=rows, source="remote", provenance={"backend": "colabfold"})


@pytest.fixture
def dimer():
    return real_alignment(Q_7M22A, with_uniprot=True), real_alignment(Q_7M22C, with_uniprot=True)


@pytest.fixture
def taxid_dimer():
    """Two chains shaped like ColabFold server results: UniRef100 headers,
    taxids from uniref_tax.tsv, no UniProt rows."""
    a = synthetic("MKVLAG", ["MRVLAG", "MKILAG", "MKVAAG", "MKVLGG"], {0: "9606", 1: "10090", 2: "7227"})
    b = synthetic("GHSEQK", ["GHTEQK", "GASEQK", "GHSEAK"], {0: "10090", 1: "9606"})
    return a, b


# --------------------------------------------------------------------------- shared rules


@pytest.mark.parametrize("adapter", [AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix])
def test_sequence_argument_must_match(adapter, tmp_path):
    if adapter is Chai:
        pytest.importorskip("pyarrow")
    m = real_alignment(Q_6T33)
    kw = {} if adapter is AlphaFold3 else {"out_dir": tmp_path / "o"}
    adapter.prepare(m, Q_6T33.lower(), **kw)
    with pytest.raises(PrepareError, match="does not match"):
        adapter.prepare(m, Q_6T33[:-1] + "W", **kw)
    with pytest.raises(PrepareError, match="2 sequence"):
        adapter.prepare(m, [Q_6T33, Q_6T33], **kw)
    with pytest.raises(TypeError, match="one per chain"):
        adapter.prepare(Q_6T33, **kw)
    with pytest.raises(TypeError):
        adapter.prepare([], **kw)


def test_an_edited_result_is_checked_again(tmp_path):
    m = real_alignment(Q_6T33)
    m.rows.append(Row("bad", "MKV"))
    with pytest.raises(PrepareError, match="match columns"):
        Boltz.prepare(m, out_dir=tmp_path)


# --------------------------------------------------------------------------- AlphaFold 3


def test_af3_monomer_is_json_ready():
    m = real_alignment(Q_6T33)
    doc = AlphaFold3.prepare(m, name="job")
    json.dumps(doc)
    assert doc["dialect"] == "alphafold3" and doc["name"] == "job"
    prot = doc["sequences"][0]["protein"]
    assert prot["sequence"] == Q_6T33 and prot["pairedMsa"] == "" and prot["templates"] == []
    assert parse_a3m(prot["unpairedMsa"], Q_6T33)[0].seq == Q_6T33


def test_af3_complex_from_server_style_rows_says_how_to_run(taxid_dimer, tmp_path, capsys):
    a, b = taxid_dimer
    doc = AlphaFold3.prepare([a, b, b], out_dir=tmp_path, name="cx")
    assert (tmp_path / "cx.json").is_file() and json.loads((tmp_path / "cx.json").read_text()) == doc
    prots = [s["protein"] for s in doc["sequences"]]
    assert prots[1]["id"] == ["B", "C"]
    ua, ub = (parse_a3m(p["unpairedMsa"]) for p in prots)
    ta, tb = ({r.header: r.taxid for r in m.rows} for m in (a, b))
    assert [ta[r.header] for r in ua[1:3]] == [tb[r.header] for r in ub[1:3]] == ["10090", "9606"]
    err = capsys.readouterr().err
    assert "paired 2 rows by species" in err and "--resolve_msa_overlaps=false" in err


def test_af3_complex_with_uniprot_rows_uses_paired_msa(dimer, capsys):
    doc = AlphaFold3.prepare(list(dimer))
    for s, m in zip(doc["sequences"], dimer):
        assert parse_a3m(s["protein"]["pairedMsa"], m.query)[1].header.startswith(("tr|", "sp|"))
    assert "AlphaFold 3 pairs them by species" in capsys.readouterr().err


# --------------------------------------------------------------------------- Boltz


def test_boltz_monomer_yaml_and_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = real_alignment(Q_LITEFOLD)
    y = Boltz.prepare(m, out_dir="boltz/")
    f = tmp_path / "boltz" / f"{m.seq_sha256}.a3m"
    assert f.read_text() == m.a3m()
    assert y.startswith("version: 1\nsequences:\n  - protein:\n      id: A\n")
    assert f"      sequence: {Q_LITEFOLD}\n" in y and f'      msa: "{f}"\n' in y


def test_boltz_complex_writes_csv_with_pair_keys(dimer, tmp_path, capsys):
    a, b = dimer
    y = Boltz.prepare([a, b, b], out_dir=tmp_path)
    assert "id: [B, C]" in y
    n = pair_by_species([a, b]).n_paired
    keys = []
    for m in (a, b):
        lines = (tmp_path / f"{m.seq_sha256}.csv").read_text().splitlines()
        assert lines[0] == "key,sequence" and lines[1] == f"-1,{m.query}"
        keys.append({l.split(",")[0] for l in lines[2:]} - {"-1"})
    assert len(keys[0] & keys[1]) == n == 8
    assert "paired 8 rows by species" in capsys.readouterr().err


def test_boltz_says_when_chains_cannot_be_paired(tmp_path, capsys):
    a = synthetic("MKVLAG", ["MRVLAG"], headers={0: "SRR1_2"})
    b = synthetic("GHSEQK", ["GHTEQK"], {0: "9606"})
    Boltz.prepare([a, b], out_dir=tmp_path)
    assert "no pairing: chain A has no rows with species information" in capsys.readouterr().err


# --------------------------------------------------------------------------- Chai-1


def test_chai_files_are_named_for_lookup(dimer, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    a, b = dimer
    d = Chai.prepare([a, b], out_dir=tmp_path / "chai")
    assert d == (tmp_path / "chai").resolve()
    import hashlib

    for m in (a, b):
        p = d / (hashlib.sha256(m.query.encode()).hexdigest() + ".aligned.pqt")
        t = pq.read_table(p).to_pydict()
        assert t["sequence"][0] == m.query and t["source_database"][0] == "query"
    assert Chai.fasta([a, b]) == f">protein|name=A\n{Q_7M22A}\n>protein|name=B\n{Q_7M22C}\n"


# --------------------------------------------------------------------------- ColabFold


def test_colabfold_monomer_and_complex(dimer, tmp_path):
    m = real_alignment(Q_6T33)
    p = ColabFold.prepare(m, out_dir=tmp_path, name="mono")
    assert p == (tmp_path / "mono.a3m").resolve() and p.read_text() == m.a3m()
    p2 = ColabFold.prepare(list(dimer), out_dir=tmp_path, name="cx", quiet=True)
    lens, cards, queries, _ = parse_colabfold_complex(p2.read_text())
    assert queries == [Q_7M22A, Q_7M22C] and cards == [1, 1]


# --------------------------------------------------------------------------- OpenFold3


def test_openfold3_query_json(taxid_dimer, tmp_path):
    a, b = taxid_dimer
    q = OpenFold3.prepare([a, b, a], out_dir=tmp_path, name="cx", quiet=True)
    chains = q["queries"]["cx"]["chains"]
    assert [c["chain_ids"] for c in chains] == [["A", "C"], ["B"]]
    for c, m in zip(chains, (a, b)):
        assert c["molecule_type"] == "protein" and c["sequence"] == m.query
        main = Path(c["main_msa_file_paths"])
        assert main.is_absolute() and (main / "colabfold_main.a3m").is_file()
        paired = parse_a3m((Path(c["paired_msa_file_paths"]) / "colabfold_paired.a3m").read_text(), m.query)
        assert len(paired) == 3
    json.dumps(q)


# --------------------------------------------------------------------------- Protenix


def test_protenix_monomer_layout(tmp_path):
    m = real_alignment(Q_6T33)
    jobs = Protenix.prepare(m, out_dir=tmp_path, name="mono")
    assert isinstance(jobs, list) and jobs[0]["name"] == "mono"
    pc = jobs[0]["sequences"][0]["proteinChain"]
    assert pc["sequence"] == Q_6T33 and pc["count"] == 1
    up = Path(pc["unpairedMsaPath"])
    assert up.is_absolute() and up.name == "non_pairing.a3m" and up.parent.name == m.seq_sha256
    assert up.read_text() == m.a3m()
    # Protenix pairs only complexes of different chains; a monomer gets no paired file.
    assert "pairedMsaPath" not in pc
    json.dumps(jobs)


def test_protenix_complex_headers_carry_the_species_protenix_reads(taxid_dimer, tmp_path):
    a, b = taxid_dimer
    jobs = Protenix.prepare([a, b, b], out_dir=tmp_path, quiet=True)
    pcs = [s["proteinChain"] for s in jobs[0]["sequences"]]
    assert [pc["count"] for pc in pcs] == [1, 2]
    species = []
    for pc, m in zip(pcs, (a, b)):
        p = Path(pc["pairedMsaPath"])
        assert p.name == "pairing.a3m" and p.parent == Path(pc["unpairedMsaPath"]).parent
        rows = parse_a3m(p.read_text(), m.query)
        assert rows[0].header == "query"
        ids = [protenix_species(r.header) for r in rows[1:]]
        assert all(ids)
        species.append(set(ids))
    assert species[0] & species[1] == {"10090", "9606"}
    assert PROTENIX_UNIREF.match("UniRef100_H0_9606/").group(1) == "9606"


def test_protenix_uniprot_rows_with_ox_get_the_taxid(dimer, tmp_path):
    # The real UniProt hits carry OX=<taxid>; the library pairs on that, so
    # Protenix must read the same taxid, not the mnemonic.
    jobs = Protenix.prepare(list(dimer), out_dir=tmp_path, quiet=True)
    for s, m in zip(jobs[0]["sequences"], dimer):
        rows = parse_a3m(Path(s["proteinChain"]["pairedMsaPath"]).read_text(), m.query)
        assert len(rows) > 1
        ox = {r.header.split("|")[1]: re.search(r"OX=(\d+)", r.header).group(1) for r in m.uniprot_rows[1:]}
        for r in rows[1:]:
            acc = r.header[len("UniRef100_"):].split("_")[0]
            assert PROTENIX_UNIREF.match(r.header) and protenix_species(r.header) == ox[acc]
        assert rows[1].header.startswith("UniRef100_") and "|" not in rows[1].header


def test_protenix_keeps_uniprot_headers_without_a_taxid(tmp_path):
    a = synthetic("MKVLAG", ["MRVLAG"], headers={0: "tr|Q9XYZ1|Q9XYZ1_FUNHE/3-8"})
    b = synthetic("GHSEQK", ["GHTEQK"], headers={0: "sp|Q8ABC1|ABC_FUNHE"})
    jobs = Protenix.prepare([a, b], out_dir=tmp_path, quiet=True)
    ra, rb = _paired_species(jobs, (a, b))
    assert [r.header for r in ra] == ["tr|Q9XYZ1|Q9XYZ1_FUNHE/3-8"]
    assert [r.header for r in rb] == ["sp|Q8ABC1|ABC_FUNHE"]


def test_protenix_no_pairing_file_when_chains_share_no_species(tmp_path, capsys):
    a, b = real_alignment(Q_1HVW, with_uniprot=True), real_alignment(Q_1PXQ, with_uniprot=True)
    jobs = Protenix.prepare([a, b], out_dir=tmp_path)
    assert all("pairedMsaPath" not in s["proteinChain"] for s in jobs[0]["sequences"])
    assert "no two chains share a species" in capsys.readouterr().err


# The Protenix checks. A past Protenix run silently ignored a paired MSA file
# that had the wrong name, so a wrong name, an empty file or a missing file
# must each be an error here.


def _job(pc):
    return [{"name": "x", "sequences": [{"proteinChain": dict({"sequence": Q_6T33, "count": 1}, **pc)}]}]


def _good_dir(tmp_path):
    d = tmp_path / "msa"
    d.mkdir()
    (d / "non_pairing.a3m").write_text(real_alignment(Q_6T33).a3m())
    (d / "pairing.a3m").write_text(f">query\n{Q_6T33}\n")
    return d


def test_protenix_check_accepts_what_prepare_writes(tmp_path):
    jobs = Protenix.prepare(real_alignment(Q_6T33), out_dir=tmp_path / "o")
    Protenix.check(jobs)
    p = tmp_path / "in.json"
    p.write_text(json.dumps(jobs))
    Protenix.check(p)
    d = _good_dir(tmp_path)
    Protenix.check(_job({"msa": {"precomputed_msa_dir": str(d), "pairing_db": "uniref100"}}))


def test_protenix_misnamed_paired_file_in_msa_dir_is_an_error(tmp_path):
    d = _good_dir(tmp_path)
    (d / "pairing.a3m").rename(d / "paired.a3m")
    with pytest.raises(PrepareError, match="reads only pairing.a3m and non_pairing.a3m.*ignore paired.a3m"):
        Protenix.check(_job({"msa": {"precomputed_msa_dir": str(d)}}))


def test_protenix_dir_with_no_recognised_file_is_an_error(tmp_path):
    d = tmp_path / "msa"
    d.mkdir()
    (d / "unpaired.a3m").write_text(f">q\n{Q_6T33}\n")
    with pytest.raises(PrepareError, match="neither pairing.a3m nor non_pairing.a3m"):
        Protenix.check(_job({"msa": {"precomputed_msa_dir": str(d)}}))


def test_protenix_swapped_file_names_are_an_error(tmp_path):
    d = _good_dir(tmp_path)
    with pytest.raises(PrepareError, match="swapped"):
        Protenix.check(_job({"pairedMsaPath": str(d / "non_pairing.a3m"), "unpairedMsaPath": str(d / "pairing.a3m")}))


@pytest.mark.parametrize("content", ["", "\n\n"])
def test_protenix_empty_file_is_an_error(tmp_path, content):
    d = _good_dir(tmp_path)
    (d / "pairing.a3m").write_text(content)
    with pytest.raises(PrepareError, match="pairing.a3m is empty"):
        Protenix.check(_job({"pairedMsaPath": str(d / "pairing.a3m"), "unpairedMsaPath": str(d / "non_pairing.a3m")}))
    with pytest.raises(PrepareError, match="is empty"):
        Protenix.check(_job({"msa": {"precomputed_msa_dir": str(d)}}))


def test_protenix_missing_file_is_an_error(tmp_path):
    with pytest.raises(PrepareError, match="does not exist; Protenix would start a new MSA search"):
        Protenix.check(_job({"unpairedMsaPath": str(tmp_path / "nope" / "non_pairing.a3m")}))


def test_protenix_wrong_first_row_is_an_error(tmp_path):
    d = _good_dir(tmp_path)
    (d / "pairing.a3m").write_text(f">query\n{Q_6T33[:-1]}W\n")
    with pytest.raises(PrepareError, match="does not match the query"):
        Protenix.check(_job({"pairedMsaPath": str(d / "pairing.a3m")}))


def test_protenix_chain_with_no_msa_is_an_error():
    with pytest.raises(PrepareError, match="run its own search"):
        Protenix.check(_job({}))


def test_protenix_species_follows_protenix():
    assert protenix_species("tr|A0A146SKV9|A0A146SKV9_FUNHE/3-80") == "FUNHE"
    assert protenix_species("UniRef100_A0A2N5_10090/") == "10090"
    assert protenix_species("UniRef100_A0A2N5\t50") == ""
    assert protenix_species("SRR123_4") == ""


# --------------------------------------------------------------------------- the public API shape


def test_public_api_shape(tmp_path):
    p = tmp_path / "x.a3m"
    p.write_text(real_alignment(Q_6T33).a3m())
    result = MSA().from_a3m(p)
    af3_input = AlphaFold3.prepare(result)
    assert json.loads(json.dumps(af3_input)) == af3_input
    boltz_yaml = Boltz.prepare(result, out_dir=tmp_path / "boltz")
    assert boltz_yaml.startswith("version: 1")
    assert isinstance(ColabFold.prepare(result, out_dir=tmp_path / "cf"), Path)
    assert "queries" in OpenFold3.prepare(result, out_dir=tmp_path / "of3")
    assert Protenix.prepare(result, out_dir=tmp_path / "px")[0]["sequences"]


def test_protenix_leaves_out_rows_whose_species_it_cannot_read(tmp_path):
    # Our pairing reads a lowercase UniProt accession; Protenix's pattern does not.
    a = synthetic("MKVLAG", ["MRVLAG", "MKILAG"], headers={0: "tr|a0a146skv9|A0A146SKV9_FUNHE", 1: "tr|Q9XYZ1|Q9XYZ1_FUNHE"})
    b = synthetic("GHSEQK", ["GHTEQK"], headers={0: "tr|Q8ABC1|Q8ABC1_FUNHE"})
    jobs = Protenix.prepare([a, b], out_dir=tmp_path, quiet=True)
    rows = parse_a3m(Path(jobs[0]["sequences"][0]["proteinChain"]["pairedMsaPath"]).read_text(), a.query)
    assert [r.header for r in rows[1:]] == ["tr|Q9XYZ1|Q9XYZ1_FUNHE"]


def test_protenix_prepare_checks_its_own_output(tmp_path, monkeypatch):
    m = real_alignment(Q_6T33)
    monkeypatch.setattr(Alignment, "a3m", lambda self, max_rows=None: "")
    with pytest.raises(PrepareError, match="non_pairing.a3m is empty"):
        Protenix.prepare(m, out_dir=tmp_path)


# --------------------------------------------------------------------------- Protenix pairs on the library's taxid


def _paired_species(jobs, chains):
    out = []
    for s, m in zip(jobs[0]["sequences"], chains):
        rows = parse_a3m(Path(s["proteinChain"]["pairedMsaPath"]).read_text(), m.query)
        out.append(rows[1:])
    return out


def test_protenix_pairing_headers_carry_the_taxid(tmp_path):
    # A full UniRef header ends in RepID=<acc>_<code>; Protenix would read the
    # code (9BETA, shared by many species) instead of the TaxID.
    qa, qb = "MKVLAAGIVA", "GHSTWQERLL"
    a = MSA().from_a3m(f">q\n{qa}\n"
                       f">UniRef100_P1 x n=1 Tax=Sp one TaxID=111 RepID=P1_9BETA\nMKVLAAGIVG\n"
                       f">UniRef100_P2 x n=1 Tax=Sp two TaxID=222 RepID=P2_9BETA\nMKVLSAGIVA\n")
    b = MSA().from_a3m(f">q\n{qb}\n"
                       f">UniRef100_P3 x n=1 Tax=Sp two TaxID=222 RepID=P3_9BETA\nGHSTWQERLI\n"
                       f">UniRef100_P4 x n=1 Tax=Sp three TaxID=333 RepID=P4_9BETA\nGHSTWAERLL\n")
    jobs = Protenix.prepare([a, b], out_dir=tmp_path, quiet=True)
    ra, rb = _paired_species(jobs, (a, b))
    assert [protenix_species(r.header) for r in ra] == ["111", "222"]
    assert [protenix_species(r.header) for r in rb] == ["222", "333"]
    assert [r.header for r in ra] == ["UniRef100_P1_111/", "UniRef100_P2_222/"]


def test_protenix_species_equals_the_taxid_the_library_paired_on(tmp_path):
    # Every kind of header a row with a taxid can carry: a ColabFold accession
    # with a taxid from uniref_tax.tsv, a full UniRef header, a UniProt header
    # with OX= whose mnemonic is not the species, and accessions with _ or /.
    from scigantic_msa.alignment import species_key

    a = synthetic("MKVLAG", ["MRVLAG", "MKILAG", "MKVAAG", "MKVLGG", "MKVLAW"],
                  taxids={0: "9606"},
                  headers={1: "UniRef100_A0A_B1 desc n=1 TaxID=10090 RepID=A0A_B1_9MURI",
                           2: "tr|Q9XYZ1|Q9XYZ1_9BETA desc OX=7227 GN=x",
                           3: "UniRef100_P5/2-40 n=1 TaxID=4932",
                           4: "tr|Q8ABC9|Q8ABC9_HUMAN no taxid here"})
    b = synthetic("GHSEQK", ["GHTEQK", "GASEQK", "GHSEAK", "GHSEQW", "GHSWQK"],
                  taxids={1: "9606"},
                  headers={0: "sp|P12345|ABC_9BETA OX=10090",
                           2: "UniRef100_X9 n=1 TaxID=7227 RepID=X9_9BETA",
                           3: "UniRef100_Y1/3-9 n=1 TaxID=4932",
                           4: "tr|Q1ABC2|Q1ABC2_HUMAN"})
    pr = pair_by_species([a, b])
    assert pr.n_paired == 5
    jobs = Protenix.prepare([a, b], out_dir=tmp_path, quiet=True)
    for rows, m in zip(_paired_species(jobs, (a, b)), (a, b)):
        by_seq = {r.seq: r for r in m.rows}
        for r in rows:
            key = species_key(by_seq[r.seq])
            assert protenix_species(r.header) == key[3:], (r.header, key)
    # The pairing Protenix will do matches the library's: the same species in both chains.
    ra, rb = _paired_species(jobs, (a, b))
    assert {protenix_species(r.header) for r in ra} == {protenix_species(r.header) for r in rb}


# --------------------------------------------------------------------------- AF3 merge


def test_af3_merge_lowercase_sequence_matches_the_msa_query():
    m = real_alignment(Q_6T33)
    inp = {"dialect": "alphafold3", "version": 2, "name": "x", "modelSeeds": [1],
           "sequences": [{"protein": {"id": "A", "sequence": Q_6T33.lower()}}]}
    doc = AlphaFold3.prepare(m, input=inp)
    p = doc["sequences"][0]["protein"]
    assert p["sequence"] == Q_6T33
    assert p["sequence"] == p["unpairedMsa"].split("\n")[1]
    assert inp["sequences"][0]["protein"]["sequence"] == Q_6T33.lower()  # the caller's dict is not changed


@pytest.mark.parametrize("ent, msg", [
    ("proteinChain", "must be an object.*not str"),
    ([{"proteinChain": {}}], "must be an object.*not list"),
    ({"proteinChain": "MKV"}, "proteinChain must be an object, not str"),
])
def test_protenix_check_rejects_a_sequences_entry_that_is_not_an_object(ent, msg):
    with pytest.raises(PrepareError, match=msg):
        Protenix.check([{"name": "x", "sequences": [ent]}])


def test_protenix_check_skips_entries_that_are_not_protein(tmp_path):
    jobs = Protenix.prepare(real_alignment(Q_6T33), out_dir=tmp_path)
    jobs[0]["sequences"].append({"ligand": {"ligand": "CCD_ATP", "count": 1}})
    Protenix.check(jobs)
