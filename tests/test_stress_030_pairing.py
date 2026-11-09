"""Defects found by the 0.3.0 stress test: pairing, the merge, writers and
the UniClust30 layout.

Offline: every host is a local fake server, and the real published files
come from fixtures/real.
"""
import json
import re
from types import SimpleNamespace

import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, REAL, md5
from scigantic_msa import MSA
from scigantic_msa.a3m import Row
from scigantic_msa.adapters import Boltz, Chai, ColabFold
from scigantic_msa.alignment import Alignment, species_key
from scigantic_msa.opendata.lookup import OpenResult, Prov
from scigantic_msa.opendata.merge import combine_pairs, merge_rows
from scigantic_msa.opendata.parse import SourceMismatch, parse_a3m_lines
from scigantic_msa.pairing import pair_by_species
from scigantic_msa.writers import ATTRIBUTION_FILE

# M4VQ73 in UniProt today. Its UniClust30 file starts with a consensus row
# that differs from it at two positions and has no representative row.
Q_M4VQ73 = ("KYPAIKDLKKPCITLGKAPDLNKAYKSVLSGMNAAKLDPDDVCSYLAAAMQFFEGTCPEDWTSYGILIARKGAKITPDSLVEIKRTDVEGNWALTGGMEL"
            "TRDPTVSEHASLVGLLLSLYRLSKISGQNTGNYKTNIADRIEQIFETAPFVKIVEHHTLMTTHKMCANWSTIPNFRFLAGTYDMFFSRIEHLYSAIRVG"
            "TVVTA")


def _lines(data: bytes):
    return [line + b"\n" for line in data.split(b"\n")]


# --------------------------------------------------------------------------- 1: species-level keys only


NOT_SPECIES = [
    "tr|K1M0V5|K1M0V5_9FLAO Barstar",                     # UniProt virtual code
    "tr|A0A1M6GYI1|A0A1M6GYI1_9FLAO Barstar",
    "UniRef100_A0A961A929 Protein n=3 Tax=Bacteria TaxID=2 RepID=A0A961A929_9BACT",
    "UniRef100_A0A961A929 Protein n=1 Tax=Bacteria TaxID=2 RepID=A0A961A929_9BACT",
    "UniRef100_K2CAX1 Protein n=1 Tax=unclassified Deltaproteobacteria TaxID=122706 RepID=K2CAX1_9BACT",
    "UniRef100_X n=2 Tax=Homo sapiens TaxID=9606 RepID=X_HUMAN",   # a cluster of two
    "UniRef50_X n=4 Tax=Homo sapiens TaxID=9606 RepID=X_HUMAN",
]


def test_only_species_level_keys_pair():
    got = {h: species_key(Row(h, "A", "uniprot")) for h in NOT_SPECIES}
    assert got == {h: None for h in NOT_SPECIES}
    # Species still pair.
    assert species_key(Row("tr|A1|A1_9GAMM P OS=Acinetobacter sp. ANC 4945 OX=1960941", "A")) == "tx:1960941"
    assert species_key(Row("UniRef100_X n=1 Tax=Homo sapiens TaxID=9606 RepID=X_HUMAN", "A")) == "tx:9606"
    assert species_key(Row("tr|P69905|HBA_HUMAN", "A")) == "mn:HUMAN"


def test_a_server_row_whose_taxid_is_a_domain_is_not_a_species():
    assert species_key(Row("UniRef100_A1\t1\t1", "A", "uniref", "2")) is None
    assert species_key(Row("UniRef100_A1\t1\t1", "A", "uniref", "562")) == "tx:562"


def test_chains_sharing_only_a_virtual_code_get_no_pairs():
    qa, qb = "ACDEFGHIKL", "MNPQRSTVWY"
    a = Alignment(query=qa, rows=[Row("q", qa), Row("tr|K1M0V5|K1M0V5_9FLAO", "ACDEFGHIKA")], source="file")
    b = Alignment(query=qb, rows=[Row("q", qb), Row("tr|A0A1M6GYI1|A0A1M6GYI1_9FLAO", "MNPQRSTVWA")], source="file")
    assert pair_by_species([a, b]).n_paired == 0


# --------------------------------------------------------------------------- 2: server pairs stay


def test_server_pairs_that_repeat_a_row_or_the_query_in_one_chain_are_kept():
    qa, qb = "ACDEFGHIKL", "MNPQRSTVWY"
    sa = [Row("UniRef100_A1", "ACDEFGHIKA", "uniref"), Row("UniRef100_A2", "ACDEFGHIKA", "uniref"),
          Row("UniRef100_A3", qa, "uniref")]
    sb = [Row("UniRef100_B1", "MNPQRSTVWA", "uniref"), Row("UniRef100_B2", "MNPQRSTVWC", "uniref"),
          Row("UniRef100_B3", "MNPQRSTVWD", "uniref")]
    merged = [[Row("q", qa)], [Row("q", qb)]]
    plan = combine_pairs([qa, qb], merged, merged, ["A", "B"], server_pairs=[sa, sb], add_open=False)
    assert plan.server_pairs == 3 and "left out" not in plan.pairing.message


# --------------------------------------------------------------------------- 3: Chai-1 quotas


def test_chai_says_when_its_quota_drops_rows(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    q = "ACDEFGHIKL"
    rows = [Row("q", q, "query")] + [Row(f"UniRef100_X{i}", "ACDEFGHIKA", "uniref") for i in range(10_005)]
    m = Alignment(query=q, rows=rows, source="file")
    Chai.prepare(m, out_dir=tmp_path)
    err = capsys.readouterr().err
    assert "10,005 uniref90 rows" in err and "first 9,999" in err and "drops 6" in err
    notes = json.loads((tmp_path / ATTRIBUTION_FILE).read_text())["outputs"][0]["notes"]
    assert any("drops 6" in n for n in notes)


# --------------------------------------------------------------------------- 4: no inherited species


def _p(src="openfold3", role="uniprot", rec=0):
    return Prov(src, role, "e", "u", rec)


def test_a_metagenomic_row_does_not_take_the_species_of_a_copy_and_inputs_stay_as_they_were():
    q = "ACDEFGHIKLMNPQRSTVWY"
    env = Row("MGYP000123", "ACDEFGHIKLMNPQRSTVWA", "env")
    env.extra["prov"] = _p("colabfold", "env")
    up = Row("tr|A1|A1_HUMAN OX=9606", "ACDEFGHIKLMNPQRSTVWA", "uniprot")
    up.extra["prov"] = _p()
    out, _ = merge_rows(q, [env], [up])
    assert len(out) == 1 and out[0].header == "MGYP000123"
    assert out[0].taxid is None and species_key(out[0]) is None
    assert out[0].extra["also"] == {"openfold3:uniprot"}
    assert env.taxid is None and "also" not in env.extra and out[0] is not env


# --------------------------------------------------------------------------- 5: open pairs are merged rows


def _fake_open(rows_by_query):
    def collect(seq):
        res = OpenResult(query=seq)
        res.organisms["uniparc"] = {"Homo sapiens"}
        res.rows = rows_by_query[seq]
        return res
    return SimpleNamespace(collect=collect)


def _open_row(header, seq, rec):
    r = Row(header, seq, "uniprot")
    r.extra["prov"] = _p(rec=rec)
    return r


def test_open_pairs_use_the_rows_the_merge_kept():
    qa, qb = "ACDEFGHIKLMNPQRSTVWY", "MNPQRSTVWYACDEFGHIKL"
    # The shorter copy of each entry comes first; the merge keeps the longer.
    rows = {qa: [_open_row("tr|A1|A1_HUMAN/2-19 OX=9606", "-CDEFGHIKLMNPQRSTVA-", 1),
                 _open_row("tr|A1|A1_HUMAN/1-20 OX=9606", "ACDEFGHIKLMNPQRSTVAA", 2)],
            qb: [_open_row("tr|B1|B1_HUMAN/2-19 OX=9606", "-NPQRSTVWYACDEFGHIA-", 1),
                 _open_row("tr|B1|B1_HUMAN/1-20 OX=9606", "MNPQRSTVWYACDEFGHIAA", 2)]}
    tool = MSA(server=None, quiet=True)
    tool.open = _fake_open(rows)
    a, b = tool.get([qa, qb])
    cp = a.complex_pairing
    assert cp.open_pairs == 1
    for m, c in ((a, 0), (b, 1)):
        r = cp.pairing.rows[c][0]
        assert any(r is x for x in m.rows), r.header


# --------------------------------------------------------------------------- 6: rows TSV locates OPS rows


def test_rows_table_gives_the_byte_range_of_an_openproteinset_row(hosts, tmp_path):
    ops = REAL / "ops-6t33_A"
    blob, files, pos = b"", [], 100
    for name in ("bfd_uniclust_hits.a3m", "mgnify_hits.a3m", "pdb70_hits.hhr", "uniref90_hits.a3m"):
        data = (ops / name).read_bytes()
        files.append([name, pos, len(data)])
        blob += data
        pos += len(data)
    cache = {"6t33_A": {"seq": Q_6T33}}
    index = {"6t33_A": {"db": "alignment_db_3.db", "files": files}}
    hosts.s.static["/data_caches/chain_data_cache.json"] = json.dumps(cache, indent=4).encode()
    hosts.s.static["/alignment_db/alignment_db.index"] = json.dumps(index, indent=4).encode()
    db = b"x" * 100 + blob
    hosts.s.static["/alignment_db/alignment_db_3.db"] = db
    m = MSA(server=None, quiet=True, openproteinset=True).get(Q_6T33)
    ColabFold.prepare(m, out_dir=tmp_path, quiet=True)
    tsv = next((tmp_path / "scigantic-msa-rows").glob("*.tsv")).read_text().splitlines()
    head = tsv[0].split("\t")
    lines = [dict(zip(head, t.split("\t"))) for t in tsv[1:]]
    ops_rows = [t for t in lines if t["source"] == "openproteinset"]
    assert ops_rows
    for t in ops_rows[:20]:
        first, last = (int(x) for x in t["bytes"].split("-"))
        part = db[first:last + 1].decode()
        names = [x[1:].split()[0] for x in part.splitlines() if x.startswith(">")]
        assert names[int(t["record"])] == t["id"]


# --------------------------------------------------------------------------- 7: UniClust30 consensus layout


def test_uniclust30_file_in_consensus_layout_is_read():
    data = (REAL / "uniclust30-M4VQ73" / "uniclust30.a3m").read_bytes()
    p = parse_a3m_lines(_lines(data), Q_M4VQ73, consensus_frame=True)
    assert p.rows[0][1] == Q_M4VQ73 and p.rows[0][0] == "tr|M4VQ73|M4VQ73_9RHAB"
    assert len(p.rows) == 201 and p.skipped["consensus_record"] == 1
    assert p.consensus_identity == pytest.approx(202 / 204, abs=1e-3)
    assert all(len([c for c in s if c == "-" or c.isupper()]) == len(Q_M4VQ73) for _, s, _ in p.rows)


def test_uniclust30_consensus_that_does_not_fit_the_query_is_refused():
    data = (REAL / "uniclust30-M4VQ73" / "uniclust30.a3m").read_bytes()
    with pytest.raises(SourceMismatch):
        parse_a3m_lines(_lines(data), Q_M4VQ73)  # only for UniClust30 files
    shifted = Q_M4VQ73[1:] + Q_M4VQ73[0]
    with pytest.raises(SourceMismatch, match="consensus"):
        parse_a3m_lines(_lines(data), shifted, consensus_frame=True)


def test_uniclust30_rows_reach_the_alignment(hosts):
    hosts.uniparc[md5(Q_M4VQ73)] = (["M4VQ73"], [("Rabies lyssavirus", 11292)])
    hosts.uniclust.add("M4VQ73")
    hosts.s.static["/uniclust30/M4VQ73/a3m/uniclust30.a3m"] = (REAL / "uniclust30-M4VQ73" / "uniclust30.a3m").read_bytes()
    m = MSA(server=None, quiet=True).get(Q_M4VQ73)
    assert m.provenance["merge"]["kept_open_rows"] == {"uniclust30": 200}


# --------------------------------------------------------------------------- 8: one pairing message


def test_the_pairing_message_is_printed_once(complex_server, hosts, tmp_path, capsys):
    hosts.uniparc[md5(Q_7M22A)] = (["P1"], [("Homo sapiens", 9606)])
    hosts.uniparc[md5(Q_7M22C)] = (["P2"], [("Homo sapiens", 9606)])
    chains = MSA().get([Q_7M22A, Q_7M22C], remote=True)
    Boltz.prepare(chains, out_dir=tmp_path)
    err = capsys.readouterr().err
    assert len(re.findall(r"^scigantic-msa: paired \d", err, re.M)) == 1, err
