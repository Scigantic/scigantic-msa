"""Open data: table lookup, reading the published files, merging with the
server's rows, dedup, pairing of complexes, what is sent where, the cache
and the attribution file. Every host is a local fake server serving the
real published files in fixtures/real."""
import gzip
import io
import json
import tarfile
import urllib.parse

import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, Q_LITEFOLD, REAL, ColabFoldStub, make_result_tar
from scigantic_msa import MSA
from scigantic_msa.a3m import Row
from scigantic_msa.adapters import Boltz
from scigantic_msa.cli import main
from scigantic_msa.opendata import lookup as lk
from scigantic_msa.opendata.lookup import Prov, species_name
from scigantic_msa.opendata.merge import accession, combine_pairs, merge_rows, neff_estimate, organism_rule
from scigantic_msa.opendata.parse import parse_a3m_lines, parse_sto_lines
from scigantic_msa.opendata.tables import iter_json_object
from scigantic_msa.remote.colabfold import REPLAY_ENV, parse_pair_result, query_text, ticket_id
from scigantic_msa.writers import ATTRIBUTION_FILE, ROWS_DIR

from conftest import OpenHosts, md5, LF_PATH, COMMIT  # noqa: F401


# --------------------------------------------------------------------------- lookup and open-only


def test_open_only_reads_the_published_files_of_an_exact_match(hosts):
    hosts.uniparc[md5(Q_6T33)] = (["P0DQ00.1"], [("Mediterraneibacter gnavus E1", 935582)])
    m = MSA(server=None, quiet=True, uniparc=True).get(Q_6T33)
    assert m.source == "open" and m.depth > 10
    srcs = {r.extra["prov"].source for r in m.rows[1:]}
    roles = {r.extra["prov"].role for r in m.rows[1:]}
    assert srcs == {"openfold3"} and {"cfdb_uniref30", "uniprot", "uniref90", "mgnify"} <= roles
    for r in m.rows[1:]:
        assert len([c for c in r.seq if c == "-" or c.isupper()]) == len(Q_6T33)
    tables = {t["table"]: t for t in m.provenance["open"]["tables"]}
    assert tables["openfold3"]["files"][0]["etag"] == "e1" and tables["openfold3"]["entries"] == 3
    assert m.provenance["organisms"]["uniparc"] == ["Mediterraneibacter gnavus"]
    assert m.provenance["merge"]["kept_open_rows"]["openfold3"] == m.depth - 1


def test_open_only_sends_only_the_md5_and_entry_requests(hosts):
    hosts.uniparc[md5(Q_6T33)] = (["P0DQ00"], [])
    MSA(server=None, quiet=True, uniparc=True).get(Q_6T33)
    reqs = hosts.s.requests
    assert all(r.method in ("GET", "HEAD") for r in reqs)
    for r in reqs:
        assert Q_6T33 not in r.path and Q_6T33.encode() not in r.body
        assert Q_6T33[:20] not in urllib.parse.unquote(r.path)
    uni = [r for r in reqs if r.path_only == "/uniparc/search"]
    assert len(uni) == 1 and md5(Q_6T33) in uni[0].query


def test_a_sequence_in_no_table_gets_the_query_only(hosts, capsys):
    q = "MKVLAAGIVGLLLSAPAWAQDTHE"
    m = MSA(server=None, quiet=True).get(q)
    assert m.depth == 1 and m.source == "open"


def test_litefold_entry_is_read_at_the_pinned_commit(hosts):
    m = MSA(server=None, quiet=True).get(Q_LITEFOLD)
    assert {r.extra["prov"].source for r in m.rows[1:]} == {"litefold"}
    assert any(COMMIT in p for p in hosts.get_paths())
    assert m.provenance["organisms"]["litefold"] == ["Homo sapiens"]


# A0A1A9ZI35 in UniProt; its published UniClust30 file starts with it.
Q_A0A1A9ZI35 = ("MKCVLWVAEGSPTHSTPVRIEDPLQFVPPNTNPKEFKRVQQQIKENRRADKISAGPQSHILEGVTWDEANRIKDATVSSAGDHVVLVGAASKGIIQRGYQH"
                "NATVYKAPYAKNPFDNVTDDELNEYKRTVERKKKSIHESEALSSQQTSGAQQHIKETTTTTTTTTAAVSQSEPEDVSDIILGLRISHLLFLFDAVSIQLL")


def test_uniclust30_entries_are_found_from_uniparc_accessions(hosts):
    hosts.uniparc[md5(Q_A0A1A9ZI35)] = (["Q1", "A0A1A9ZI35"], [("Glossina pallidipes", 7398)])
    hosts.uniclust.add("A0A1A9ZI35")
    a3m = REAL / "uniclust30-A0A1A9ZI35" / "uniclust30.a3m"
    hosts.s.static["/uniclust30/A0A1A9ZI35/a3m/uniclust30.a3m"] = a3m.read_bytes()
    m = MSA(server=None, quiet=True, uniparc=True).get(Q_A0A1A9ZI35)
    assert {r.extra["prov"].source for r in m.rows[1:]} == {"uniclust30"} and m.depth > 100
    # The answer is cached: a second run probes nothing.
    n = hosts.s.count(lambda r: r.path_only.startswith("/uniclust30/") and r.method == "HEAD")
    MSA(server=None, quiet=True, uniparc=True).get(Q_A0A1A9ZI35)
    assert hosts.s.count(lambda r: r.path_only.startswith("/uniclust30/") and r.method == "HEAD") == n


# --------------------------------------------------------------------------- cache and ETag


def test_second_run_reads_nothing_and_works_offline(hosts, monkeypatch):
    first = MSA(server=None, quiet=True).get(Q_6T33)
    n = len(hosts.s.requests)
    second = MSA(server=None, quiet=True).get(Q_6T33)
    assert [r.seq for r in second.rows] == [r.seq for r in first.rows]
    assert len(hosts.s.requests) == n
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    monkeypatch.setenv("SCIGANTIC_MSA_OPEN_REFRESH_HOURS", "0")
    third = MSA(server=None, quiet=True).get(Q_6T33)
    assert [r.seq for r in third.rows] == [r.seq for r in first.rows] and len(hosts.s.requests) == n


def test_no_remote_blocks_open_data_too(hosts, monkeypatch):
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert m.depth == 1 and hosts.s.requests == []


def test_a_changed_etag_downloads_the_table_again(hosts, monkeypatch):
    MSA(server=None, quiet=True).get(Q_6T33)
    gets = lambda: sum(1 for p in hosts.get_paths() if p.endswith("representives.fasta"))  # noqa: E731
    assert gets() == 1
    monkeypatch.setenv("SCIGANTIC_MSA_OPEN_REFRESH_HOURS", "0")
    MSA(server=None, quiet=True).get(Q_6T33)
    assert gets() == 1  # HEAD only: same ETag
    hosts.etag["fasta"] = "e2"
    hosts.fasta = f">6t33_A\n{Q_6T33}\n".encode()
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert gets() == 2
    t = {t["table"]: t for t in m.provenance["open"]["tables"]}["openfold3"]
    assert t["files"][0]["etag"] == "e2" and t["entries"] == 1


def test_within_a_day_the_etag_is_not_checked(hosts):
    MSA(server=None, quiet=True).get(Q_6T33)
    MSA(server=None, quiet=True).get(Q_6T33)
    assert hosts.s.count(lambda r: r.method == "HEAD" and r.path_only.endswith(".fasta")) == 0


# --------------------------------------------------------------------------- merge


def P(src="openfold3", role="uniprot", rec=0):
    return Prov(src, role, "e", "u", rec)


def row(h, s, src="openfold3", role="uniprot", taxid=None):
    r = Row(h, s, "other", taxid)
    r.extra["prov"] = P(src, role)
    return r


def test_merge_drops_same_residues_and_same_entry_span_but_keeps_other_spans():
    q = "ACDEFGHIKLMNPQRSTVWY"
    server = [row("UniRef100_P11111\t1\t1\t1\t1\t20\t20\t1\t20\t30", "ACDEFGHIKLMNPQRSTVWF", "colabfold", "uniref")]
    open_rows = [
        row("tr|P11111|X_HUMAN/1-20 OX=9606", "ACDEFGHIKLMNPQRSTVWF"),           # same residues
        row("tr|P11111|X_HUMAN/2-19 OX=9606", "-CDEFGHIKLMNPQRSTVW-"),          # same entry, same part
        row("tr|P11111|X_HUMAN/40-59 OX=9606", "ACDEFGHIKLMNPQRSTVWA"),         # same entry, other part
        row("tr|Q22222|Y_MOUSE/1-20 OX=10090", "ACDEFGHIKLMNPQRSTVAA"),
        row("tr|Q33333|Z_MOUSE/1-20 OX=10090", q),                              # copy of the query
    ]
    out, cnt = merge_rows(q, server, open_rows)
    heads = [r.header.split()[0] for r in out]
    assert heads[0].startswith("UniRef100_P11111")
    assert "tr|P11111|X_HUMAN/40-59" in heads and "tr|Q22222|Y_MOUSE/1-20" in heads
    assert len(out) == 3
    assert cnt.same_residues == 1 and cnt.same_entry_span == 1 and cnt.query_copies == 1
    # The server row keeps its own species (none); it never takes a copy's.
    assert out[0].taxid is None and server[0].taxid is None
    assert "openfold3:uniprot" in out[0].extra["also"]


def test_merge_keeps_the_copy_with_more_residues():
    q = "ACDEFGHIKLMNPQRSTVWY"
    a = row("tr|P11111|X/1-10 OX=1", "ACDEFGHIKL----------")
    b = row("tr|P11111|X/1-20 OX=1", "ACDEFGHIKLMNPQRSTVWA", role="uniref90")
    out, cnt = merge_rows(q, [], [a, b])
    assert out == [b] and cnt.same_entry_span == 1


def test_open_rows_are_interleaved_by_file():
    q = "ACDEFGHIKL"
    f1 = [row(f"A{i}", "ACDEFGHIKA"[:9] + "ACDEFGHIK"[i], role="uniprot") for i in range(3)]
    f2 = [row(f"B{i}", "CCDEFGHIK" + "ACD"[i], role="mgnify") for i in range(3)]
    out, _ = merge_rows(q, [], f1 + f2)
    assert [r.header[0] for r in out] == ["A", "B", "A", "B", "A", "B"]


def test_accession_forms():
    assert accession("tr|A0A1|X_HUMAN/3-50 OX=1") == "A0A1"
    assert accession("UniRef100_P12345\t10\t0.9") == "P12345"
    assert accession("UniRef90_UPI0001/1-20 n=1") == "UPI0001"
    assert accession("MGYP000123/5-9") == "MGYP000123"


def test_neff_estimate_is_exact_when_small_and_close_when_sampled():
    q = "ACDEFGHIKLMNPQRSTVWY"
    rows = [Row("a", q[:-1] + "A"), Row("b", "W" * 20), Row("c", "Y" * 20)]
    # The query and row a are within 80% of each other; b and c of nothing.
    assert neff_estimate(q, rows) == pytest.approx(0.5 + 0.5 + 1 + 1)
    import random

    rng = random.Random(1)
    many = [Row("r", "".join(rng.choice("ACDEFGHIKL") for _ in range(20))) for _ in range(600)]
    exact = neff_estimate(q, many, sample=10_000, against=10_000)
    est = neff_estimate(q, many)
    assert abs(est - exact) / exact < 0.1


# --------------------------------------------------------------------------- pairing


def test_organism_rule():
    assert organism_rule([{"Homo sapiens"}, {"Homo sapiens", "Pan troglodytes"}], ["A", "B"])[0]
    ok, why = organism_rule([{"Homo sapiens"}, {"Escherichia coli"}], ["A", "B"])
    assert not ok and "different organisms" in why
    ok, why = organism_rule([{"Homo sapiens"}, set()], ["A", "B"])
    assert not ok and "chain B is not known" in why


def test_species_name():
    assert species_name("Escherichia coli (strain K12)") == "Escherichia coli"
    assert species_name("Severe acute respiratory syndrome coronavirus 2") == \
        "Severe acute respiratory syndrome coronavirus 2"
    assert species_name("synthetic construct") is None
    assert species_name("Candidatus Pelagibacter ubique HTCC1062") == "Candidatus Pelagibacter ubique"


def _pairs_fixture():
    qa, qb = "ACDEFGHIKL", "MNPQRSTVWY"
    sa = [Row("UniRef100_A1", "ACDEFGHIKA", "uniref", "9606")]
    sb = [Row("UniRef100_B1", "MNPQRSTVWA", "uniref", "9606")]
    oa = [Row("query", qa), Row("tr|A1|X OX=9606", "ACDEFGHIKC"), Row("tr|A2|X OX=10090", "ACDEFGHIKD")]
    ob = [Row("query", qb), Row("tr|B1|X OX=9606", "MNPQRSTVWC"), Row("tr|B2|X OX=10090", "MNPQRSTVWD")]
    merged = [[Row("q", qa)] + sa + oa[1:], [Row("q", qb)] + sb + ob[1:]]
    return qa, qb, sa, sb, oa, ob, merged


def test_server_pairs_first_then_only_new_species_from_open():
    qa, qb, sa, sb, oa, ob, merged = _pairs_fixture()
    plan = combine_pairs([qa, qb], merged, [oa, ob], ["A", "B"], server_pairs=[sa, sb], add_open=True)
    p = plan.pairing
    assert plan.server_pairs == 1 and plan.open_pairs == 1
    assert p.rows[0][0] is sa[0] and p.keys == ["server:1", "tx:10090"]
    assert "1 more by species from open data" in p.message


def test_server_pairs_that_repeat_a_row_in_one_chain_are_kept_but_repeated_pairs_are_not():
    qa, qb, sa, sb, oa, ob, merged = _pairs_fixture()
    sa2 = [sa[0], Row("UniRef100_A9", sa[0].seq, "uniref"), Row("UniRef100_A8", sa[0].seq, "uniref")]
    sb2 = [sb[0], Row("UniRef100_B9", "MNPQRSTVWE", "uniref"), Row("UniRef100_B8", "MNPQRSTVWE", "uniref")]
    plan = combine_pairs([qa, qb], merged, [oa, ob], ["A", "B"], server_pairs=[sa2, sb2], add_open=False)
    assert plan.server_pairs == 2 and "1 more repeat another pair" in plan.pairing.message


def test_no_open_pairs_when_the_rule_says_no():
    qa, qb, sa, sb, oa, ob, merged = _pairs_fixture()
    plan = combine_pairs([qa, qb], merged, [oa, ob], ["A", "B"], server_pairs=[sa, sb], add_open=False,
                         rule="the chains come from different organisms")
    assert plan.open_pairs == 0 and plan.pairing.n_paired == 1
    assert "different organisms" in plan.pairing.message


def test_parse_pair_result_with_a_gap_row():
    qa, qb = "ACDEFGHIKL", "MNPQRSTVWY"
    text = (f">101\n{qa}\n>UniRef100_A1\nACDEFGHIKA\n>x\n----------\n\x00"
            f">102\n{qb}\n>UniRef100_B1\nMNPQRSTVWA\n>UniRef100_B2\nMNPQRSTVWC\n\x00")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("pair.a3m")
        info.size = len(text.encode())
        tf.addfile(info, io.BytesIO(text.encode()))
    res = parse_pair_result(buf.getvalue(), [qa, qb], "t")
    assert [r is None for r in res.rows[0]] == [False, True] and len(res.rows[1]) == 2


# --------------------------------------------------------------------------- default run: server + open


def test_default_run_merges_server_and_open_and_pairs(complex_server, hosts):
    hosts.uniparc[md5(Q_7M22A)] = (["P1"], [("Homo sapiens", 9606)])
    hosts.uniparc[md5(Q_7M22C)] = (["P2"], [("Homo sapiens (human)", 9606)])
    a, c = MSA(quiet=True).get([Q_7M22A, Q_7M22C], remote=True)
    assert a.source == "merged" and a.rows[1].header.startswith("UniRef100_S1")
    cp = a.complex_pairing
    assert cp is c.complex_pairing and cp.server_pairs == 1
    assert cp.pairing.keys[0] == "server:1"
    assert cp.open_pairs > 0 and "Homo sapiens" in cp.rule
    assert hosts.s.count(lambda r: r.path_only == "/ticket/pair") == 1
    # The paired search is cached too.
    MSA(quiet=True).get([Q_7M22A, Q_7M22C], remote=True)
    assert hosts.s.count(lambda r: r.path_only == "/ticket/pair") == 1


def test_cross_organism_complex_gets_server_pairs_only(complex_server, hosts):
    # The organisms then come from UniParc alone (the UniProt hits of 7m22_C
    # would also name its own organism).
    del hosts.s.static["/msas/pdb_msas_completed/7m22_C/uniprot_hits.sto"]
    hosts.uniparc[md5(Q_7M22A)] = (["P1"], [("Homo sapiens", 9606)])
    hosts.uniparc[md5(Q_7M22C)] = (["P2"], [("Dendroaspis angusticeps", 8618)])
    a, _ = MSA(quiet=True).get([Q_7M22A, Q_7M22C], remote=True)
    cp = a.complex_pairing
    assert cp.server_pairs == 1 and cp.open_pairs == 0 and "different organisms" in cp.rule
    a, _ = MSA(quiet=True, open_pairs="always").get([Q_7M22A, Q_7M22C], remote=True)
    assert a.complex_pairing.open_pairs > 0


def test_unknown_organism_gets_no_open_pairs(complex_server, hosts):
    del hosts.s.static["/msas/pdb_msas_completed/7m22_C/uniprot_hits.sto"]
    a, _ = MSA(quiet=True).get([Q_7M22A, Q_7M22C], remote=True)
    assert a.complex_pairing.open_pairs == 0 and "not known" in a.complex_pairing.rule


def test_writers_use_the_complex_pairs_and_attribution_lists_sources(complex_server, hosts, tmp_path):
    hosts.uniparc[md5(Q_7M22A)] = (["P1"], [("Homo sapiens", 9606)])
    hosts.uniparc[md5(Q_7M22C)] = (["P2"], [("Homo sapiens", 9606)])
    chains = MSA(quiet=True).get([Q_7M22A, Q_7M22C], remote=True)
    Boltz.prepare(chains, out_dir=tmp_path, quiet=True)
    n = chains[0].complex_pairing.pairing.n_paired
    csvs = sorted(tmp_path.glob("*.csv"))
    keyed = [sum(1 for line in p.read_text().splitlines()[2:] if not line.startswith("-1,")) for p in csvs]
    assert keyed == [n, n]
    doc = json.loads((tmp_path / ATTRIBUTION_FILE).read_text())
    ids = {s["id"]: s for s in doc["sources"]}
    assert ids["openfold3"]["license"] == "CC-BY-4.0" and ids["openfold3"]["table"]["files"][0]["etag"] == "e1"
    assert "colabfold-api" in ids and "uniprot-uniparc" in ids
    chain = doc["outputs"][0]["chains"][0]
    assert chain["provenance"]["merge"]["kept_open_rows"]["openfold3"] > 0
    assert "ticket" not in json.dumps(chain["provenance"]["pairing"])
    tables = doc["outputs"][0]["files"]["row_provenance"]
    t = (tmp_path / tables[chains[0].seq_sha256]).read_text().splitlines()
    assert t[0].split("\t")[:3] == ["row", "id", "source"]
    assert len([x for x in t[1:] if not x.startswith("pair")]) == chains[0].depth - 1
    assert any("\topenfold3\tuniprot\t7m22_A\t" in x for x in t)
    assert (tmp_path / ROWS_DIR).is_dir()


def test_open_only_run_from_the_command_line(hosts, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # no terminal, no --yes: still runs
    assert main(["run", Q_6T33, "-m", "colabfold", "-o", str(tmp_path), "--open-only"]) == 0
    err = capsys.readouterr().err
    assert "not sent to any search server" in err and "MD5" not in err
    assert "rows from open data" in err
    assert "open data sent: requests for the openfold3 table" in err
    assert "requests for the public files of openfold3 entry 6t33_A" in err
    assert not any(r.path_only == "/uniparc/search" for r in hosts.s.requests)
    assert main(["run", Q_6T33, "-m", "colabfold", "-o", str(tmp_path), "--open-only", "--uniparc"]) == 0
    err = capsys.readouterr().err
    assert "MD5" in err and "open data sent: the MD5 of the sequence to" in err
    assert main(["run", Q_6T33, "-m", "colabfold", "-o", str(tmp_path), "--open-only", "--uniparc"]) == 0
    assert "open data sent: nothing (tables" in capsys.readouterr().err
    assert all(not r.path_only.startswith("/ticket") for r in hosts.s.requests)


def test_open_only_refuses_a_host(tmp_path, capsys):
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only", "--host", "http://x"]) == 1


def test_default_cli_summary_separates_server_and_open(colabfold, hosts, tmp_path, capsys):
    ColabFoldStub(colabfold, {Q_6T33: make_result_tar(Q_6T33, [("UniRef100_S1", Q_6T33[:-1] + "A", "9606")])})
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--yes"]) == 0
    err = capsys.readouterr().err
    assert "ColabFold alone: 1 rows" in err and "Open data added" in err


# --------------------------------------------------------------------------- replay hook


def test_replay_folder_answers_without_any_request(server, monkeypatch, tmp_path):
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", server.url)
    tid = ticket_id(query_text(Q_6T33), "env-taxonomy")
    (tmp_path / f"{tid}.tar.gz").write_bytes(make_result_tar(Q_6T33, [("UniRef100_X", Q_6T33[:-1] + "A", "1")]))
    monkeypatch.setenv(REPLAY_ENV, str(tmp_path))
    m = MSA(quiet=True, open_data=False).get(Q_6T33, remote=True)
    assert m.depth == 2 and server.requests == []


# --------------------------------------------------------------------------- parsers


def test_iter_json_object_across_chunk_boundaries():
    doc = {f"k{i}": {"seq": "A" * (i % 7), "n": i * 1234567, "l": [1, 2.5, None]} for i in range(300)}
    text = json.dumps(doc, indent=2)
    assert dict(iter_json_object(io.StringIO(text), chunk=37)) == doc


def test_wrapped_stockholm_is_read_again_and_joined():
    q = "ACDEFGHIKLMN"
    text = ("# STOCKHOLM 1.0\n\n#=GS h1/1-12 DE x OX=5\n\n"
            "query ACDEFG\nh1/1-12 ACDEFA\nh2 ACD---\n\n"
            "query HIKLMN\nh1/1-12 HIKLMA\nh2 HIKLMN\n//\n").encode()
    lines = text.split(b"\n")
    p = parse_sto_lines([x + b"\n" for x in lines], q, reopen=lambda: [x + b"\n" for x in lines])
    assert [r[1] for r in p.rows] == [q, "ACDEFAHIKLMA", "ACD---HIKLMN"]
    assert p.rows[1][0] == "h1/1-12 x OX=5"


def test_a_wrapped_stockholm_over_the_cap_keeps_only_its_first_rows():
    q = "ACDEFGHIKLMN"
    names = ["query"] + [f"h{i}" for i in range(20)]
    b1 = "".join(f"{n} {'ACDEFG' if n == 'query' else 'ACDEFA'}\n" for n in names)
    b2 = "".join(f"{n} {'HIKLMN' if n == 'query' else 'HIKLMA'}\n" for n in names)
    text = f"# STOCKHOLM 1.0\n\n{b1}\n{b2}//\n".encode()
    lines = [x + b"\n" for x in text.split(b"\n")]
    p = parse_sto_lines(lines, q, max_bytes=len(text) // 4, size=len(text), reopen=lambda: iter(lines))
    assert 1 < len(p.rows) <= 7 and p.truncated
    assert all(r[1] == "ACDEFAHIKLMA" for r in p.rows[1:])


def test_caps_cut_a_file_and_say_so():
    q = "ACDEFGHIKL"
    body = f">q\n{q}\n" + "".join(f">r{i}\nACDEFGHIK{'ACDEFGHIKLMNPQRSTVWY'[i % 20]}\n" for i in range(1, 40))
    lines = [x.encode() + b"\n" for x in body.splitlines()]
    p = parse_a3m_lines(lines, q, max_rows=5)
    assert len(p.rows) == 6 and p.truncated
    p = parse_a3m_lines(lines, q, max_bytes=60)
    assert p.truncated and len(p.rows) < 6
    p = parse_a3m_lines(lines, q)
    assert not p.truncated and p.skipped["copy_of_query"] == 2


def test_a_file_whose_query_differs_is_refused(hosts):
    hosts.s.static["/msas/pdb_msas_completed/6t33_A/cfdb_uniref30.a3m"] = b">x\nAAAA\n"
    m = MSA(server=None, quiet=True).get(Q_6T33)
    err = [o for o in m.provenance["open"]["objects"] if o["role"] == "cfdb_uniref30"][0]["error"]
    assert "not the expected query" in err
    assert m.depth > 1  # the other files still count


def test_cached_objects_are_private(hosts):
    MSA(server=None, quiet=True).get(Q_6T33)
    import os

    root = lk.Path(os.environ["SCIGANTIC_MSA_CACHE"]) / "open" / "objects"
    files = [p for p in root.rglob("*.a3m.gz")]
    assert files and all(p.stat().st_mode & 0o077 == 0 for p in files)
    assert gzip.decompress(files[0].read_bytes()).startswith(b"#scigantic-msa-open ")


def test_openproteinset_entries_are_read_by_byte_range(hosts):
    ops = REAL / "ops-6t33_A"
    blob, files, pos = b"", [], 0
    for name in ("bfd_uniclust_hits.a3m", "mgnify_hits.a3m", "pdb70_hits.hhr", "uniref90_hits.a3m"):
        data = (ops / name).read_bytes()
        files.append([name, pos, len(data)])
        blob += data
        pos += len(data)
    cache = {"6t33_B": {"seq": Q_6T33.lower()}, "6t33_A": {"seq": Q_6T33}, "9xyz_A": {"seq": "MKV"}}
    index = {"6t33_A": {"db": "alignment_db_3.db", "files": files}, "6t33_B": {"db": "alignment_db_3.db", "files": files}}
    hosts.s.static["/data_caches/chain_data_cache.json"] = json.dumps(cache, indent=4).encode()
    hosts.s.static["/alignment_db/alignment_db.index"] = json.dumps(index, indent=4).encode()
    hosts.s.static["/alignment_db/alignment_db_3.db"] = b"x" * 100 + blob
    for f in files:
        f[1] += 100
    hosts.s.static["/alignment_db/alignment_db.index"] = json.dumps(index, indent=4).encode()
    m = MSA(server=None, quiet=True, openproteinset=True).get(Q_6T33)
    roles = {r.extra["prov"].role for r in m.rows[1:] if r.extra["prov"].source == "openproteinset"}
    assert roles == {"uniref90", "bfd_uniclust", "mgnify"}
    ranged = [r for r in hosts.s.requests if r.path_only.endswith("alignment_db_3.db")]
    assert ranged and all(r.headers.get("Range", "").startswith("bytes=") for r in ranged)


def test_sources_command_lists_sets_and_fetches_tables(hosts, capsys):
    assert main(["sources"]) == 0
    out = capsys.readouterr().out
    assert "openfold3: OpenFold3 training MSAs for PDB chains, CC-BY-4.0" in out
    assert "table openfold3: not downloaded" in out and "UniClust30: not confirmed" in out
    assert main(["sources", "--fetch"]) == 0
    out = capsys.readouterr().out
    assert "table openfold3: 3 sequences" in out and "table litefold: 1 sequences" in out
    assert "table openproteinset: not downloaded" in out
