import gzip
import logging

import pytest

from conftest import Q_6T33, Q_LIVE, REAL, ColabFoldStub, make_result_tar
from test_colabfold import FakeColabFold
from scigantic_msa import MSA, A3MError, RemoteError, RemoteNotAllowed

SEQ_B = "GHSEQKLLVAGPRTWWKD"


def calls(server, prefix=("/ticket", "/result")):
    return server.count(lambda r: r.path.startswith(prefix))


def test_get_refuses_without_remote_true_and_names_the_server(colabfold):
    FakeColabFold(colabfold, ["COMPLETE"])
    for bad in (False, None, 1, "yes", "colabfold"):
        with pytest.raises(RemoteNotAllowed, match="127.0.0.1") as e:
            MSA().get(Q_LIVE, remote=bad)
        assert "remote=True" in str(e.value)
    assert calls(colabfold) == 0


def test_the_error_names_the_public_server_by_default(monkeypatch):
    monkeypatch.delenv("SCIGANTIC_MSA_COLABFOLD_URL", raising=False)
    with pytest.raises(RemoteNotAllowed, match="api.colabfold.com .*anyone who knows the sequence"):
        MSA().get(Q_LIVE)
    with pytest.raises(RemoteNotAllowed, match="health.api.nvidia.com"):
        MSA("nvidia").get(Q_LIVE)


def test_kill_switch(colabfold, monkeypatch):
    FakeColabFold(colabfold, ["COMPLETE"])
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    with pytest.raises(RemoteNotAllowed, match="SCIGANTIC_MSA_NO_REMOTE"):
        MSA().get(Q_LIVE, remote=True)
    assert calls(colabfold) == 0


def test_get_then_cache_does_not_submit_again(colabfold, capsys):
    FakeColabFold(colabfold, ["UNKNOWN", "COMPLETE"])
    m = MSA().get(Q_LIVE, remote=True)
    assert m.source == "remote" and m.rows[0].seq == Q_LIVE and m.depth > 1000
    assert m.provenance["server"] == colabfold.url
    n = calls(colabfold)
    again = MSA().get(Q_LIVE, remote=True)
    assert again.source == "cache" and calls(colabfold) == n
    assert [(r.seq, r.taxid) for r in again.rows] == [(r.seq, r.taxid) for r in m.rows]
    assert MSA().cached(Q_LIVE) is not None
    assert MSA(cache=False).cached(Q_LIVE) is None
    err = capsys.readouterr().err
    assert "nothing sent" in err


def test_no_cache_means_no_cache_files(colabfold, tmp_path):
    FakeColabFold(colabfold, ["COMPLETE"])
    MSA(cache=False).get(Q_LIVE, remote=True)
    assert not (tmp_path / "cache").exists()
    MSA(cache_dir=tmp_path / "mine").get(Q_LIVE, remote=True)
    assert any((tmp_path / "mine").rglob("*.msa.gz"))


def test_complex_searches_each_distinct_chain_once(colabfold):
    stub = ColabFoldStub(colabfold, {
        Q_6T33: make_result_tar(Q_6T33, [("UniRef100_A1", Q_6T33[:-1] + "A", "9606")]),
        SEQ_B: make_result_tar(SEQ_B, [("UniRef100_B1", SEQ_B[:-1] + "A", "9606")]),
    })
    got = MSA(quiet=True).get([Q_6T33, SEQ_B, Q_6T33], remote=True)
    assert [g.query for g in got] == [Q_6T33, SEQ_B, Q_6T33]
    assert got[0] is got[2]
    assert stub.submits == 2
    assert got[0].rows[1].taxid == "9606"


def test_server_errors_surface_as_remote_error(colabfold):
    FakeColabFold(colabfold, ["ERROR"])
    with pytest.raises(RemoteError, match="reported an error"):
        MSA(quiet=True).get(Q_LIVE, remote=True)
    with pytest.raises(ValueError, match="unknown server"):
        MSA("elsewhere")


def test_the_sequence_is_never_logged(colabfold, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    FakeColabFold(colabfold, ["UNKNOWN", "PENDING", "COMPLETE"])
    MSA().get(Q_LIVE, remote=True)
    MSA().get(Q_LIVE, remote=True)
    out = capsys.readouterr()
    text = out.out + out.err + caplog.text
    assert "sending the sequence to 127.0.0.1" in text
    assert Q_LIVE not in text and Q_LIVE[:12] not in text
    # Errors do not quote it either.
    with pytest.raises(A3MError) as e:
        MSA().from_a3m(f">q\n{Q_LIVE}\n", sequence=Q_6T33)
    assert Q_LIVE[:12] not in str(e.value) and Q_6T33[:12] not in str(e.value)


def test_from_a3m_reads_files_and_text_without_network(tmp_path):
    p = REAL / "of3-6t33_A" / "cfdb_uniref30.a3m"
    m = MSA().from_a3m(p)
    assert m.source == "file" and m.query == Q_6T33 and m.provenance["file"] == p.name
    assert MSA().from_a3m(str(p), sequence=Q_6T33.lower()).depth == m.depth
    gz = tmp_path / "x.a3m.gz"
    gz.write_bytes(gzip.compress(p.read_bytes()))
    assert MSA().from_a3m(gz).depth == m.depth
    assert MSA().from_a3m(f">q\n{Q_6T33}\n>h\n{Q_6T33[:-1]}-\n").depth == 2
    sto = MSA().from_a3m(REAL / "of3-6t33_A" / "uniprot_hits.sto")
    assert sto.query == Q_6T33 and sto.depth > 1


@pytest.mark.parametrize("text,msg", [
    ("", "empty"),
    ("   \n", "empty"),
    (">q\nMKV-L\n", "gaps or lowercase"),
    (">q\nMKVL\n>h\nMKV\n", "match columns"),
    (">q\nMKVL\n>h\nMK*L\n", "characters outside"),
    ("#4,3\t1,1\n>101\t102\nMKVLGHS\n", "complex header"),
])
def test_from_a3m_rejects_bad_files(tmp_path, text, msg):
    p = tmp_path / "bad.a3m"
    p.write_text(text)
    with pytest.raises(A3MError, match=msg):
        MSA().from_a3m(p)


def test_from_a3m_checks_the_sequence(tmp_path):
    with pytest.raises(A3MError, match="does not match the query"):
        MSA().from_a3m(f">q\n{Q_6T33}\n", sequence=Q_6T33[:-1] + "W")
    with pytest.raises(FileNotFoundError):
        MSA().from_a3m(tmp_path / "missing.a3m")
