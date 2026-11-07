import io
import json

import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, Q_LIVE
from test_colabfold import FakeColabFold
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.cli import main, read_jobs


@pytest.fixture
def env(index_setup, server, monkeypatch):
    _, base, _ = index_setup
    monkeypatch.setenv("SCIGANTIC_MSA_INDEX_URL", base)
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", server.url)
    return server


def test_read_jobs(tmp_path):
    assert read_jobs("mkv aa") == [("query", ["MKVAA"])]
    assert read_jobs("MKV:GHS") == [("query", ["MKV", "GHS"])]
    f = tmp_path / "x.fa"
    f.write_text(">a desc\nMKV\nAA\n>b\nGG:HH\n")
    assert read_jobs(str(f)) == [("a", ["MKVAA"]), ("b", ["GG", "HH"])]


def test_get_a3m_to_stdout(env, capsys):
    assert main(["get", Q_6T33, "-q"]) == 0
    out = capsys.readouterr().out
    assert parse_a3m(out, Q_6T33)[0].seq == Q_6T33


def test_get_af3_complex(env, tmp_path, capsys):
    out = tmp_path / "in.json"
    assert main(["get", f"{Q_7M22A}:{Q_7M22C}", "--format", "af3", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert len(doc["sequences"]) == 2
    assert all(s["protein"]["pairedMsa"] for s in doc["sequences"])


def test_get_each_format_writes_files(env, tmp_path, capsys):
    try:
        import pyarrow  # noqa: F401

        formats = ("boltz", "chai", "colabfold", "openfold3")
    except ImportError:
        formats = ("boltz", "colabfold", "openfold3")
        assert main(["get", Q_6T33, "-f", "chai", "-o", str(tmp_path / "c") + "/", "-q"]) == 1
        assert "pip install 'scigantic-msa[chai]'" in capsys.readouterr().err
    for fmt in formats:
        d = tmp_path / fmt
        assert main(["get", Q_6T33, "-f", fmt, "-o", str(d) + "/", "-q"]) == 0, fmt
        assert any(d.rglob("*")), fmt


def test_get_miss_without_tty_does_not_prompt_or_send(env, capsys, monkeypatch):
    FakeColabFold(env, ["COMPLETE"])
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["get", Q_LIVE]) == 1
    err = capsys.readouterr().err
    assert "remote='colabfold'" in err
    assert env.count(lambda r: r.path.startswith("/ticket")) == 0


def test_get_with_explicit_remote(env, capsys):
    FakeColabFold(env, ["COMPLETE"])
    assert main(["get", Q_LIVE, "--remote", "colabfold", "-q"]) == 0
    assert parse_a3m(capsys.readouterr().out, Q_LIVE)


def test_where(env, capsys):
    assert main(["where", Q_6T33]) == 0
    out = capsys.readouterr().out
    assert "exact    hit" in out and "-> exact would answer" in out
    assert main(["where", Q_LIVE]) == 0
    assert "nothing would answer" in capsys.readouterr().out


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.startswith("scigantic-msa 0.1.0")


def test_index_info(env, capsys):
    assert main(["index", "info"]) == 0
    assert "test-build-1" in capsys.readouterr().out


