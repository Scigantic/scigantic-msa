import io
import json
from pathlib import Path

import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, Q_LIVE, REAL, ColabFoldStub, make_result_tar
from test_colabfold import FakeColabFold
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.cli import MODELS, main, read_chains


def sent(server):
    return server.count(lambda r: r.path.startswith(("/ticket", "/result")))


class TTY(io.StringIO):
    def isatty(self):
        return True


def test_read_chains(tmp_path):
    assert read_chains("mkv aa") == ("query", ["MKVAA"])
    assert read_chains("MKV:GHS") == ("query", ["MKV", "GHS"])
    f = tmp_path / "x.fa"
    f.write_text(">job1 desc\nMKV\nAA\n>b\nGG:HH\n")
    assert read_chains(str(f)) == ("job1", ["MKVAA", "GG", "HH"])


def test_run_with_yes(colabfold, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    FakeColabFold(colabfold, ["UNKNOWN", "COMPLETE"])
    assert main(["run", Q_LIVE, "--model", "boltz", "--out", str(tmp_path / "o"), "--yes"]) == 0
    err = capsys.readouterr().err
    assert f"This sends 1 sequence (length {len(Q_LIVE)}) to 127.0.0.1" in err
    assert "boltz predict" in err and Q_LIVE not in err
    y = (tmp_path / "o" / "query.yaml").read_text()
    assert f"sequence: {Q_LIVE}" in y


def test_run_without_a_tty_and_without_yes_sends_nothing(colabfold, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    FakeColabFold(colabfold, ["COMPLETE"])
    assert main(["run", Q_LIVE, "-m", "af3", "-o", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "--yes" in err and "convert" in err
    assert sent(colabfold) == 0 and not (tmp_path / "o").exists()


def test_run_asks_once_and_honours_no(colabfold, tmp_path, capsys, monkeypatch):
    stub = ColabFoldStub(colabfold, {q: make_result_tar(q, [("UniRef100_X", q[:-1] + "A", "9606")])
                                     for q in (Q_7M22A, Q_7M22C)})
    asked = []
    monkeypatch.setattr("sys.stdin", TTY())
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "n")
    assert main(["run", f"{Q_7M22A}:{Q_7M22C}", "-m", "af3", "-o", str(tmp_path / "o")]) == 1
    assert len(asked) == 1 and stub.submits == 0 and sent(colabfold) == 0
    assert "Nothing was sent" in capsys.readouterr().err
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "y")
    assert main(["run", f"{Q_7M22A}:{Q_7M22C}", "-m", "af3", "-o", str(tmp_path / "o")]) == 0
    assert len(asked) == 2 and stub.submits == 2
    err = capsys.readouterr().err
    assert "This sends 2 sequences" in err and "--resolve_msa_overlaps=false" in err
    doc = json.loads((tmp_path / "o" / "query.json").read_text())
    assert len(doc["sequences"]) == 2


def test_run_from_cache_does_not_ask_or_send(colabfold, tmp_path, capsys, monkeypatch):
    FakeColabFold(colabfold, ["COMPLETE"])
    assert main(["run", Q_LIVE, "-m", "colabfold", "-o", str(tmp_path / "a"), "-y", "-q"]) == 0
    n = sent(colabfold)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["run", Q_LIVE, "-m", "colabfold", "-o", str(tmp_path / "b")]) == 0
    assert sent(colabfold) == n
    assert "nothing is sent" in capsys.readouterr().err
    assert parse_a3m((tmp_path / "b" / "query.a3m").read_text(), Q_LIVE)


def test_run_reports_server_failure_without_a_traceback(colabfold, tmp_path, capsys):
    FakeColabFold(colabfold, ["ERROR"])
    assert main(["run", Q_LIVE, "-m", "af3", "-o", str(tmp_path), "-y", "-q"]) == 1
    err = capsys.readouterr().err
    # The notice of where the sequence goes is printed even with -q.
    assert "This sends 1 sequence" in err and "\nerror: " in err and "reported an error" in err


@pytest.mark.parametrize("model", MODELS)
def test_convert_each_model(model, tmp_path, capsys):
    if model == "chai":
        pytest.importorskip("pyarrow")
    a3m = REAL / "of3-6t33_A" / "cfdb_uniref30.a3m"
    assert main(["convert", str(a3m), "-m", model, "-o", str(tmp_path / "o"), "--sequence", Q_6T33]) == 0
    err = capsys.readouterr().err
    assert "wrote" in err and "next:" in err
    assert any((tmp_path / "o").rglob("*"))


def test_convert_complex_and_checks(tmp_path, capsys):
    a = tmp_path / "a.a3m"
    b = tmp_path / "b.a3m"
    a.write_text((REAL / "of3-7m22_A" / "cfdb_uniref30.a3m").read_text())
    b.write_text((REAL / "of3-7m22_C" / "cfdb_uniref30.a3m").read_text())
    assert main(["convert", str(a), str(b), "-m", "protenix", "-o", str(tmp_path / "o"), "-q"]) == 0
    jobs = json.loads((tmp_path / "o" / "a.json").read_text())
    assert [s["proteinChain"]["sequence"] for s in jobs[0]["sequences"]] == [Q_7M22A, Q_7M22C]
    assert main(["convert", str(a), "-m", "af3", "-o", str(tmp_path / "x"), "--sequence", Q_7M22C]) == 1
    assert "not the sequence given" in capsys.readouterr().err
    empty = tmp_path / "empty.a3m"
    empty.write_text("")
    assert main(["convert", str(empty), "-m", "boltz", "-o", str(tmp_path / "y")]) == 1
    assert "empty" in capsys.readouterr().err
    assert main(["convert", str(tmp_path / "missing.a3m"), "-m", "boltz", "-o", str(tmp_path / "z")]) == 1


def test_version(capsys):
    """Against __version__, not a literal.

    This asserted "scigantic-msa 0.2.0" verbatim, so it had to be edited on
    every release and could not notice the thing worth noticing: 0.2.1 shipped
    reporting itself as 0.2.0, because pyproject.toml and __init__.py each
    carried a version and only one was bumped.
    """
    import scigantic_msa

    assert main(["version"]) == 0
    assert capsys.readouterr().out.startswith(f"scigantic-msa {scigantic_msa.__version__}")


def test_removed_commands_are_gone(capsys):
    for cmd in ("login", "logout", "push", "pull", "list", "rm", "account", "config", "where", "get", "index"):
        with pytest.raises(SystemExit):
            main([cmd])
    capsys.readouterr()


def test_output_names(tmp_path):
    a3m = REAL / "of3-6t33_A" / "cfdb_uniref30.a3m"
    assert main(["convert", str(a3m), "-m", "openfold3", "-o", str(tmp_path), "--name", "my job", "-q"]) == 0
    q = json.loads((tmp_path / "my_job.json").read_text())
    assert list(q["queries"]) == ["my job"]
    assert Path(q["queries"]["my job"]["chains"][0]["main_msa_file_paths"]).is_dir()


def test_run_with_a_missing_file_sends_nothing(colabfold, tmp_path, capsys):
    FakeColabFold(colabfold, ["COMPLETE"])
    assert main(["run", str(tmp_path / "chains.fasta"), "-m", "af3", "-o", str(tmp_path / "o"), "-y"]) == 1
    assert "not amino acid letters" in capsys.readouterr().err and sent(colabfold) == 0
