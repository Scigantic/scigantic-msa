"""Input safety and command line checks: what is refused before anything
is sent, what odd but valid files are read, and what the command line says."""
import io

import pytest

from scigantic_msa import MSA
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.cli import main
from scigantic_msa.remote.colabfold import RemoteResult

Q = "MYRNEFTAWYRRMSVVYGIGTWSVLGSLLYYSRTMAKSSVDQKDGSASEVPSELSERPKGFYVETVVTYKEDFVPNTEKILNYWKSWTGGPGTEP"
HIT = "MYRNEFTAWYRRMSVVYGIGTWSVLGSLLYYGRTMAKSSVDQKDGSTSEVPSELSERPKGFYVETVVTYKEDFVPNTEKILNYWKSWTGGPGTEP"
A3M = f">101\n{Q}\n>UniRef100_A0A2I2ZEU8\n{HIT}\n"


class FakeBackend:
    name, server_name, dbkey = "colabfold", "fake.invalid", "fakedb"

    def __init__(self):
        self.calls = []

    def search(self, q):
        self.calls.append(q)
        return RemoteResult(rows=parse_a3m(f">101\n{q}\n"), provenance={"server": "fake"})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    b = FakeBackend()
    monkeypatch.setattr("scigantic_msa.tool.make_backend", lambda *a, **k: b)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return b


def run(*args):
    return main(["run", *args])


# Defect 1: a mistyped file name, or a directory, is never sent.
def test_letters_only_missing_file_name_is_not_sent(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "query.fasta").write_text(f">q\n{Q}\n")
    assert run("query", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 1
    assert fake.calls == []
    err = capsys.readouterr().err
    assert "query.fasta" in err and "This sends" not in err


def test_short_literal_is_not_sent_without_equals(fake, tmp_path, capsys):
    assert run("chains", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 1
    assert fake.calls == []
    assert "=chains" in capsys.readouterr().err


def test_short_peptide_with_equals_is_sent(fake, tmp_path):
    assert run("=MKVLAAG", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    assert fake.calls == ["MKVLAAG"]


def test_directory_argument_is_not_sent(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chains").mkdir()
    assert run("chains", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 1
    assert fake.calls == []
    assert "directory" in capsys.readouterr().err


def test_long_directory_name_is_not_sent(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "MYRNEFTAWYRRMSVV").mkdir()
    assert run("MYRNEFTAWYRRMSVV", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 1
    assert fake.calls == []


def test_stdin_and_fasta_skip_the_length_rule(fake, tmp_path, monkeypatch):
    f = tmp_path / "pep.fa"
    f.write_text(">pep\nMKVLAAG\n")
    assert run(str(f), "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    monkeypatch.setattr("sys.stdin", io.StringIO(">p2\nGHSTW\n"))
    assert run("-", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    assert fake.calls == ["MKVLAAG", "GHSTW"]


# Defect 6: several FASTA records need --complex or --batch.
Q2 = "GHSTWQERLLGHSTWQERLLGHSTW"


def fasta(tmp_path, n, name="many.fa"):
    f = tmp_path / name
    f.write_text("".join(f">r{i}\n{Q[:20 + i]}\n" for i in range(n)))
    return f


def test_multi_record_fasta_needs_a_choice(fake, tmp_path, capsys):
    f = fasta(tmp_path, 3)
    assert run(str(f), "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 1
    err = capsys.readouterr().err
    assert "3 records" in err and "--complex" in err and "--batch" in err
    assert fake.calls == []


def test_complex_flag_makes_one_complex(fake, tmp_path, capsys):
    import json
    f = fasta(tmp_path, 3)
    assert run(str(f), "-m", "af3", "-o", str(tmp_path / "o"), "--yes", "--complex") == 0
    assert "One complex of 3 chains" in capsys.readouterr().err
    doc = json.loads((tmp_path / "o" / "r0.json").read_text())
    assert len(doc["sequences"]) == 3


def test_batch_writes_one_folder_per_record(fake, tmp_path, capsys):
    f = tmp_path / "b.fa"
    f.write_text(f">a x\n{Q}\n>b\n{Q2}\n>a\n{Q[:30]}\n")
    assert run(str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--batch") == 0
    err = capsys.readouterr().err
    assert "Batch of 3 separate jobs" in err
    for d in ("a", "b", "a_2"):
        assert (tmp_path / "o" / d / f"{d}.yaml").is_file()
    assert sorted(map(len, fake.calls)) == sorted([len(Q), len(Q2), 30])


def test_complex_and_batch_are_exclusive(fake, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(str(fasta(tmp_path, 2)), "-m", "af3", "-o", str(tmp_path / "o"), "--complex", "--batch")
    capsys.readouterr()


def test_caps(fake, tmp_path, capsys, monkeypatch):
    import scigantic_msa.cli as cli
    f = fasta(tmp_path, cli.MAX_CHAINS + 1)
    assert run(str(f), "-m", "af3", "-o", str(tmp_path / "o"), "--yes", "--complex") == 1
    assert f"at most {cli.MAX_CHAINS}" in capsys.readouterr().err
    colons = ":".join([Q2] * (cli.MAX_CHAINS + 1))
    assert run(colons, "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 1
    assert f"at most {cli.MAX_CHAINS}" in capsys.readouterr().err
    monkeypatch.setattr(cli, "MAX_JOBS", 2)
    assert run(str(fasta(tmp_path, 3)), "-m", "af3", "-o", str(tmp_path / "o"), "--yes", "--batch") == 1
    assert "at most 2" in capsys.readouterr().err
    assert fake.calls == []


def test_colon_in_one_record_is_still_a_complex(fake, tmp_path):
    assert run(f"{Q}:{Q2}", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    assert fake.calls == [Q, Q2]


# Defect 8: with remote search disabled the CLI does not say it sends.
def test_no_remote_cli_does_not_claim_it_sends(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    assert run(Q, "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 1
    err = capsys.readouterr().err
    assert "This sends" not in err and "disabled" in err
    assert fake.calls == []


# Defect 9: convert of a directory says it is a directory.
def test_convert_directory_message(tmp_path, capsys):
    (tmp_path / "d").mkdir()
    assert main(["convert", str(tmp_path / "d"), "-m", "af3", "-o", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert "does not exist" not in err and "directory" in err


# Defect 11: depth is always reported, and depth 1 or shallow is a warning.
def test_convert_query_only_says_depth_1(tmp_path, capsys):
    f = tmp_path / "q.a3m"
    f.write_text(f">q\n{Q}\n")
    assert main(["convert", str(f), "-m", "boltz", "-o", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert "sequence: depth 1" in err and "single-sequence" in err
    assert main(["convert", str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "-q"]) == 0
    err = capsys.readouterr().err
    assert "warning:" in err and "single-sequence" in err


def test_convert_shallow_and_deep(tmp_path, capsys):
    f = tmp_path / "s.a3m"
    f.write_text(A3M)
    assert main(["convert", str(f), "-m", "boltz", "-o", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert "sequence: depth 2" in err and "shallow" in err
    hits = "".join(f">h{i}\n{HIT[:i] + 'A' + HIT[i + 1:]}\n" for i in range(30))
    f.write_text(A3M + hits)
    assert main(["convert", str(f), "-m", "boltz", "-o", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert "depth 32" in err and "warning" not in err


def test_run_reports_depth_per_chain(fake, tmp_path, capsys):
    assert run(f"{Q}:{Q2}", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    err = capsys.readouterr().err
    assert "chain 1: depth 1" in err and "chain 2: depth 1" in err


# Item 12: DNA warning, Ctrl-C hint.
def test_dna_warns(fake, tmp_path, capsys):
    assert run("ACGTACGTTGCAACGT", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    assert "DNA or RNA" in capsys.readouterr().err
    assert run(Q, "-m", "af3", "-o", str(tmp_path / "p"), "--yes") == 0
    assert "DNA or RNA" not in capsys.readouterr().err


def test_ctrl_c_says_rerun_resumes(fake, tmp_path, capsys):
    def stop(q):
        raise KeyboardInterrupt
    fake.search = stop
    assert run(Q, "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 130
    assert "run the same command again" in capsys.readouterr().err
