"""First-run journeys and messy input: real-world FASTA files, settings
from the environment, and what happens before anything is sent."""
import io

import pytest

from scigantic_msa import MSA
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.cli import main
from scigantic_msa.remote import RateLimited
from scigantic_msa.remote.colabfold import JobFailed, RemoteResult

Q = "MYRNEFTAWYRRMSVVYGIGTWSVLGSLLYYSRTMAKSSVDQKDGSASEVPSELSERPKGFYVETVVTYKEDFVPNTEKILNYWKSWTGGPGTEP"
Q2 = "GHSEQKLLVAGPRTWWKDLLNESAQLAK"
Q3 = "MKTAYIAKQRQISFVKSHFSRQLEERLG"


class FakeBackend:
    name, server_name, dbkey = "colabfold", "fake.invalid", "fakedb"

    def __init__(self):
        self.calls = []
        self.fail = {}

    def search(self, q):
        self.calls.append(q)
        if q in self.fail:
            raise self.fail[q]
        return RemoteResult(rows=parse_a3m(f">101\n{q}\n"), provenance={"server": "fake"})


@pytest.fixture
def fake(monkeypatch):
    b = FakeBackend()
    monkeypatch.setattr("scigantic_msa.tool.make_backend", lambda *a, **k: b)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return b


def run(*args):
    return main(["run", *args])


def test_fasta_with_blank_lines_and_an_indented_header(fake, tmp_path):
    f = tmp_path / "x.fasta"
    f.write_text(f"\n\n  >a\n{Q}\n")
    assert run(str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes") == 0
    assert fake.calls == [Q]
    assert (tmp_path / "o" / "boltz" / "a.yaml").exists()


def test_empty_record_is_named(fake, tmp_path, capsys):
    f = tmp_path / "x.fasta"
    f.write_text(f">a\n{Q}\n>nothing here\n\n>b\n{Q2}\n")
    assert run(str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--batch") == 1
    err = capsys.readouterr().err
    assert "record 2" in err and "nothing" in err and "empty" in err
    assert fake.calls == []


def test_bad_letters_name_the_record_not_sequence_1(fake, tmp_path, capsys):
    f = tmp_path / "x.fasta"
    f.write_text(f">a\n{Q}\n>b\n{Q2}1\n")
    assert run(str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--batch") == 1
    err = capsys.readouterr().err
    assert "record 2 (b)" in err and "'1'" in err and "sequence 1" not in err
    assert fake.calls == []


def test_empty_chain_from_a_stray_colon_is_named(fake, tmp_path, capsys):
    assert run(f"{Q}:", "-m", "boltz", "-o", str(tmp_path / "o"), "--yes") == 1
    err = capsys.readouterr().err
    assert "chain 2" in err and "empty" in err
    assert fake.calls == []


def test_name_with_a_digit_next_to_a_fasta_suggests_the_file(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r2.fasta").write_text(f">q\n{Q}\n")
    assert run("r2", "-m", "boltz", "-o", str(tmp_path / "o"), "--yes") == 1
    assert "r2.fasta" in capsys.readouterr().err
    assert fake.calls == []


def test_no_remote_still_uses_the_cache(fake, tmp_path, monkeypatch, capsys):
    assert run(Q, "-m", "boltz", "-o", str(tmp_path / "o1"), "--yes") == 0
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    capsys.readouterr()
    assert run(Q, "-m", "boltz", "-o", str(tmp_path / "o2")) == 0
    assert "nothing is sent" in capsys.readouterr().err
    assert (tmp_path / "o2" / "boltz" / "query.yaml").exists()
    assert MSA().get(Q, remote=True).source == "cache"
    assert fake.calls == [Q]


def test_no_remote_still_blocks_a_sequence_not_in_the_cache(fake, monkeypatch):
    from scigantic_msa import RemoteNotAllowed

    MSA().get(Q, remote=True)
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    with pytest.raises(RemoteNotAllowed):
        MSA().get([Q, Q2], remote=True)
    assert fake.calls == [Q]


@pytest.mark.parametrize("var,server", [("SCIGANTIC_MSA_COLABFOLD_URL", "colabfold"),
                                        ("SCIGANTIC_MSA_NVIDIA_URL", "nvidia")])
@pytest.mark.parametrize("url", ["localhost:8080", "file:///tmp", "ftp://127.0.0.1:21"])
def test_server_url_from_the_environment_is_checked(var, server, url, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(var, url)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(ValueError, match=var):
        MSA(server)
    assert run(Q, "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--server", server) == 1
    assert var in capsys.readouterr().err


def test_out_that_is_a_file_is_refused_before_sending(fake, tmp_path, capsys):
    (tmp_path / "out").write_text("")
    assert run(Q, "-m", "boltz", "-o", str(tmp_path / "out"), "--yes") == 1
    assert "not a directory" in capsys.readouterr().err
    assert fake.calls == []


def _three(tmp_path):
    f = tmp_path / "x.fasta"
    f.write_text(f">a\n{Q}\n>b\n{Q2}\n>c\n{Q3}\n")
    return f


def test_batch_goes_on_after_a_job_the_server_rejects(fake, tmp_path, capsys):
    fake.fail[Q2] = JobFailed("fake.invalid reported an error for job x")
    assert run(str(_three(tmp_path)), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--batch") == 1
    assert fake.calls == [Q, Q2, Q3]
    assert (tmp_path / "o" / "a" / "boltz" / "a.yaml").exists()
    assert (tmp_path / "o" / "c" / "boltz" / "c.yaml").exists()
    assert not (tmp_path / "o" / "b").exists()
    err = capsys.readouterr().err
    assert "1 of 3 jobs failed: b" in err


def test_batch_stops_when_rate_limited(fake, tmp_path, capsys):
    fake.fail[Q2] = RateLimited("rate limited")
    assert run(str(_three(tmp_path)), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--batch") == 1
    assert fake.calls == [Q, Q2]


def test_batch_says_which_job_it_is_on(fake, tmp_path, capsys):
    assert run(str(_three(tmp_path)), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--batch") == 0
    err = capsys.readouterr().err
    assert "job 1 of 3: a" in err and "job 3 of 3: c" in err
