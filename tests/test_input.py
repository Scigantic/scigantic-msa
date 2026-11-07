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
    doc = json.loads((tmp_path / "o" / "af3" / "r0.json").read_text())
    assert len(doc["sequences"]) == 3


def test_batch_writes_one_folder_per_record(fake, tmp_path, capsys):
    f = tmp_path / "b.fa"
    f.write_text(f">a x\n{Q}\n>b\n{Q2}\n>a\n{Q[:30]}\n")
    assert run(str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--batch") == 0
    err = capsys.readouterr().err
    assert "Batch of 3 separate jobs" in err
    for d in ("a", "b", "a_2"):
        assert (tmp_path / "o" / d / "boltz" / f"{d}.yaml").is_file()
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


# Defect 7: a byte order mark, CR-only line endings and an HH-suite '#A3M#'
# first line are valid input.
BOM = b"\xef\xbb\xbf"
STO = f"# STOCKHOLM 1.0\n101 {Q}\nhit {HIT}\n//\n"


@pytest.mark.parametrize("body", [
    BOM + A3M.encode(),
    A3M.replace("\n", "\r").encode(),
    BOM + A3M.replace("\n", "\r\n").encode(),
    b"#A3M#\n" + A3M.encode(),
    BOM + b"#A3M#\r" + A3M.replace("\n", "\r").encode(),
], ids=["bom", "cr", "bom-crlf", "hhsuite-tag", "all"])
def test_odd_but_valid_a3m(tmp_path, body):
    p = tmp_path / "x.a3m"
    p.write_bytes(body)
    assert MSA().from_a3m(p).depth == 2
    assert MSA().from_a3m(body.decode("utf-8")).depth == 2
    assert len(parse_a3m(body)) == 2


@pytest.mark.parametrize("body", [BOM + STO.encode(), STO.replace("\n", "\r").encode()], ids=["bom", "cr"])
def test_odd_but_valid_stockholm(tmp_path, body):
    p = tmp_path / "x.sto"
    p.write_bytes(body)
    assert MSA().from_a3m(p).depth == 2


def test_hhsuite_tag_only_first(tmp_path):
    from scigantic_msa.a3m import A3MError
    with pytest.raises(A3MError, match="not a ColabFold header"):
        parse_a3m(A3M + "#A3M#\n")


@pytest.mark.parametrize("body", [BOM + f">q\n{Q}\n".encode(), f">q\r{Q[:40]}\r{Q[40:]}\r".encode()],
                         ids=["bom", "cr"])
def test_odd_but_valid_fasta(fake, tmp_path, body):
    p = tmp_path / "x.fasta"
    p.write_bytes(body)
    assert run(str(p), "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    assert fake.calls == [Q]


def test_bom_on_stdin(fake, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(f"﻿>q\n{Q}\n"))
    assert run("-", "-m", "af3", "-o", str(tmp_path / "o"), "--yes") == 0
    assert fake.calls == [Q]


# Item 12: --host must be an http(s) URL, and only the public server is
# described as run by the ColabFold team.
@pytest.mark.parametrize("host,msg", [
    ("localhost:8080", "for example http://localhost:8080"),
    ("ftp://example.org", "not ftp://"),
    ("file:///etc/passwd", "not file://"),
    ("http://", "no host name"),
])
def test_bad_host_is_refused(tmp_path, capsys, monkeypatch, host, msg):
    import socket

    def no_net(*a, **k):
        raise AssertionError("tried to open a connection")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.setattr(socket, "create_connection", no_net)
    assert run(Q, "-m", "af3", "-o", str(tmp_path / "o"), "--yes", "--host", host) == 1
    err = capsys.readouterr().err
    assert msg in err and "This sends" not in err


def test_host_wording(monkeypatch):
    from scigantic_msa.remote import announce
    monkeypatch.delenv("SCIGANTIC_MSA_COLABFOLD_URL")
    assert "ColabFold team" in MSA().describe_server()
    own = MSA(host="http://10.0.0.5:8080")
    d = own.describe_server()
    assert d.startswith("10.0.0.5:8080") and "ColabFold team" not in d and "you chose" in d
    buf = io.StringIO()
    announce(own.backend, buf)
    assert "public server" not in buf.getvalue()
    buf = io.StringIO()
    announce(MSA().backend, buf)
    assert "public server" in buf.getvalue()


def test_bad_letters_name_a_short_argument_but_never_echo_a_long_one(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert run("mono.fsta", "-m", "boltz", "-o", str(tmp_path / "o"), "--yes") == 1
    assert "No file named 'mono.fsta'" in capsys.readouterr().err
    long_seq = Q[:50] + "1" + Q[50:]
    assert run(long_seq, "-m", "boltz", "-o", str(tmp_path / "o"), "--yes") == 1
    err = capsys.readouterr().err
    assert "not amino acid letters" in err
    assert Q[:20] not in err and Q[-20:] not in err
    assert fake.calls == []
