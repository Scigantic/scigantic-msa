"""Regression tests for defects found by a stress test of 0.2.0.

Self-contained and offline. Each test names the defect it covers.
"""
import io
import os
import stat

import pytest

from scigantic_msa import MSA
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.adapters import AlphaFold3, Boltz, Protenix, protenix_species
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
        self.calls.append(len(q))
        return RemoteResult(rows=parse_a3m(f">101\n{q}\n"), provenance={"server": "fake"})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    b = FakeBackend()
    monkeypatch.setattr("scigantic_msa.tool.make_backend", lambda *a, **k: b)
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return b


# D1: a mistyped file name made only of letters is sent to the server as a sequence.
def test_letters_only_missing_file_name_is_not_sent(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "query.fasta").write_text(f">q\n{Q}\n")
    rc = main(["run", "query", "--model", "af3", "--out", str(tmp_path / "o"), "--yes"])  # meant query.fasta
    assert fake.calls == [], "the word 'query' was sent to the server as a 5-residue protein"
    assert rc != 0


def test_directory_argument_is_not_sent(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chains").mkdir()
    rc = main(["run", "chains", "--model", "af3", "--out", str(tmp_path / "o"), "--yes"])
    assert fake.calls == [], "a directory name was sent to the server as a sequence"
    assert rc != 0


# D2: an unwritable cache directory makes get()/run fail before any search,
# although caching is optional and put() already tolerates write errors.
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_unwritable_cache_falls_back_to_no_cache(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        b = FakeBackend()
        r = MSA(backend=b, cache_dir=ro, quiet=True).get(Q, remote=True)
        assert r.query == Q and b.calls == [len(Q)]
    finally:
        ro.chmod(0o755)


# D3: every MSA file the adapters write is mode 0600 whatever the umask, so a
# predictor running as another user (container, cluster service account) cannot read it.
def test_written_msa_files_follow_umask(tmp_path):
    old = os.umask(0o022)
    try:
        m = MSA().from_a3m(A3M)
        Boltz.prepare(m, out_dir=tmp_path / "b", quiet=True)
        AlphaFold3.prepare(m, out_dir=tmp_path / "a", name="job", quiet=True)
        files = list((tmp_path / "b").glob("*.a3m")) + [tmp_path / "a" / "job.json"]
        modes = {f.name: oct(stat.S_IMODE(f.stat().st_mode)) for f in files}
        assert all(v == "0o644" for v in modes.values()), modes
    finally:
        os.umask(old)


# D4: a UTF-8 byte order mark (Windows editors) makes a valid A3M, Stockholm
# or FASTA unreadable, with a misleading message.
def test_a3m_with_bom_is_read(tmp_path):
    p = tmp_path / "bom.a3m"
    p.write_bytes(b"\xef\xbb\xbf" + A3M.encode())
    assert MSA().from_a3m(p).depth == 2


def test_fasta_with_bom_is_read(fake, tmp_path):
    p = tmp_path / "bom.fasta"
    p.write_bytes(b"\xef\xbb\xbf>q\n" + Q.encode() + b"\n")
    assert main(["run", str(p), "--model", "af3", "--out", str(tmp_path / "o"), "--yes"]) == 0


# D5: Protenix reads the species of a full UniRef header from the RepID suffix
# (e.g. 9BETA, a code shared by many species), not from the TaxID the library
# paired on, so pairing.a3m can pair different species.
def test_protenix_pairing_headers_carry_the_taxid(tmp_path):
    def chain(q, hits):
        text = f">q\n{q}\n" + "".join(f">{h}\n{s}\n" for h, s in hits)
        return MSA().from_a3m(text)

    qa, qb = "MKVLAAGIVA", "GHSTWQERLL"
    a = chain(qa, [("UniRef100_P1 x n=1 Tax=Sp one TaxID=111 RepID=P1_9BETA", "MKVLAAGIVG"),
                   ("UniRef100_P2 x n=1 Tax=Sp two TaxID=222 RepID=P2_9BETA", "MKVLSAGIVA")])
    b = chain(qb, [("UniRef100_P3 x n=1 Tax=Sp two TaxID=222 RepID=P3_9BETA", "GHSTWQERLI"),
                   ("UniRef100_P4 x n=1 Tax=Sp three TaxID=333 RepID=P4_9BETA", "GHSTWAERLL")])
    jobs = Protenix.prepare([a, b], out_dir=tmp_path, quiet=True)
    for pc in (s["proteinChain"] for s in jobs[0]["sequences"]):
        rows = parse_a3m(open(pc["pairedMsaPath"]).read())[1:]
        species = [protenix_species(r.header) for r in rows]
        # Only taxid 222 is shared; Protenix must not see one species for all four rows.
        assert len(set(species)) == len(species), species


# D6: with SCIGANTIC_MSA_NO_REMOTE=1 the CLI still prints "This sends 1 sequence ... to api.colabfold.com".
def test_no_remote_cli_does_not_claim_it_sends(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    assert main(["run", Q, "--model", "af3", "--out", str(tmp_path / "o"), "--yes"]) == 1
    assert "This sends" not in capsys.readouterr().err


# D7: convert of a directory says "<dir> does not exist".
def test_convert_directory_message(tmp_path, capsys):
    (tmp_path / "d").mkdir()
    assert main(["convert", str(tmp_path / "d"), "--model", "af3", "--out", str(tmp_path / "o")]) == 1
    assert "does not exist" not in capsys.readouterr().err


# D8: merging into an AF3 input whose protein sequence is lowercase keeps the
# lowercase sequence next to an uppercase MSA query row.
def test_af3_merge_lowercase_sequence(tmp_path):
    m = MSA().from_a3m(A3M)
    inp = {"dialect": "alphafold3", "version": 2, "name": "x", "modelSeeds": [1],
           "sequences": [{"protein": {"id": "A", "sequence": Q.lower()}}]}
    try:
        doc = AlphaFold3.prepare(m, input=inp)
    except ValueError:
        return  # refusing is also fine
    p = doc["sequences"][0]["protein"]
    assert p["sequence"] == p["unpairedMsa"].split("\n")[1]


# D9: convert of a query-only alignment (depth 1) succeeds without any warning.
def test_convert_query_only_says_depth_1(tmp_path, capsys):
    f = tmp_path / "q.a3m"
    f.write_text(f">q\n{Q}\n")
    assert main(["convert", str(f), "--model", "boltz", "--out", str(tmp_path / "o")]) == 0
    assert "depth 1" in capsys.readouterr().err
