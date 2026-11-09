"""Regression tests for defects found by a stress test of 0.2.3.

Self-contained and offline. Each test names the defect it covers.
"""
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scigantic_msa import MSA, write_model
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.adapters import Chai
from scigantic_msa.cli import main
from scigantic_msa.remote import announce, remote_disabled
from scigantic_msa.remote.colabfold import DEFAULT_HOST, ColabFoldClient, JobFailed, RemoteResult
from scigantic_msa.tool import RemoteNotAllowed

Q = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # ubiquitin
G = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # GB1


def deep(q, n=30):
    return f">101\n{q}\n" + "".join(f">h{k}\n{q[:k % len(q)]}A{q[k % len(q) + 1:]}\n" for k in range(n))


class FakeBackend:
    name, server_name, dbkey = "colabfold", "fake.invalid", "fakedb"
    host = DEFAULT_HOST  # counts as the public server

    def __init__(self, depth=30):
        self.calls = []
        self.depth = depth

    def search(self, q):
        self.calls.append(q)
        return RemoteResult(rows=parse_a3m(deep(q, self.depth) if self.depth > 1 else f">101\n{q}\n"),
                            provenance={"server": "fake"})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    b = FakeBackend()
    monkeypatch.setattr("scigantic_msa.tool.make_backend", lambda *a, **k: b)
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return b


# 1: the kill switch took only 1, true and yes; "on" sent the sequence.
@pytest.mark.parametrize("value", ["on", "y", "2", "enabled", " ON "])
def test_no_remote_blocks_any_value_but_off(fake, monkeypatch, value):
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", value)
    with pytest.raises(RemoteNotAllowed):
        MSA(cache=False, quiet=True).get(Q, remote=True)
    assert fake.calls == []


@pytest.mark.parametrize("value", ["", "0", "false", "No", "off"])
def test_no_remote_off_values(monkeypatch, value):
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", value)
    assert not remote_disabled()


# 2: Chai wrote through one fixed temp name, so parallel writers into one
# msas folder failed or left a corrupt parquet file.
def test_chai_parallel_writers_to_one_directory(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    m = MSA().from_a3m(deep(Q, 2000))

    def work(_):
        errs = []
        for _ in range(10):
            try:
                Chai.prepare(m, out_dir=tmp_path / "msas", quiet=True)
            except Exception as e:  # noqa: BLE001 - any failure is the defect
                errs.append(type(e).__name__)
        return errs

    with ThreadPoolExecutor(8) as ex:
        errs = [e for r in ex.map(work, range(8)) for e in r]
    assert errs == [], f"{len(errs)} of 80 concurrent writes failed: {sorted(set(errs))}"
    (f,) = (tmp_path / "msas").glob("*/*.pqt")
    assert pq.read_table(f).num_rows == 2001
    assert not list((tmp_path / "msas").glob("**/.*.tmp"))


# 3: the public server was recognised only by its exact URL, so these got
# no warning that anyone can download the result.
@pytest.mark.parametrize("host", ["https://API.colabfold.com", "https://api.colabfold.com:443",
                                  "HTTPS://api.colabfold.com", " https://api.colabfold.com",
                                  "https://api.colabfold.com/", "http://api.colabfold.com"])
def test_public_host_variants_are_the_public_server(host, capsys):
    tool = MSA(host=host, cache=False)
    assert tool.backend.host == DEFAULT_HOST  # also moves http to https
    text = tool.describe_server()
    assert "not the public one" not in text and "anyone who knows the sequence" in text
    announce(tool.backend)
    err = capsys.readouterr().err
    assert not err or "can be read by anyone" in err


def test_own_server_is_still_not_the_public_one():
    assert "not the public one" in MSA(host="https://msa.example.org", cache=False).describe_server()


@pytest.mark.parametrize("host", ["https://my host", "https://x?y=1", "https://x/#a"])
def test_host_with_space_or_query_is_refused(host):
    with pytest.raises(ValueError):
        MSA(host=host, cache=False)


# 4: --batch allowed 100 jobs of 20 chains: 2000 searches on the public
# server from one command.
def test_new_searches_on_the_public_server_are_capped(fake, tmp_path, capsys):
    rnd = random.Random(1)
    f = tmp_path / "big.fasta"
    f.write_text("".join(f">r{i}\n" + ":".join("".join(rnd.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(30))
                                               for _ in range(2)) + "\n" for i in range(51)))
    rc = main(["run", str(f), "--batch", "--model", "colabfold", "--out", str(tmp_path / "o"), "--yes", "-q"])
    assert fake.calls == [] and rc == 1
    assert "102 new jobs" in capsys.readouterr().err


def test_cap_counts_only_new_sequences(fake, tmp_path):
    f = tmp_path / "many.fasta"
    f.write_text("".join(f">r{i}\n{Q}\n" for i in range(100)))
    assert main(["run", str(f), "--batch", "--model", "af3", "--out", str(tmp_path / "o"), "--yes", "-q"]) == 0
    assert fake.calls == [Q]


# 5a: --out was checked before sending, but --out/<model> was not.
@pytest.mark.parametrize("batch", [False, True])
def test_unusable_model_folder_is_found_before_sending(fake, tmp_path, batch):
    out = tmp_path / "o"
    if batch:
        (out / "alpha").mkdir(parents=True)
        (out / "alpha" / "boltz").write_text("not a folder")
        f = tmp_path / "two.fasta"
        f.write_text(f">alpha\n{Q}\n>beta\n{G}\n")
        args = [str(f), "--batch"]
    else:
        out.mkdir()
        (out / "boltz").write_text("not a folder")
        args = [Q]
    assert main(["run", *args, "--model", "boltz", "--out", str(out), "--yes"]) == 1
    assert fake.calls == []


# 5b: names were cut to 100 characters, not bytes: 100 CJK characters are
# 300 bytes, over NAME_MAX on Linux. Windows reserved names get a prefix.
def test_file_names_fit_in_255_bytes(tmp_path):
    m = MSA().from_a3m(deep(Q))
    for model in ("af3", "colabfold", "protenix"):
        files, _ = write_model(model, [m], tmp_path, name="蛋" * 120, quiet=True)
        assert len(Path(files[0]).name.encode()) <= 255


def test_batch_folder_avoids_windows_reserved_names(fake, tmp_path):
    f = tmp_path / "two.fasta"
    f.write_text(f">CON\n{Q}\n>nul.txt\n{G}\n")
    assert main(["run", str(f), "--batch", "--model", "af3", "--out", str(tmp_path / "o"), "--yes", "-q"]) == 0
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == ["_CON", "_nul.txt"]


# 6: the 10-residue error printed the whole typed sequence, twice.
def test_short_chain_error_does_not_echo_long_sequence(fake, tmp_path, capsys):
    rc = main(["run", Q + ":MKV", "--model", "af3", "--out", str(tmp_path / "o"), "--yes"])
    err = capsys.readouterr().err
    assert rc == 1 and fake.calls == []
    assert Q[:20] not in err and "3 residues" in err and "'='" in err


# 7: a corrupt download was a RemoteError, so one bad result stopped the
# rest of a batch with no summary.
def test_corrupt_download_is_a_per_job_failure():
    with pytest.raises(JobFailed):
        ColabFoldClient().parse_result(b"not a tar", Q, "tid")


# 8: --server nvidia without NVIDIA_API_KEY asked to send, then failed.
def test_nvidia_missing_key_is_found_before_confirming(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "cache"))
    rc = main(["run", Q, "--server", "nvidia", "--model", "af3", "--out", str(tmp_path / "o"), "--yes"])
    err = capsys.readouterr().err
    assert rc == 1 and "NVIDIA_API_KEY" in err and "Nothing was sent" in err
    assert "This sends" not in err and "sending the sequence" not in err


# 9: write_model("af3", ..., name="") returned a path it never wrote.
@pytest.mark.parametrize("model", ["af3", "boltz", "colabfold", "openfold3", "protenix"])
def test_write_model_empty_name_returns_written_paths(tmp_path, model):
    m = MSA().from_a3m(deep(Q))
    files, _ = write_model(model, [m], tmp_path, name="", quiet=True)
    assert all(Path(f).exists() for f in files), files
    assert Path(files[0]).stem == "query"


# 10: with --batch -q, depth warnings did not say which job they were about.
def test_quiet_batch_depth_warning_names_the_job(fake, tmp_path, capsys):
    fake.depth = 1
    f = tmp_path / "two.fasta"
    f.write_text(f">alpha\n{Q}\n>beta\n{G}\n")
    main(["run", str(f), "--batch", "--model", "boltz", "--out", str(tmp_path / "o"), "--yes", "-q"])
    warnings = [line for line in capsys.readouterr().err.splitlines() if "depth 1" in line]
    assert len(warnings) == 2
    assert "alpha" in warnings[0] and "beta" in warnings[1], warnings


# Smaller items.
def test_length_summary_with_equal_lengths(fake, tmp_path, capsys):
    rnd = random.Random(2)
    seqs = ["".join(rnd.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(60)) for _ in range(11)]
    f = tmp_path / "eleven.fasta"
    f.write_text("".join(f">r{i}\n{s}\n" for i, s in enumerate(seqs)))
    main(["run", str(f), "--batch", "--model", "af3", "--out", str(tmp_path / "o"), "--yes", "-q"])
    err = capsys.readouterr().err
    assert "length 60 each" in err and "60 to 60" not in err


def test_convert_refuses_an_empty_uniprot_name(tmp_path, capsys):
    a = tmp_path / "a.a3m"
    a.write_text(deep(Q))
    assert main(["convert", str(a), "--uniprot", "", "-m", "af3", "-o", str(tmp_path / "o")]) == 1
    assert "--uniprot" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_attribution_still_written_for_chai(tmp_path):
    pytest.importorskip("pyarrow")
    Chai.prepare(MSA().from_a3m(deep(Q)), out_dir=tmp_path, quiet=True)
    assert json.loads((tmp_path / "scigantic-msa-attribution.json").read_text())["outputs"]


def test_refused_input_leaves_no_empty_folders(tmp_path, capsys):
    # Protenix refuses U before writing anything; the folders went with it.
    a = tmp_path / "u.a3m"
    a.write_text(f">101\n{Q[:30]}U{Q[31:]}\n")
    out = tmp_path / "new" / "o"
    assert main(["convert", str(a), "-m", "protenix", "-o", str(out)]) == 1
    assert "U" in capsys.readouterr().err
    assert not (tmp_path / "new").exists()
