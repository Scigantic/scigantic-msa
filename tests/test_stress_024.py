"""Regression tests for defects found by a stress test of 0.2.4.

Self-contained and offline. Each test names the defect it covers.
"""
import io
import json
import sys
import unicodedata

import pytest

from scigantic_msa import MSA, write_model
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.adapters import _safe_name
from scigantic_msa.cli import _jobs, main
from scigantic_msa.remote.colabfold import DEFAULT_HOST, ColabFoldClient, RemoteResult, public_host

Q = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # ubiquitin
G = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # GB1


def mut(q, k):
    return q[:k] + ("A" if q[k] != "A" else "G") + q[k + 1:]


class FakeBackend:
    name, server_name, dbkey = "colabfold", "fake.invalid", "fakedb"
    host = "https://fake.invalid"

    def __init__(self):
        self.calls = []

    def search(self, q):
        self.calls.append(q)
        text = f">101\n{q}\n" + "".join(f">h{k}\n{mut(q, k)}\n" for k in range(1, 30))
        return RemoteResult(rows=parse_a3m(text), provenance={"server": "fake"})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    b = FakeBackend()
    monkeypatch.setattr("scigantic_msa.tool.make_backend", lambda *a, **k: b)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return b


# 3a: Protenix refused U, B, Z, J, O only after run had sent the sequence.
def test_protenix_letter_refusal_comes_before_sending(fake, tmp_path):
    rc = main(["run", Q[:-1] + "U", "-m", "protenix", "-o", str(tmp_path / "o"), "--yes"])
    assert rc == 1
    assert fake.calls == [], "the sequence was sent, then Protenix refused it"


def test_protenix_letter_refusal_in_a_batch_sends_only_the_good_jobs(fake, tmp_path, capsys):
    f = tmp_path / "b.fasta"
    f.write_text(f">good\n{Q}\n>sec\n{G[:-1]}U\n")
    rc = main(["run", str(f), "--batch", "-m", "protenix", "-o", str(tmp_path / "o"), "--yes", "-q"])
    err = capsys.readouterr().err
    assert rc == 1
    assert fake.calls == [Q], "the refused job's sequence was sent"
    assert (tmp_path / "o" / "good" / "protenix").is_dir()
    assert not (tmp_path / "o" / "sec").exists()
    last = err.strip().splitlines()[-1]
    assert "sec" in last and "retry" not in last, last


def test_batch_of_only_refused_jobs_sends_nothing(fake, tmp_path, capsys):
    f = tmp_path / "b.fasta"
    f.write_text(f">a\n{Q[:-1]}B\n>b\n{G[:-1]}Z\n")
    rc = main(["run", str(f), "--batch", "-m", "protenix", "-o", str(tmp_path / "o"), "--yes"])
    err = capsys.readouterr().err
    assert rc == 1 and fake.calls == []
    assert "This sends" not in err and "retry" not in err


# 3b: with plain `pip install scigantic-msa`, run --model chai sent the
# sequence, then failed on the missing pyarrow.
@pytest.mark.parametrize("batch", [False, True])
def test_chai_without_pyarrow_stops_before_sending(fake, tmp_path, monkeypatch, capsys, batch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    if batch:
        f = tmp_path / "b.fasta"
        f.write_text(f">a\n{Q}\n>b\n{G}\n")
        args = [str(f), "--batch"]
    else:
        args = [Q]
    rc = main(["run", *args, "-m", "chai", "-o", str(tmp_path / "o"), "--yes"])
    err = capsys.readouterr().err
    assert rc == 1
    assert fake.calls == [], "the sequence was sent, then the Chai writer needed pyarrow"
    assert "pyarrow" in err and "retry" not in err


# 4: a public-server URL with a dot path segment reaches api.colabfold.com
# but was called "a server you chose": no warning, no cap, its own cache.
@pytest.mark.parametrize("host", ["https://api.colabfold.com/.", "https://api.colabfold.com/./",
                                  "https://api.colabfold.com/a/..", "https://api.colabfold.com/%2e/"])
def test_dot_path_public_url_is_the_public_server(host, monkeypatch):
    assert public_host(host)
    assert "not the public one" not in MSA(host=host, cache=False).describe_server()
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", host)
    assert ColabFoldClient().host == DEFAULT_HOST
    assert "not the public one" not in MSA(cache=False).describe_server()


def test_other_paths_are_still_not_the_public_server():
    assert not public_host("https://api.colabfold.com/proxy")
    assert not public_host("https://api.colabfold.com/./proxy")


# 5: a non-ASCII host passed --host, then the request failed with a
# latin-1 codec error after "sending the sequence to ...".
@pytest.mark.parametrize("host", ["https://ａｐｉ.colabfold.com", "https://bücher.example/", "https://x.example/é"])
def test_non_ascii_host_is_refused(host, monkeypatch):
    with pytest.raises(ValueError, match="ASCII"):
        MSA(host=host, cache=False)
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", host)
    with pytest.raises(ValueError, match="ASCII"):
        MSA(cache=False)
    with pytest.raises(ValueError, match="ASCII"):
        ColabFoldClient()


def test_non_ascii_host_error_suggests_the_ascii_spelling():
    with pytest.raises(ValueError, match="https://api.colabfold.com"):
        MSA(host="https://ａｐｉ.colabfold.com", cache=False)


def test_fullwidth_public_host_is_recognised():
    assert public_host("https://ａｐｉ．colabfold．com")


def test_non_ascii_host_from_the_cli_is_refused_before_sending(fake, tmp_path, capsys):
    rc = main(["run", Q, "-m", "af3", "-o", str(tmp_path / "o"), "--yes", "--host", "https://ａｐｉ.colabfold.com"])
    err = capsys.readouterr().err
    assert rc == 1 and "ASCII" in err and "latin-1" not in err


# 6: an attribution file that is valid JSON with an odd entry crashed the
# write with AttributeError, after the model's files were written.
@pytest.mark.parametrize("entry", [
    {"chains": ["x"], "files": {"k": "v"}},
    {"chains": [{"attribution": "x"}], "files": ["a"]},
    {"chains": [{"attribution": [{"id": "keep"}, 3]}], "files": {"other.json": 1}},
    {"chains": "x", "files": None},
])
def test_odd_attribution_entry_does_not_crash_the_write(tmp_path, entry):
    m = MSA().from_a3m(f">q\n{Q}\n>h\n{mut(Q, 3)}\n")
    d = tmp_path / "af3"
    d.mkdir()
    side = d / "scigantic-msa-attribution.json"
    side.write_text(json.dumps({"format": 2, "outputs": [entry]}))
    write_model("af3", [m], tmp_path, name="j", quiet=True)
    data = json.loads(side.read_text())
    assert data["outputs"][-1]["files"] == {"af3_json": "j.json"}
    assert all(isinstance(a, dict) for a in data["sources"])


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '"x"', '{"format": 9}'])
def test_unreadable_attribution_file_is_replaced_with_a_warning(tmp_path, capsys, text):
    m = MSA().from_a3m(f">q\n{Q}\n>h\n{mut(Q, 3)}\n")
    d = tmp_path / "af3"
    d.mkdir()
    side = d / "scigantic-msa-attribution.json"
    side.write_text(text)
    write_model("af3", [m], tmp_path, name="j", quiet=True)
    err = capsys.readouterr().err
    assert err.count("scigantic-msa-attribution.json") == 1 and "warning" in err, err
    assert len(json.loads(side.read_text())["outputs"]) == 1
    write_model("af3", [m], tmp_path, name="k", quiet=True)
    assert capsys.readouterr().err == ""


# 7: batch job folders were deduplicated with str.lower(), but APFS also
# folds normalization forms and "SS" onto "ß", so two records shared one
# folder and the second overwrote the first.
def test_batch_job_folders_are_distinct_on_insensitive_filesystems(tmp_path):
    nfc = unicodedata.normalize("NFC", "한")
    nfd = unicodedata.normalize("NFD", "한")
    f = tmp_path / "n.fasta"
    f.write_text(f">{nfc}\n{Q}\n>{nfd}\n{G}\n>ß\n{Q}\n>SS\n{G}\n>Ǆ\n{Q}\n>ǅ\n{G}\n", encoding="utf-8")
    names = [n for n, _ in _jobs(str(f), False, True, None)]
    folded = {unicodedata.normalize("NFD", n).casefold() for n in names}
    assert len(folded) == len(names), names


# 8: Windows also reserves COM and LPT with a superscript digit, and a
# name cut to its byte limit could end in a dot.
@pytest.mark.parametrize("name", ["COM¹", "LPT³.txt", "com²", "COM0", "lpt0.a3m"])
def test_superscript_reserved_names_get_a_prefix(name):
    assert _safe_name(name).startswith("_")


def test_cut_name_does_not_end_in_a_dot():
    s = _safe_name("a" * 199 + "." + "b" * 50)
    assert len(s.encode()) <= 200 and not s.endswith((".", " "))
    s = _safe_name("蛋" * 66 + "." * 10 + "x")
    assert not s.endswith((".", " "))
