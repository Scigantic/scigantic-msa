"""Defects found by the 0.2.7 stress test (new-user round).

Offline: nothing here opens a connection to anything but a stub.
"""
import gzip
import io
import json
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

import scigantic_msa
from scigantic_msa import MSA, write_model
from scigantic_msa import http as smhttp
from scigantic_msa.alignment import Alignment
from scigantic_msa.cli import main
from scigantic_msa.remote import colabfold as cfmod
from scigantic_msa.remote.colabfold import ColabFoldClient, RemoteError

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
README = Path(__file__).resolve().parent.parent / "README.md"


def mut(s, k):
    i = k % len(s)
    return s[:i] + ("A" if s[i] != "A" else "G") + s[i + 1:]


def result_tar(seq):
    """A ColabFold result with MGnify (MGYP) rows in the environmental file."""
    uniref = [f">101\n{seq}"] + [f">UniRef100_X{k}\t{k}\n{mut(seq, k)}" for k in range(6)]
    env = [f">101\n{seq}"] + [f">MGYP{k:012d}\n{mut(seq, k + 20)}" for k in range(4)] \
        + [f">SRR{k}_contig\n{mut(seq, k + 40)}" for k in range(2)]
    files = {"uniref.a3m": "\n".join(uniref) + "\n",
             cfmod.ENV_A3M: "\n".join(env) + "\n",
             "uniref_tax.tsv": "".join(f"{9606 + k}\tUniRef100_X{k}\n" for k in range(6)),
             "msa.sh": "#!/bin/sh\n", "pdb70.m8": "101\t1abc_A\t0.9\n"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for n, t in files.items():
            b = t.encode()
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    return buf.getvalue()


class StubBackend:
    """Answers search() the way ColabFoldClient does, from a local tar."""
    name = "colabfold"
    host = "http://127.0.0.1:9"
    server_name = "127.0.0.1:9"
    dbkey = "stub"

    def search(self, seq):
        return ColabFoldClient(self.host).parse_result(result_tar(seq), seq, "tid")


def no_network(monkeypatch):
    calls = []

    def fake_open(req, timeout=None):
        calls.append(req.full_url)
        raise AssertionError("opened a connection")

    monkeypatch.setattr(smhttp._opener, "open", fake_open)
    return calls


# ------------------------------------------------------------------ 1


def test_no_remote_blocks_colabfoldclient_used_directly(monkeypatch):
    calls = []

    def fake_request(method, url, **kw):
        calls.append(url)
        raise smhttp.HTTPError("stub: no network", 0, url)

    monkeypatch.setattr(smhttp, "request", fake_request)
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    with pytest.raises(scigantic_msa.RemoteNotAllowed):
        scigantic_msa.ColabFoldClient("http://127.0.0.1:9").search(UBQ)
    assert calls == []


def test_no_remote_blocks_nvidiaclient_used_directly(monkeypatch):
    calls = no_network(monkeypatch)
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "yes")
    with pytest.raises(scigantic_msa.RemoteNotAllowed):
        scigantic_msa.NvidiaClient("http://127.0.0.1:9/predict", api_key="k").search(UBQ)
    assert calls == []


def test_no_remote_is_enforced_in_the_http_layer(monkeypatch):
    calls = no_network(monkeypatch)
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    with pytest.raises(scigantic_msa.RemoteNotAllowed):
        smhttp.request("GET", "http://127.0.0.1:9/ticket/x")
    with pytest.raises(scigantic_msa.RemoteNotAllowed):
        ColabFoldClient("http://127.0.0.1:9").status("x")
    assert calls == []


# ------------------------------------------------------------------ 2


def test_cached_result_has_the_fresh_labels_templates_and_provenance(tmp_path):
    m = MSA(backend=StubBackend(), cache_dir=tmp_path / "cache", quiet=True)
    fresh = m.get(UBQ, remote=True)
    cached = m.get(UBQ, remote=True)
    assert fresh.source == "remote" and cached.source == "cache"
    assert [r.db for r in fresh.rows] == [r.db for r in cached.rows]
    assert [r.header for r in fresh.rows] == [r.header for r in cached.rows]
    assert [r.taxid for r in fresh.rows] == [r.taxid for r in cached.rows]
    assert {r.db for r in fresh.rows if r.header.startswith("MGYP")} == {"env"}
    assert fresh.provenance == cached.provenance
    assert fresh.templates == cached.templates is not None


def test_chai_labels_mgnify_env_rows_as_chai_lab_does(tmp_path):
    """chai_lab's own ColabFold path gives every non-UniRef row bfd_uniclust."""
    pq = pytest.importorskip("pyarrow.parquet")
    m = MSA(backend=StubBackend(), cache_dir=tmp_path / "cache", quiet=True)
    for k in range(2):
        files, _ = write_model("chai", [m.get(UBQ, remote=True)], tmp_path / f"o{k}", quiet=True)
        t = pq.read_table(next(Path(files[1]).glob("*.aligned.pqt")))
        assert "mgnify" not in t.column("source_database").to_pylist()


@pytest.mark.parametrize("model", scigantic_msa.MODELS)
def test_cache_round_trip_writes_byte_identical_files(tmp_path, model):
    if model == "chai":
        pytest.importorskip("pyarrow")
    m = MSA(backend=StubBackend(), cache_dir=tmp_path / "cache", quiet=True)
    fresh = m.get([UBQ, mut(UBQ, 7)], remote=True)
    cached = m.get([UBQ, mut(UBQ, 7)], remote=True)
    assert {c.source for c in cached} == {"cache"}
    write_model(model, fresh, tmp_path / "a", name="j", quiet=True)
    write_model(model, cached, tmp_path / "b", name="j", quiet=True)

    def tree(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
                if p.is_file() and p.name != "scigantic-msa-attribution.json"}

    a, b = tree(tmp_path / "a" / model), tree(tmp_path / "b" / model)
    assert a and a.keys() == b.keys()
    for k in a:
        assert a[k].replace(str(tmp_path / "a").encode(), b"") == b[k].replace(str(tmp_path / "b").encode(), b""), k


def test_old_cache_entries_get_the_fresh_colabfold_labels(tmp_path):
    """An entry written by 0.2.7 has no labels in it."""
    m = MSA(backend=StubBackend(), cache_dir=tmp_path / "cache", quiet=True)
    fresh = m.get(UBQ, remote=True)
    p = Path(m.cached(UBQ).cache_path)
    text = gzip.decompress(p.read_bytes()).decode()
    first, _, rest = text.partition("\n")
    meta = json.loads(first[len("#scigantic-msa "):])
    del meta["dbs"]
    old = "#scigantic-msa " + json.dumps(meta) + "\n" + rest
    back = Alignment.from_bundle(gzip.compress(old.encode()), source="cache")
    assert [r.db for r in back.rows] == [r.db for r in fresh.rows]


# ------------------------------------------------------------------ 3


def test_attribution_file_has_no_local_path_or_ticket(tmp_path):
    cache = tmp_path / "home_of_alice" / "cache"
    m = MSA(backend=StubBackend(), cache_dir=cache, quiet=True)
    m.get(UBQ, remote=True)
    r = m.get(UBQ, remote=True)
    assert r.cache_path and "home_of_alice" in r.cache_path
    assert "cache_file" not in r.provenance
    write_model("boltz", [r], tmp_path / "out", quiet=True)
    text = (tmp_path / "out" / "boltz" / "msas" / "scigantic-msa-attribution.json").read_text()
    assert "home_of_alice" not in text
    assert '"ticket"' not in text and "tid" not in json.loads(text)["outputs"][0]["chains"][0]["provenance"]
    prov = json.loads(text)["outputs"][0]["chains"][0]["provenance"]
    assert prov["server"] == "http://127.0.0.1:9" and prov["mode"] == "env-taxonomy" and "computed_at" in prov


# ------------------------------------------------------------------ 4


@pytest.mark.parametrize("url", ["http://alice:s3cret@127.0.0.1:9", "https://s3cret@x.org/", "alice:s3cret@127.0.0.1"])
def test_host_with_credentials_is_refused_without_echo(url):
    with pytest.raises(ValueError) as e:
        MSA(host=url)
    assert "s3cret" not in str(e.value)
    with pytest.raises(ValueError) as e:
        ColabFoldClient(url)
    assert "s3cret" not in str(e.value)
    with pytest.raises(ValueError) as e:
        scigantic_msa.NvidiaClient(url)
    assert "s3cret" not in str(e.value)


def test_host_env_with_credentials_is_refused(monkeypatch):
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", "http://alice:s3cret@127.0.0.1:9")
    with pytest.raises(ValueError) as e:
        MSA()
    assert "s3cret" not in str(e.value)
    with pytest.raises(ValueError):
        ColabFoldClient()


def test_cli_host_with_credentials_is_refused(tmp_path, capsys, monkeypatch):
    calls = no_network(monkeypatch)
    rc = main(["run", UBQ, "-m", "af3", "-o", str(tmp_path / "o"), "--yes",
               "--host", "http://alice:s3cret@127.0.0.1:9"])
    captured = capsys.readouterr()
    assert rc != 0 and calls == []
    assert "s3cret" not in captured.err + captured.out
    assert "password" in captured.err


# ------------------------------------------------------------------ 5


def test_nvidia_without_key_does_not_claim_sending(monkeypatch, capsys):
    from scigantic_msa import remote
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    calls = no_network(monkeypatch)
    with pytest.raises(RemoteError):
        MSA("nvidia", cache=False).get(UBQ, remote=True)
    assert "sending" not in capsys.readouterr().err
    assert not remote._announced and calls == []


# ------------------------------------------------------------------ 6


def short_hint(tmp_path, capsys):
    rc = main(["run", "MKVLAAG", "-m", "af3", "-o", str(tmp_path / "o"), "--yes"])
    assert rc != 0
    return capsys.readouterr().err.split("write it as ", 1)[1].strip().rstrip(".")


def test_short_peptide_hint_is_quoted(tmp_path, capsys):
    assert short_hint(tmp_path, capsys) == "'=MKVLAAG'"
    assert "'=MKVLAAG'" in README.read_text()


@pytest.mark.parametrize("shell", ["zsh", "bash", "fish"])
def test_short_peptide_hint_survives_the_shell(tmp_path, capsys, shell):
    if shutil.which(shell) is None:
        pytest.skip(f"needs {shell}")
    hint = short_hint(tmp_path, capsys)
    args = [shell, "-fc"] if shell == "zsh" else [shell, "-c"]
    p = subprocess.run(args + [f"printf %s {hint}"], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout == "=MKVLAAG", p.stderr


# ------------------------------------------------------------------ 7


def test_readme_names_every_cache_location():
    text = README.read_text()
    assert "XDG_CACHE_HOME" in text and "LOCALAPPDATA" in text and "scigantic-msa cache" in text


def test_cache_command_prints_the_directory_in_use(tmp_path, monkeypatch, capsys):
    from scigantic_msa.cache import cache_dir
    monkeypatch.delenv("SCIGANTIC_MSA_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert main(["cache"]) == 0
    assert capsys.readouterr().out.strip() == str(cache_dir())


# ------------------------------------------------------------------ 8


def loose_dirs(cache):
    return [str(d.relative_to(cache)) for d in [cache, *cache.rglob("*")]
            if d.is_dir() and stat.S_IMODE(d.stat().st_mode) & 0o077]


def test_cache_directories_are_private(tmp_path):
    cache = tmp_path / "cache"
    MSA(backend=StubBackend(), cache_dir=cache, quiet=True).get(UBQ, remote=True)
    assert loose_dirs(cache) == []


def test_existing_loose_cache_directories_are_tightened(tmp_path):
    """Directories from 0.2.7 are 0755; a write under them tightens them. A
    cache directory the user made keeps its mode."""
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o755)
    cache.chmod(0o755)
    m = MSA(backend=StubBackend(), cache_dir=cache, quiet=True)
    m.get(UBQ, remote=True)
    for d in [p for p in cache.rglob("*") if p.is_dir()]:
        d.chmod(0o755)
    Path(m.cached(UBQ).cache_path).unlink()
    MSA(backend=StubBackend(), cache_dir=cache, quiet=True).get(UBQ, remote=True)
    assert loose_dirs(cache) == ["."]


# ------------------------------------------------------------------ 9


def test_alphafold3_prepare_takes_inline_false(tmp_path):
    from scigantic_msa.adapters import AlphaFold3, PrepareError
    r = MSA().from_a3m(f">101\n{UBQ}\n>h\n{mut(UBQ, 3)}\n")
    doc = AlphaFold3.prepare(r, name="j", out_dir=tmp_path, inline=False, quiet=True)
    prot = doc["sequences"][0]["protein"]
    assert "unpairedMsa" not in prot
    f = tmp_path / prot["unpairedMsaPath"]
    assert f.name.startswith(r.seq_sha256) and f.name.endswith(".unpaired.a3m")
    assert f.read_text().startswith(f">101\n{UBQ}\n")
    with pytest.raises(PrepareError):
        AlphaFold3.prepare(r, name="j", inline=False, quiet=True)


# ------------------------------------------------------------------ minor


def test_cache_path_is_set_on_cache_hits(tmp_path):
    m = MSA(backend=StubBackend(), cache_dir=tmp_path / "cache", quiet=True)
    fresh = m.get(UBQ, remote=True)
    cached = m.get(UBQ, remote=True)
    assert fresh.cache_path and fresh.cache_path == cached.cache_path


def test_unusable_cache_warns_once_per_process(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    for _ in range(2):
        MSA(backend=StubBackend(), cache_dir=blocker / "c").get(UBQ, remote=True)
    assert capsys.readouterr().err.count("cannot use the cache") == 1


def test_pyarrow_hint_names_the_uv_tool_form():
    from scigantic_msa.writers import PYARROW_HINT
    assert "uv tool install 'scigantic-msa[chai]'" in PYARROW_HINT
