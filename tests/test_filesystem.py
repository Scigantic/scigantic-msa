"""Cache, file modes and output layout on disk.

Found by the 0.2.0 stress test: a read-only cache directory stopped every
search, written MSA files were 0600 whatever the umask, lock files piled up,
two models in one --out overwrote each other, and a job name like '../x'
reached the JSON 'name' that OpenFold3 and Protenix turn into a directory.
"""
from __future__ import annotations

import io
import json
import os
import stat
import sys
import threading
import time

import pytest

from scigantic_msa import MODELS, MSA, write_model
from scigantic_msa.a3m import parse_a3m
from scigantic_msa.adapters import AlphaFold3, Boltz, ColabFold, OpenFold3, Protenix
from scigantic_msa.cache import FileLock
from scigantic_msa.cli import main
from scigantic_msa.remote.colabfold import RemoteResult
from scigantic_msa.writers import ATTRIBUTION_FILE

Q = "MYRNEFTAWYRRMSVVYGIGTWSVLGSLLYYSRTMAKSSVDQKDGSASEVPSELSERPKGFYVETVVTYKEDFVPNTEKILNYWKSWTGGPGTEP"
HIT = "MYRNEFTAWYRRMSVVYGIGTWSVLGSLLYYGRTMAKSSVDQKDGSTSEVPSELSERPKGFYVETVVTYKEDFVPNTEKILNYWKSWTGGPGTEP"
Q2 = "SGSMKTAISLPDETFDRVSRRASELGMSRSEFFTKAAQR"
A3M = f">101\n{Q}\n>UniRef100_A0A2I2ZEU8\n{HIT}\n"

ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


class FakeBackend:
    name, server_name, dbkey = "colabfold", "fake.invalid", "fakedb"

    def __init__(self):
        self.calls = []

    def search(self, q):
        self.calls.append(len(q))
        return RemoteResult(rows=parse_a3m(f">101\n{q}\n"), provenance={"server": "fake"})


@pytest.fixture
def fake(monkeypatch):
    b = FakeBackend()
    monkeypatch.setattr("scigantic_msa.tool.make_backend", lambda *a, **k: b)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return b


# --------------------------------------------------------------------------- cache that cannot be used


@pytest.mark.skipif(ROOT, reason="root ignores permissions")
def test_read_only_cache_dir_does_not_stop_a_search(tmp_path, capsys):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        b = FakeBackend()
        r = MSA(backend=b, cache_dir=ro, quiet=True).get(Q, remote=True)
        assert r.query == Q and b.calls == [len(Q)]
    finally:
        ro.chmod(0o755)
    err = capsys.readouterr().err
    assert str(ro) in err and "--no-cache" in err


def test_cache_path_that_is_a_file_does_not_stop_a_search(tmp_path, capsys):
    f = tmp_path / "not-a-dir"
    f.write_text("x")
    b = FakeBackend()
    tool = MSA(backend=b, cache_dir=f, quiet=True)
    assert tool.cached(Q) is None
    got = tool.get([Q, Q2], remote=True)
    assert [m.query for m in got] == [Q, Q2] and b.calls == [len(Q), len(Q2)]
    err = capsys.readouterr().err
    # Once, not once per chain.
    assert err.count("--no-cache") == 1 and str(f) in err
    assert f.read_text() == "x"


@pytest.mark.skipif(ROOT, reason="root ignores permissions")
def test_cache_that_fails_only_on_write_still_returns_the_result(tmp_path, capsys, monkeypatch):
    b = FakeBackend()
    tool = MSA(backend=b, cache_dir=tmp_path / "c", quiet=True)

    def boom(*a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tool._cache, "put", boom)
    assert tool.get(Q, remote=True).query == Q
    assert "--no-cache" in capsys.readouterr().err


def test_cli_run_with_a_cache_path_that_is_a_file(fake, tmp_path, monkeypatch, capsys):
    f = tmp_path / "cachefile"
    f.write_text("x")
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(f))
    assert main(["run", Q, "--model", "boltz", "--out", str(tmp_path / "o"), "--yes"]) == 0
    assert fake.calls == [len(Q)]
    assert "--no-cache" in capsys.readouterr().err


# --------------------------------------------------------------------------- lock files


def test_lock_file_is_removed_after_a_search(tmp_path):
    b = FakeBackend()
    MSA(backend=b, cache_dir=tmp_path, quiet=True).get(Q, remote=True)
    left = [p for p in tmp_path.rglob("*") if p.name.endswith(".lock")]
    assert left == []


def test_leftover_lock_from_a_crashed_run_does_not_block(tmp_path):
    from scigantic_msa.cache import AlignmentCache
    from scigantic_msa.seqkey import seq_sha256

    c = AlignmentCache(tmp_path)
    lock = c.lock(seq_sha256(Q), "colabfold", "fakedb")
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text("12345\n")  # nobody holds it: the process is gone
    old = time.time() - 7 * 86400
    os.utime(lock.path, (old, old))
    b = FakeBackend()
    MSA(backend=b, cache_dir=tmp_path, quiet=True).get(Q, remote=True)
    assert b.calls == [len(Q)] and not lock.path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="posix flock")
def test_a_waiter_never_holds_a_lock_file_that_was_removed(tmp_path):
    """Removing the lock file on release must not let a waiter lock the old,
    unlinked file while a newcomer locks a new one: both would run."""
    a = FileLock(tmp_path / "k", poll=0.01)
    a.acquire()
    b = FileLock(tmp_path / "k", poll=0.01)
    t = threading.Thread(target=b.acquire)
    t.start()
    time.sleep(0.1)
    a.release()
    t.join(5)
    assert b._fh is not None
    try:
        assert b.path.exists()
        assert os.stat(b.path).st_ino == os.fstat(b._fh.fileno()).st_ino
        c = FileLock(tmp_path / "k", timeout=0.2, poll=0.01)
        with pytest.raises(TimeoutError):
            c.acquire()
    finally:
        b.release()
    assert not b.path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="posix flock")
@pytest.mark.parametrize("newcomer", [True, False])
def test_lock_taken_on_a_file_that_was_just_removed_is_retried(tmp_path, monkeypatch, newcomer):
    """The holder can remove the file between our open and our lock. Then we
    hold a file nobody else can see, so acquire must start again."""
    lk = FileLock(tmp_path / "k", poll=0.01)
    real = FileLock._try_lock
    calls = []

    def racy(fh):
        if not calls:
            os.unlink(lk.path)  # the holder releases and removes it now,
            if newcomer:
                lk.path.write_bytes(b"")  # and a newcomer makes a new one
        calls.append(1)
        return real(fh)

    monkeypatch.setattr(FileLock, "_try_lock", staticmethod(racy))
    lk.acquire()
    try:
        assert len(calls) == 2
        assert os.stat(lk.path).st_ino == os.fstat(lk._fh.fileno()).st_ino
    finally:
        lk.release()


# --------------------------------------------------------------------------- file modes


@pytest.mark.parametrize("umask,want", [(0o022, 0o644), (0o077, 0o600), (0o002, 0o664)])
def test_written_files_follow_the_umask(tmp_path, umask, want):
    old = os.umask(umask)
    try:
        m = MSA().from_a3m(A3M)
        Boltz.prepare(m, out_dir=tmp_path / "b", quiet=True)
        AlphaFold3.prepare(m, out_dir=tmp_path / "a", name="job", quiet=True)
        ColabFold.prepare(m, out_dir=tmp_path / "c", quiet=True)
        OpenFold3.prepare(m, out_dir=tmp_path / "f", quiet=True)
        Protenix.prepare(m, out_dir=tmp_path / "p", quiet=True)
    finally:
        os.umask(old)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) >= 10
    modes = {str(p.relative_to(tmp_path)): oct(stat.S_IMODE(p.stat().st_mode)) for p in files}
    assert set(modes.values()) == {oct(want)}, modes


# --------------------------------------------------------------------------- several models in one --out


def test_every_model_in_one_out_dir_keeps_its_own_files(tmp_path):
    pytest.importorskip("pyarrow")
    chains = [MSA().from_a3m(A3M)]
    written = {}
    for model in MODELS:
        files, nxt = write_model(model, chains, tmp_path, "job", quiet=True)
        written[model] = files
    # Every model's main input still holds that model's input.
    af3 = json.loads(open(written["af3"][0]).read())
    assert af3["dialect"] == "alphafold3"
    of3 = json.loads(open(written["openfold3"][0]).read())
    assert "queries" in of3
    for ch in of3["queries"]["job"]["chains"]:
        assert os.path.isfile(os.path.join(ch["main_msa_file_paths"], "colabfold_main.a3m"))
    ptx = json.loads(open(written["protenix"][0]).read())
    assert isinstance(ptx, list)
    Protenix.check(ptx)
    yaml = open(written["boltz"][0]).read()
    msa_path = yaml.split("msa: ", 1)[1].split()[0].strip('"')
    assert os.path.isfile(msa_path)
    # One attribution file per model, each naming only that model's files.
    sidecars = sorted(tmp_path.rglob(ATTRIBUTION_FILE))
    assert len(sidecars) == len(MODELS)
    # No two models share a main input file.
    mains = [files[0] for files in written.values()]
    assert len(set(mains)) == len(mains)


# --------------------------------------------------------------------------- names that are paths


@pytest.mark.parametrize("name", ["../x", "1abc_A/1-120", "..", "a/../../b", "/etc/passwd"])
def test_names_never_become_paths(tmp_path, name):
    m = MSA().from_a3m(A3M)
    q = OpenFold3.prepare(m, out_dir=tmp_path / "f", name=name, quiet=True)
    jobs = Protenix.prepare(m, out_dir=tmp_path / "p", name=name, quiet=True)
    doc = AlphaFold3.prepare(m, name=name, quiet=True)
    for n in list(q["queries"]) + [jobs[0]["name"], doc["name"]]:
        assert n and "/" not in n and "\\" not in n and not n.startswith("."), n
    out = tmp_path / "out"
    for model in ("af3", "boltz", "colabfold", "openfold3", "protenix"):
        files, _ = write_model(model, [m], out, name, quiet=True)
        for f in files:
            assert os.path.realpath(f).startswith(os.path.realpath(out) + os.sep), f
    assert {p.name for p in tmp_path.iterdir()} <= {"f", "out", "p", "cache"}
    assert not any(p.name.startswith(".") for p in out.rglob("*"))
