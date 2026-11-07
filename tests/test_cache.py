import multiprocessing as mp
import os
import sys
import time

import pytest

from conftest import Q_LIVE, key
from scigantic_msa.a3m import Row
from scigantic_msa.cache import AlignmentCache, FileLock, atomic_write
from scigantic_msa.alignment import Alignment


def msa(q=Q_LIVE):
    return Alignment(query=q, rows=[Row("101", q, "query"), Row("h", q, "uniref", "1")], source="remote",
               provenance={"backend": "colabfold"})


def test_put_get_and_corrupt_entries(tmp_path):
    c = AlignmentCache(tmp_path)
    p = c.put(msa(), "colabfold", "k1")
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"
    got = c.get(key(Q_LIVE))
    assert got.source == "cache" and got.rows[1].taxid == "1"
    p.write_bytes(b"\x1f\x8bjunk")
    assert c.get(key(Q_LIVE)) is None
    assert not p.exists()  # removed so it cannot shadow a good entry


def test_atomic_write_leaves_no_temp_files(tmp_path):
    p = tmp_path / "d" / "f"
    atomic_write(p, b"abc")
    atomic_write(p, b"def")
    assert p.read_bytes() == b"def"
    assert [x.name for x in p.parent.iterdir()] == ["f"]


def _hold(path, seconds, q):
    with FileLock(path):
        q.put(("got", time.time()))
        time.sleep(seconds)
    q.put(("released", time.time()))


@pytest.mark.skipif(sys.platform == "win32", reason="posix timing test")
def test_lock_excludes_other_processes(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_hold, args=(tmp_path / "x", 1.0, q))
    p.start()
    assert q.get(timeout=20)[0] == "got"
    t0 = time.time()
    with FileLock(tmp_path / "x", poll=0.05):
        waited = time.time() - t0
    p.join(10)
    assert waited > 0.5


def _worker(env, q):
    os.environ.update(env)
    from scigantic_msa import get_msa

    m = get_msa(Q_LIVE, remote="colabfold", quiet=True)
    q.put(m.source)


@pytest.mark.skipif(sys.platform == "win32", reason="posix")
def test_parallel_jobs_submit_once(server, tmp_path, monkeypatch):
    from test_colabfold import FakeColabFold

    fake = FakeColabFold(server, ["COMPLETE"])
    # Slow download so the second process has to wait on the lock.
    tar = fake.tar
    server.route_prefix("GET", "/result/download/", lambda r: (time.sleep(1.0), (200, {}, tar))[1])
    env = {k: v for k, v in os.environ.items() if k.startswith("SCIGANTIC_")}
    env.update({"SCIGANTIC_MSA_COLABFOLD_URL": server.url, "SCIGANTIC_MSA_INDEX_URL": "http://127.0.0.1:9/"})
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(env, q)) for _ in range(2)]
    for p in ps:
        p.start()
    got = sorted(q.get(timeout=60) for _ in ps)
    for p in ps:
        p.join(10)
    assert got == ["cache", "remote"]
    assert server.count(lambda r: r.path.startswith("/result/download/")) == 1
