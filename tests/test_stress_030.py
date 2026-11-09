"""Defects found by the 0.3.0 stress test: downloads, the open data cache,
what the lookups say they sent, and the command line around them.

Offline: every host is a local server on 127.0.0.1, or a fake opener.
"""
import gzip
import io
import json
import os
import socket
import sqlite3
import stat
import threading
import time
import urllib.error
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import LF_PATH, Q_6T33, Q_LITEFOLD, REAL, md5, real_alignment
from scigantic_msa import MSA
from scigantic_msa import http as smhttp
from scigantic_msa.a3m import A3MError
from scigantic_msa.alignment import Alignment
from scigantic_msa.cache import AlignmentCache, FileLock, private_dirs
from scigantic_msa.cli import main
from scigantic_msa.opendata import net, tables
from scigantic_msa.opendata.tables import Tables

ROOT = Path(__file__).resolve().parent.parent
OF3 = "/msas/pdb_msas_completed/6t33_A"
FASTA = "/msas/pdb_msas_completed_representives.fasta"


def cache() -> Path:
    return Path(os.environ["SCIGANTIC_MSA_CACHE"])


def objects(m):
    return {o["role"]: o for o in m.provenance["open"]["objects"]}


def gets(hosts, path):
    return hosts.s.count(lambda r: r.method == "GET" and r.path_only == path)


# ------------------------------------------------------------------ A: cut downloads


def test_a_cut_a3m_is_an_error_and_is_not_cached(hosts):
    hosts.s.faults[f"{OF3}/cfdb_uniref30.a3m"] = [("truncate", 1000)] * net.ATTEMPTS
    m = MSA(server=None, quiet=True).get(Q_6T33)
    cf = objects(m)["cfdb_uniref30"]
    assert "error" in cf and "got" in cf["error"], cf
    assert not any(r.extra["prov"].role == "cfdb_uniref30" for r in m.rows[1:])
    # Not cached: the next run reads the whole file.
    full = MSA(server=None, quiet=True).get(Q_6T33)
    cf = objects(full)["cfdb_uniref30"]
    assert "error" not in cf and cf["rows"] > 5 and cf["truncated"] is False


def test_a_stockholm_without_its_closing_line_is_an_error_and_is_not_cached(hosts):
    path = f"{OF3}/uniref90_hits.sto"
    whole = hosts.s.static[path]
    cut = whole[: whole.rindex(b"\n", 0, len(whole) - 200) + 1]
    assert b"\n//" not in cut
    hosts.s.static[path] = cut  # the server itself says this length
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert "//" in objects(m)["uniref90"].get("error", ""), objects(m)["uniref90"]
    hosts.s.static[path] = whole
    again = MSA(server=None, quiet=True).get(Q_6T33)
    assert "error" not in objects(again)["uniref90"]


def _raw_server(answer):
    """A server that answers each connection with ``answer(method)`` bytes
    and closes it."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            data = c.recv(65536)
            c.sendall(answer(data.split(b" ", 1)[0].decode()))
            c.close()

    threading.Thread(target=loop, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.getsockname()[1]}"


def test_a_cut_table_download_with_no_length_is_refused(tmp_path):
    body = b">1abc_A\nACDEFGHIKLMNPQRSTVWY\n>2abc_A\nACDEFGHIK"  # cut inside a line

    def answer(method):
        return b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n" + (body if method == "GET" else b"")

    srv, url = _raw_server(answer)
    try:
        with pytest.raises(net.OpenDataError, match="middle of a line"):
            tables._fetch_to(url + "/t.fasta", tmp_path / "t.download")
    finally:
        srv.close()


def test_a_cut_chunked_table_download_is_an_open_data_error(tmp_path):
    def answer(method):
        return (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                b"20\r\n>1abc_A\nACDEFGHIKLMNPQRSTVWY\nAC")  # the chunk ends early

    srv, url = _raw_server(answer)
    try:
        with pytest.raises(net.OpenDataError):
            tables._fetch_to(url + "/t.fasta", tmp_path / "t.download")
    finally:
        srv.close()


def test_a_litefold_file_is_read_whole_even_when_the_manifest_says_fewer_bytes(hosts):
    size = len(hosts.s.static[f"{LF_PATH}/a3m/A0A096LP01.a3m"])
    full = MSA(server=None, quiet=True, cache_dir=cache() / "a").get(Q_LITEFOLD)
    hosts.manifest = hosts.manifest.replace(f"\t{size}\t".encode(), f"\t{size // 3}\t".encode())
    m = MSA(server=None, quiet=True, cache_dir=cache() / "b").get(Q_LITEFOLD)
    assert m.depth == full.depth > 10


def test_one_bad_byte_count_in_the_litefold_manifest_skips_only_that_row(hosts):
    hosts.manifest += b"Q9ZZZ1\tY_HUMAN\tY\t20\tMKVLAAGIVGLLLSAPAWAQ\ta3m/Q9ZZZ1.a3m\t\tlots\t0\n"
    m = MSA(server=None, quiet=True).get(Q_LITEFOLD)
    assert {r.extra["prov"].source for r in m.rows[1:]} == {"litefold"}
    t = {t["table"]: t for t in m.provenance["open"]["tables"]}["litefold"]
    assert t["entries"] == 1


# ------------------------------------------------------------------ B: failed lookups


def test_open_only_with_failed_downloads_exits_non_zero_and_says_what_failed(hosts, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    for n in ("cfdb_uniref30.a3m", "uniref90_hits.sto", "mgnify_hits.sto", "uniprot_hits.sto"):
        hosts.s.faults[f"{OF3}/{n}"] = [("status", 503, {}, b"busy")] * net.ATTEMPTS
    rc = main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path / "o"), "--open-only"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "could not be searched" in err and "HTTP 503" in err
    assert "No open alignment holds this sequence" not in err
    assert "came from the cache" not in err
    assert not (tmp_path / "o" / "boltz").exists()


def test_open_only_offline_exits_non_zero(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path / "o"), "--open-only"])
    err = capsys.readouterr().err
    assert rc != 0 and "the openfold3 table could not be read" in err
    assert "No open alignment holds this sequence" not in err


def test_open_only_with_no_remote_and_no_tables_names_the_switch(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    rc = main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path / "o"), "--open-only"])
    err = capsys.readouterr().err
    assert rc != 0 and "SCIGANTIC_MSA_NO_REMOTE is set" in err and "sources --fetch" in err


def test_an_entry_whose_files_are_not_published_says_so(hosts, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    for k in [k for k in hosts.s.static if k.startswith(OF3 + "/")]:
        del hosts.s.static[k]
    rc = main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path / "o"), "--open-only"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "the openfold3 table lists 6t33_A, but none of its files is published" in err
    assert "No open alignment holds this sequence" not in err


def test_a_sequence_in_no_table_is_not_a_failure(hosts):
    m = MSA(server=None, quiet=True).get("MKVLAAGIVGLLLSAPAWAQDTHE")
    assert m.depth == 1 and m.provenance["open"]["failed"] == []


# ------------------------------------------------------------------ C: what was sent


def test_a_failed_uniparc_request_is_still_named_as_sent(hosts):
    hosts.s.faults["/uniparc/search"] = [("status", 503, {}, b"")] * net.ATTEMPTS
    m = MSA(server=None, quiet=True, uniparc=True).get(Q_6T33)
    assert any("MD5" in s for s in m.provenance["open"]["sent"])


def test_requests_for_files_that_are_not_published_are_named_as_sent(hosts):
    for k in [k for k in hosts.s.static if k.startswith(OF3 + "/")]:
        del hosts.s.static[k]
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert any("public files of openfold3 entry 6t33_A" in s for s in m.provenance["open"]["sent"])


def test_uniclust30_checks_sent_before_a_failure_are_named(hosts):
    hosts.uniparc[md5(Q_6T33)] = (["P11111", "P22222"], [])
    hosts.s.prefix_handlers.insert(0, ("/uniclust30/", "HEAD", lambda r: (503, {}, b"")))
    m = MSA(server=None, quiet=True, uniparc=True).get(Q_6T33)
    sent = m.provenance["open"]["sent"]
    assert "1 UniProt accession of the sequence to 127.0.0.1:%d (UniClust30 entries)" % hosts.s.port in sent, sent


def test_one_accession_is_singular(hosts):
    hosts.uniparc[md5(Q_6T33)] = (["P11111"], [])
    m = MSA(server=None, quiet=True, uniparc=True).get(Q_6T33)
    assert any(s.startswith("1 UniProt accession of") for s in m.provenance["open"]["sent"])


def test_the_notice_before_a_run_names_what_the_summary_names(hosts, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only", "--uniparc"]) == 0
    notice = [x for x in capsys.readouterr().err.splitlines() if x.startswith("Also uses")][0]
    for part in ("requests for those tables", "public files of matching entries", "MD5", "UniProt accessions"):
        assert part in notice, notice
    assert "only its MD5" not in notice


def test_the_notice_under_no_remote_does_not_say_anything_is_sent(hosts, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only"]) == 0
    capsys.readouterr()
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only", "--uniparc"]) == 0
    notice = [x for x in capsys.readouterr().err.splitlines() if x.startswith("Also uses")][0]
    assert "nothing is sent" in notice and "MD5" not in notice


# ------------------------------------------------------------------ D: UniParc off for open data only


def test_open_only_does_not_send_the_md5_unless_asked(hosts, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    uniparc = lambda: hosts.s.count(lambda r: r.path_only == "/uniparc/search")  # noqa: E731
    MSA(server=None, quiet=True).get(Q_6T33)
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only", "-q"]) == 0
    assert uniparc() == 0
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only", "--uniparc", "-q"]) == 0
    assert uniparc() == 1
    assert MSA(quiet=True).open.uniparc is True
    assert MSA(server=None, quiet=True).open.uniparc is False


def test_readme_does_not_promise_that_only_the_md5_is_sent():
    text = (ROOT / "README.md").read_text()
    assert "only its MD5" not in text and "only the MD5" not in text
    assert "seq_sha256" in text and "reversed" in text


# ------------------------------------------------------------------ E: a damaged table


def _db(tid="openfold3"):
    return next((cache() / "open" / "tables" / tid).glob("*.sqlite"))


@pytest.mark.parametrize("damage", ["zero", "html", "half"])
def test_a_damaged_table_is_built_again(hosts, damage):
    first = MSA(server=None, quiet=True).get(Q_6T33)
    db = _db()
    if damage == "zero":
        db.write_bytes(b"")
    elif damage == "html":
        db.write_bytes(b"<html><body>portal</body></html>" * 100)
    else:
        db.write_bytes(db.read_bytes()[: db.stat().st_size // 2])
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert m.depth == first.depth
    assert gets(hosts, FASTA) == 2


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_table_is_built_again(hosts):
    first = MSA(server=None, quiet=True).get(Q_6T33)
    os.chmod(_db(), 0)
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert m.depth == first.depth


def test_a_damaged_table_with_the_network_off_is_a_note_not_a_crash(hosts, monkeypatch):
    MSA(server=None, quiet=True).get(Q_6T33)
    _db().write_bytes(b"\0" * 4096)
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert any("openfold3 table" in f for f in m.provenance["open"]["failed"])


def test_sources_reports_a_damaged_table_and_fetch_mends_it(hosts, capsys):
    assert main(["sources", "--fetch"]) == 0
    db = _db()
    db.write_bytes(db.read_bytes()[:2048] + b"\0" * 2048)
    capsys.readouterr()
    main(["sources"])
    assert "table openfold3: damaged in the cache" in capsys.readouterr().out
    assert main(["sources", "--fetch"]) == 0
    assert "table openfold3: 3 sequences" in capsys.readouterr().out


# ------------------------------------------------------------------ F: one flipped bit


def _flip_to_zlib_error(data: bytes) -> bytes:
    for i in range(12, len(data) - 8):
        b = bytearray(data)
        b[i] ^= 0x40
        try:
            gzip.decompress(bytes(b))
        except zlib.error:
            return bytes(b)
        except Exception:  # noqa: BLE001 - another kind of damage; try the next byte
            continue
    pytest.skip("no flip gave a zlib error")


def test_a_flipped_bit_in_a_cached_open_file_is_a_miss(hosts):
    first = MSA(server=None, quiet=True).get(Q_6T33)
    files = list((cache() / "open" / "objects").rglob("*.a3m.gz"))
    for f in files:
        f.write_bytes(_flip_to_zlib_error(f.read_bytes()))
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert m.depth == first.depth
    for f in files:
        assert gzip.decompress(f.read_bytes())  # read again and stored whole


def test_a_flipped_bit_in_a_cached_server_result_is_a_miss(tmp_path):
    a = real_alignment(Q_6T33)
    bad = _flip_to_zlib_error(a.to_bundle())
    with pytest.raises(A3MError):
        Alignment.from_bundle(bad, source="cache")
    c = AlignmentCache(tmp_path)
    p = c.put(a, "colabfold", "k")
    p.write_bytes(bad)
    assert c.get(a.seq_sha256, "colabfold", "k") is None and not p.exists()


# ------------------------------------------------------------------ G: a web page for a table


@pytest.mark.parametrize("ctype", ["text/html; charset=utf-8", "application/octet-stream"])
def test_a_web_page_is_not_kept_as_a_table(hosts, monkeypatch, ctype):
    page = b"<!DOCTYPE html><html><body>Sign in to the Wi-Fi</body></html>\n"
    hosts.s.route("GET", FASTA, lambda r: (200, {"Content-Type": ctype, "ETag": '"e1"'}, page))
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert m.depth == 1 and any("openfold3 table" in f for f in m.provenance["open"]["failed"])
    assert not (cache() / "open" / "tables" / "openfold3" / "meta.json").exists()
    # Even with the ETag check failing, nothing was kept to be reused.
    monkeypatch.setenv("SCIGANTIC_MSA_OPEN_REFRESH_HOURS", "0")
    hosts.s.route("HEAD", FASTA, lambda r: (503, {}, b""))
    hosts.s.route("GET", FASTA, lambda r: (200, {"ETag": '"e1"'}, hosts.fasta))
    assert MSA(server=None, quiet=True).get(Q_6T33).depth > 10


def test_an_empty_manifest_is_not_kept(hosts):
    hosts.manifest = b"acc\tsequence\ta3m_path\ta3m_bytes\n"
    MSA(server=None, quiet=True).get(Q_LITEFOLD)
    assert not (cache() / "open" / "tables" / "litefold" / "meta.json").exists()


def test_a_cached_table_with_too_few_entries_is_not_used(hosts, monkeypatch):
    MSA(server=None, quiet=True).get(Q_6T33)
    monkeypatch.setitem(tables.MIN_ENTRIES, "openfold3", 10)
    t = Tables(cache() / "open" / "tables", cache())
    assert t.info("openfold3") is None


# ------------------------------------------------------------------ H: offline and slow hosts


@pytest.mark.parametrize("reason", [ConnectionRefusedError(61, "refused"), socket.gaierror(8, "no name")])
def test_an_unreachable_host_is_not_tried_again(monkeypatch, reason):
    calls = []

    def open_(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.URLError(reason)

    monkeypatch.setattr(smhttp._opener, "open", open_)
    with pytest.raises(net.Unreachable):
        net.get("http://127.0.0.1:9/x")
    with pytest.raises(net.Unreachable):
        net.head("http://127.0.0.1:9/x")
    assert len(calls) == 2


def test_a_table_that_failed_is_tried_again_after_a_while(monkeypatch, tmp_path):
    t = Tables(tmp_path / "open" / "tables", tmp_path)
    with pytest.raises(net.OpenDataError):
        t.ensure("openfold3")
    with pytest.raises(net.OpenDataError):
        t.ensure("openfold3")
    tried = []
    monkeypatch.setattr(t, "_download", lambda tid: tried.append(tid) or (_ for _ in ()).throw(net.OpenDataError("x")))
    now = time.monotonic()
    monkeypatch.setattr(tables.time, "monotonic", lambda: now + tables.DEAD_SECONDS + 1)
    with pytest.raises(net.OpenDataError):
        t.ensure("openfold3")
    assert tried == ["openfold3"]


def test_a_dripping_body_is_given_up(monkeypatch):
    class Drip(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "4000")
            self.end_headers()
            try:
                for _ in range(4000):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.005)
            except OSError:
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Drip)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(net, "GRACE", 0.3)
    monkeypatch.setattr(net, "MIN_RATE", 100_000)
    t0 = time.monotonic()
    try:
        with pytest.raises(net.TooSlow):
            net.get(f"http://127.0.0.1:{srv.server_address[1]}/x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert time.monotonic() - t0 < 5


def test_a_host_that_never_answers_is_given_up_within_the_deadline(monkeypatch):
    lst = socket.socket()
    lst.bind(("127.0.0.1", 0))
    lst.listen(16)  # connections complete, nothing is ever sent back
    monkeypatch.setattr(net, "DEADLINE", 1.0)
    t0 = time.monotonic()
    try:
        with pytest.raises(net.OpenDataError):
            net.get(f"http://127.0.0.1:{lst.getsockname()[1]}/uniparc", timeout=3)
    finally:
        lst.close()
    assert time.monotonic() - t0 < 5


# ------------------------------------------------------------------ I: odd UniParc answers


@pytest.mark.parametrize("body", [b"[]", b'{"results": {"a": 1}}', b'{"results": [1]}',
                                  b'{"results": [{"uniProtKBAccessions": [5, "P1"], "organisms": [3]}]}'])
def test_an_odd_uniparc_answer_is_no_answer(hosts, body):
    hosts.s.route("GET", "/uniparc/search", lambda r: (200, {}, body))
    m = MSA(server=None, quiet=True, uniparc=True).get(Q_6T33)
    assert m.depth > 10


# ------------------------------------------------------------------ J: one download per object


def test_two_runs_at_once_read_each_file_once(hosts):
    hosts.s.faults[f"{OF3}/cfdb_uniref30.a3m"] = [("sleep", 1.5)]
    MSA(server=None, quiet=True).get("MKVLAAGIVGLLLSAPAWAQDTHE")  # the tables, first
    out, errs = [], []

    def go():
        try:
            out.append(MSA(server=None, quiet=True).get(Q_6T33).depth)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=go) for _ in range(2)]
    for t in ts:
        t.start()
        time.sleep(0.2)
    for t in ts:
        t.join()
    assert not errs and len(set(out)) == 1
    assert gets(hosts, f"{OF3}/cfdb_uniref30.a3m") == 1


# ------------------------------------------------------------------ K: without the cache


def test_without_the_cache_the_tables_are_downloaded_once_per_process(hosts):
    a = MSA(server=None, quiet=True, cache=False)
    b = MSA(server=None, quiet=True, cache=False)
    assert a.get(Q_6T33).depth == b.get(Q_6T33).depth > 10
    assert gets(hosts, FASTA) == 1


def test_without_the_cache_close_removes_what_the_object_read(hosts):
    with MSA(server=None, quiet=True, cache=False, uniparc=True) as m:
        m.get(Q_6T33)
        work = m.open.work
        assert list(work.rglob("*.a3m.gz")) and list(work.rglob("*.json"))
    assert not work.exists()


def test_without_the_cache_the_download_message_does_not_say_once(hosts, capsys):
    MSA(server=None, cache=False).get(Q_6T33)
    err = capsys.readouterr().err
    assert "later runs only check" not in err and "until this process ends" in err


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_without_the_cache_sigterm_removes_the_temporary_folder():
    import subprocess
    import sys

    code = ("import time; from scigantic_msa import MSA; m = MSA(server=None, cache=False, quiet=True); "
            "print(m.open.base, flush=True); time.sleep(60)")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        base = Path(p.stdout.readline().strip())
        assert base.is_dir()
        p.terminate()
        p.wait(timeout=20)
    finally:
        p.kill()
    assert not base.exists()


# ------------------------------------------------------------------ L: threads


def test_one_object_can_be_used_from_several_threads(hosts):
    m = MSA(server=None, quiet=True)
    want = m.get(Q_6T33).depth
    misses = ["MKVLAAGIVGLLLSAPAWAQ" + "ACDEFGHIKLMNPQRSTVWY"[i % 20] * (i % 7 + 1) for i in range(40)]
    out, errs = [], []

    def go(k):
        try:
            for i in range(30):
                if i % 10 == k % 10:
                    out.append(m.get(Q_6T33).depth)
                else:
                    m.get(misses[(i + k) % len(misses)])
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=go, args=(k,)) for k in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs[:3]
    assert len(out) == 16 * 3 and set(out) == {want}


# ------------------------------------------------------------------ M, N: modes


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_private_dirs_never_adds_owner_bits(tmp_path):
    d = tmp_path / "base" / "ro"
    d.mkdir(parents=True)
    os.chmod(d, 0o555)
    try:
        private_dirs(d, tmp_path / "base")
        assert stat.S_IMODE(d.stat().st_mode) == 0o500
    finally:
        os.chmod(d, 0o700)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_partial_table_files_are_private(hosts, tmp_path, monkeypatch):
    old = os.umask(0o022)
    try:
        p = tmp_path / ".fasta.download"
        tables._fetch_to(hosts.s.url + FASTA, p)
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
        modes = []
        real_build = tables._BUILDERS["openfold3"]

        def build(raw, out):
            n = real_build(raw, out)
            modes.append(stat.S_IMODE(out.stat().st_mode))
            return n

        monkeypatch.setitem(tables._BUILDERS, "openfold3", build)
        Tables(cache() / "open" / "tables", cache()).ensure("openfold3")
        assert modes == [0o600]
    finally:
        os.umask(old)


# ------------------------------------------------------------------ O: the command line


def test_a_process_waiting_for_a_table_says_so(hosts):
    said = []
    t = Tables(cache() / "open" / "tables", cache(), say=said.append)
    private_dirs(cache() / "open" / "tables" / "openfold3", cache())
    lock = FileLock(cache() / "open" / "tables" / "openfold3" / "table")
    lock.acquire()
    th = threading.Thread(target=t.ensure, args=("openfold3",))
    th.start()
    time.sleep(0.3)
    lock.release()
    th.join()
    assert any("waiting for another process to download the openfold3 table" in s for s in said)


def test_openproteinset_needs_yes_without_a_terminal(hosts, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only", "--openproteinset"])
    assert rc == 2 and "--yes" in capsys.readouterr().err
    assert hosts.s.requests == []


def test_sources_prints_the_cache_and_refuses_openproteinset_without_fetch(hosts, capsys):
    assert main(["sources"]) == 0
    out = capsys.readouterr().out
    assert f"cache: {cache()}" in out
    line = [x for x in out.splitlines() if x.strip().startswith("UniClust30:")][0]
    assert line.count("not confirmed") == 1
    assert main(["sources", "--openproteinset"]) == 2
    assert "--fetch" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["run", "convert"])
def test_quiet_has_help(cmd, capsys):
    with pytest.raises(SystemExit):
        main([cmd, "--help"])
    assert "print only warnings and errors" in capsys.readouterr().out


def test_an_alignment_given_to_run_points_to_convert(tmp_path, capsys):
    f = tmp_path / "x.a3m"
    f.write_text(f">q\n{Q_6T33}\n>r\n{Q_6T33[:-3]}--a\n")
    assert main(["run", str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "--open-only"]) == 1
    assert "scigantic-msa convert" in capsys.readouterr().err


def test_a_file_with_no_header_and_bad_characters_names_the_format(tmp_path, capsys):
    f = tmp_path / "seq.txt"
    f.write_text("MKVLAAG 1 IVGLLLSAPAWAQ 2\n")
    assert main(["run", str(f), "-m", "boltz", "-o", str(tmp_path / "o"), "--open-only"]) == 1
    assert "no FASTA header" in capsys.readouterr().err


def test_ctrl_c_says_interrupted(hosts, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    def stop(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(MSA, "get", stop)
    assert main(["run", Q_6T33, "-m", "boltz", "-o", str(tmp_path), "--open-only"]) == 130
    assert "Interrupted." in capsys.readouterr().err


def test_readme_is_accurate_about_sizes_and_commands():
    text = (ROOT / "README.md").read_text()
    assert "96 MB" in text and "sources --fetch --openproteinset" in text and "scigantic-msa version" in text
    assert "run` and `convert` print the depth of each chain's\nalignment: rows, an estimated Neff" not in text


# ------------------------------------------------------------------ P: the sdist


def test_the_sdist_carries_what_the_tests_need():
    manifest = ROOT / "MANIFEST.in"
    if not (ROOT / "pyproject.toml").is_file():
        pytest.skip("not a source checkout")
    text = manifest.read_text() if manifest.is_file() else ""
    assert "graft tests" in text or ("conftest.py" in text and "fixtures" in text)


def test_sqlite_error_never_escapes_a_lookup(hosts, monkeypatch):
    MSA(server=None, quiet=True).get(Q_6T33)

    def broken(self, t):
        raise sqlite3.InterfaceError("bad")

    monkeypatch.setattr(Tables, "_conn", broken)
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    m = MSA(server=None, quiet=True).get(Q_6T33)
    assert m.provenance["open"]["failed"]
    assert json.dumps(m.provenance)
    assert REAL.is_dir()
