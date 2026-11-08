"""Defects found by the 0.2.8 stress test.

Offline: nothing here reaches anything but 127.0.0.1, and name lookups are
trapped where a test needs to see them.
"""
import os
import socket
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scigantic_msa import MSA
from scigantic_msa import cache as cachemod
from scigantic_msa import http as smhttp
from scigantic_msa.cache import cache_dir
from scigantic_msa.cli import main
from scigantic_msa.remote.colabfold import ColabFoldClient
from scigantic_msa.remote.nvidia import NvidiaClient
from scigantic_msa.tool import check_host

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
README = Path(__file__).resolve().parent.parent / "README.md"


def _server(status=503, body=b""):
    hits = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _answer(self):
            hits.append(self.path)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _answer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


@pytest.fixture
def trapped_dns(monkeypatch):
    looked_up = []

    def gai(name, *a, **k):
        looked_up.append(name)
        raise socket.gaierror(8, "trapped by test")

    monkeypatch.setattr(socket, "getaddrinfo", gai)
    return looked_up


# 1: http://u:PW\@host passed the check (cut at '\'), while urllib read
# u:PW\@host as the host part: the password was printed and looked up.
@pytest.mark.parametrize("host", ["http://u:SECRETPW\\@127.0.0.1:18765", "http://SECRETPW\\@127.0.0.1"])
def test_backslash_userinfo_is_refused_and_never_resolved(host, tmp_path, monkeypatch, capsys, trapped_dns):
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "c"))
    monkeypatch.delenv("SCIGANTIC_MSA_NO_REMOTE", raising=False)
    rc = main(["run", "=" + UBQ, "-m", "colabfold", "-o", str(tmp_path / "o"), "--yes", "--host", host])
    err = capsys.readouterr().err
    assert rc != 0
    assert trapped_dns == []
    assert "SECRETPW" not in err, err
    assert "backslash" in err


@pytest.mark.parametrize("url", ["http://host\\path", "http:\\\\host", "https://api.colabfold.com\\"])
def test_any_backslash_is_refused(url):
    with pytest.raises(ValueError, match="backslash"):
        check_host(url)
    with pytest.raises(ValueError, match="backslash"):
        ColabFoldClient(url)


def test_request_refuses_userinfo_before_any_lookup(trapped_dns):
    """The HTTP layer is the last line: a URL with userinfo that reached it
    by any path is refused before the name is looked up."""
    for url in ("http://u:SECRETPW@127.0.0.1/x", "http://u:SECRETPW\\@127.0.0.1/x"):
        with pytest.raises(ValueError) as ei:
            smhttp.request("GET", url)
        assert "SECRETPW" not in str(ei.value)
    assert trapped_dns == []


def test_short_never_prints_userinfo():
    assert "SECRETPW" not in smhttp._short("http://u:SECRETPW@host/p")
    assert "SECRETPW" not in smhttp._short("//u:SECRETPW@host/p")


# 2: refused URLs were repeated in the error, password and all.
SCHEMELESS = ["//u:SECRETPW@127.0.0.1", "http:/u:SECRETPW@host", "http:\\\\u:SECRETPW@host",
              "http:///u:SECRETPW@host", "u:SECRETPW@host", "http://u:SECRETPW＠host"]


@pytest.mark.parametrize("host", SCHEMELESS)
def test_refused_host_error_does_not_print_password(host):
    with pytest.raises(ValueError) as ei:
        check_host(host)
    assert "SECRETPW" not in str(ei.value), str(ei.value)


@pytest.mark.parametrize("host", SCHEMELESS)
def test_clients_refuse_userinfo_without_printing_it(host):
    for make in (lambda: ColabFoldClient(host), lambda: NvidiaClient(host, api_key="k"),
                 lambda: NvidiaClient(api_key="k", status_url=host + "/{reqid}")):
        with pytest.raises(ValueError) as ei:
            make()
        assert "SECRETPW" not in str(ei.value), str(ei.value)


@pytest.mark.parametrize("env", ["SCIGANTIC_MSA_COLABFOLD_URL", "SCIGANTIC_MSA_NVIDIA_URL"])
@pytest.mark.parametrize("host", SCHEMELESS)
def test_env_url_with_userinfo_is_refused_quietly(env, host, tmp_path, monkeypatch, capsys, trapped_dns):
    monkeypatch.setenv(env, host)
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "c"))
    monkeypatch.delenv("SCIGANTIC_MSA_NO_REMOTE", raising=False)
    server = "nvidia" if "NVIDIA" in env else "colabfold"
    rc = main(["run", "=" + UBQ, "-m", "boltz", "-o", str(tmp_path / "o"), "--yes", "--server", server])
    err = capsys.readouterr().err
    assert rc != 0
    assert "SECRETPW" not in err, err
    assert trapped_dns == []
    # The client read from the environment names the variable.
    make = ColabFoldClient if server == "colabfold" else NvidiaClient
    with pytest.raises(ValueError) as ei:
        make()
    assert env in str(ei.value) and "SECRETPW" not in str(ei.value)


def test_colabfoldclient_refuses_schemeless_userinfo():
    with pytest.raises(ValueError):
        ColabFoldClient("//u:SECRETPW@127.0.0.1")


# 3: the kill switch was read once, before the retry loop.
def test_no_remote_stops_retries_already_under_way(monkeypatch):
    srv, hits = _server(503)
    monkeypatch.delenv("SCIGANTIC_MSA_NO_REMOTE", raising=False)
    monkeypatch.setattr(smhttp, "_sleep", lambda s: os.environ.__setitem__("SCIGANTIC_MSA_NO_REMOTE", "1"))
    try:
        with pytest.raises(smhttp.RemoteNotAllowed):
            smhttp.request("POST", f"http://127.0.0.1:{srv.server_port}/ticket/msa", data=b"q=" + UBQ.encode(),
                           retry=smhttp.RetryPolicy(attempts=4))
    finally:
        srv.shutdown()
    assert len(hits) == 1


# 6: a value of only spaces was stripped to '' and read as unset.
@pytest.mark.parametrize("value", [" ", "\t", "  \n"])
def test_no_remote_blank_but_set_value_counts(value, monkeypatch):
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", value)
    assert smhttp.remote_disabled()


def test_no_remote_empty_is_unset_and_readme_says_so(monkeypatch):
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "")
    assert not smhttp.remote_disabled()
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", " Off ")
    assert not smhttp.remote_disabled()
    text = " ".join(README.read_text().split())
    assert "including one of only spaces" in text
    assert "an empty value (`SCIGANTIC_MSA_NO_REMOTE=`) is the same as not setting it" in text
    assert "and so does every retry" in text


# 5: a relative XDG_CACHE_HOME was used, so the cache landed in the
# folder a run started in.
@pytest.mark.parametrize("xdg", ["relative/cache", "~/xdg", "."])
def test_relative_xdg_cache_home_is_ignored(xdg, tmp_path, monkeypatch):
    monkeypatch.delenv("SCIGANTIC_MSA_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    assert cache_dir() == Path.home() / ".cache" / "scigantic-msa"


def test_absolute_xdg_cache_home_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv("SCIGANTIC_MSA_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "scigantic-msa"


@pytest.mark.skipif(sys.platform == "win32", reason="checks the POSIX reading of a Windows setting")
def test_relative_localappdata_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("SCIGANTIC_MSA_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cachemod.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "AppData")
    assert cache_dir() == Path.home() / "AppData" / "Local" / "scigantic-msa" / "cache"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    assert cache_dir() == tmp_path / "lad" / "scigantic-msa" / "cache"


# 7: an older 0755 cache stayed listable when a run only read it.
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_reading_an_old_cache_makes_it_private(tmp_path):
    root = tmp_path / "cache"
    d = root / "alignments" / "ab" / ("ab" + "0" * 62)
    d.mkdir(parents=True)
    for p in (root, root / "alignments", d.parent, d):
        os.chmod(p, 0o755)
    MSA(host="http://127.0.0.1:9", cache_dir=root, quiet=True).cached(UBQ)
    assert stat.S_IMODE((root / "alignments").stat().st_mode) == 0o700
    # The folder the user chose keeps its mode.
    assert stat.S_IMODE(root.stat().st_mode) == 0o755


def test_readme_says_an_old_cache_is_made_private_on_read():
    text = " ".join(README.read_text().split())
    assert "a cache made by 0.2.7 or older is made so the first time it is used, even only to read it" in text


# 4: a contact that is not Latin-1 failed with a codec error after the
# tool said it was sending.
def test_non_latin1_contact_is_refused_before_saying_it_sends(tmp_path, monkeypatch, capsys):
    srv, hits = _server(200, b'{"status": "ERROR"}')
    monkeypatch.setenv("SCIGANTIC_MSA_CONTACT", "名前 <a@example.org>")
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "c"))
    monkeypatch.delenv("SCIGANTIC_MSA_NO_REMOTE", raising=False)
    try:
        rc = main(["run", "=" + UBQ, "-m", "colabfold", "-o", str(tmp_path / "o"), "--yes",
                   "--host", f"http://127.0.0.1:{srv.server_port}"])
    finally:
        srv.shutdown()
    err = capsys.readouterr().err
    assert rc != 0
    assert hits == []
    assert "codec" not in err, err
    assert "This sends" not in err and "sending" not in err, err
    assert "SCIGANTIC_MSA_CONTACT" in err


@pytest.mark.parametrize("contact", ["a@example.org\r\nX-Injected: 1", "line\nbreak", "tab\there"])
def test_contact_with_control_characters_is_refused(contact, monkeypatch):
    with pytest.raises(ValueError, match="the contact"):
        ColabFoldClient("http://127.0.0.1:9", contact=contact)
    monkeypatch.setenv("SCIGANTIC_MSA_CONTACT", contact)
    with pytest.raises(ValueError, match="SCIGANTIC_MSA_CONTACT"):
        ColabFoldClient("http://127.0.0.1:9")


def test_latin1_contact_is_still_accepted():
    c = ColabFoldClient("http://127.0.0.1:9", contact="José <jose@example.org>")
    assert c.user_agent.endswith("José <jose@example.org>")


def test_request_refuses_a_bad_header_before_sending(monkeypatch):
    srv, hits = _server(200)
    monkeypatch.delenv("SCIGANTIC_MSA_NO_REMOTE", raising=False)
    try:
        with pytest.raises(ValueError, match="User-Agent"):
            smhttp.request("GET", f"http://127.0.0.1:{srv.server_port}/", headers={"User-Agent": "x 名"})
    finally:
        srv.shutdown()
    assert hits == []
