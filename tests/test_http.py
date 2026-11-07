import email.utils
import time

import pytest

from scigantic_msa import http


@pytest.fixture
def sleeps(monkeypatch):
    got = []
    monkeypatch.setattr(http, "_sleep", lambda s: got.append(s))
    return got


def test_range_read(server):
    server.static["/f"] = bytes(range(256)) * 4
    assert http.get_bytes(server.url + "/f", offset=10, length=5) == bytes([10, 11, 12, 13, 14])
    r = server.requests[-1]
    assert r.headers["Range"] == "bytes=10-14"
    assert r.headers["User-Agent"].startswith("scigantic-msa/")
    assert "github.com/Scigantic/scigantic-msa" in r.headers["User-Agent"]


def test_server_that_ignores_range_is_sliced(server):
    server.static["/f"] = b"0123456789"
    server.faults["/f"].append(("ignore_range",))
    assert http.get_bytes(server.url + "/f", offset=3, length=4) == b"3456"


def test_short_body_is_an_error_not_data(server):
    server.static["/f"] = b"0123456789"
    with pytest.raises(http.HTTPError, match="wanted 20 bytes, received"):
        http.get_bytes(server.url + "/f", offset=0, length=20)


def test_truncated_response_is_retried_then_fails(server, sleeps):
    server.static["/f"] = b"x" * 1000
    for _ in range(5):
        server.faults["/f"].append(("truncate", 10))
    with pytest.raises(http.HTTPError):
        http.get_bytes(server.url + "/f")


def test_truncated_once_then_ok(server, sleeps):
    server.static["/f"] = b"x" * 1000
    server.faults["/f"].append(("truncate", 10))
    assert http.get_bytes(server.url + "/f", length=1000) == b"x" * 1000


def test_429_honours_retry_after_seconds(server, sleeps):
    server.static["/f"] = b"ok"
    server.faults["/f"].append(("status", 429, {"Retry-After": "7"}, b"slow down"))
    assert http.get_bytes(server.url + "/f") == b"ok"
    assert sleeps == [7.0]


def test_429_retry_after_http_date(server, sleeps):
    server.static["/f"] = b"ok"
    when = email.utils.formatdate(time.time() + 30, usegmt=True)
    server.faults["/f"].append(("status", 429, {"Retry-After": when}, b""))
    assert http.get_bytes(server.url + "/f") == b"ok"
    assert 25 <= sleeps[0] <= 31


def test_retry_after_beyond_the_cap_gives_up(server, sleeps):
    server.static["/f"] = b"ok"
    server.faults["/f"].append(("status", 429, {"Retry-After": "86400"}, b""))
    with pytest.raises(http.HTTPError, match="wait 86400"):
        http.get_bytes(server.url + "/f")
    assert sleeps == []


def test_5xx_backoff_then_success(server, sleeps):
    server.static["/f"] = b"ok"
    for code in (500, 502, 503):
        server.faults["/f"].append(("status", code, {}, b""))
    assert http.get_bytes(server.url + "/f") == b"ok"
    assert len(sleeps) == 3 and sleeps[0] < sleeps[2]


def test_persistent_5xx_fails(server, sleeps):
    server.static["/f"] = b"ok"
    for _ in range(10):
        server.faults["/f"].append(("status", 503, {}, b""))
    with pytest.raises(http.HTTPError, match="after 5 attempts"):
        http.get_bytes(server.url + "/f")


def test_404_is_not_retried(server, sleeps):
    with pytest.raises(http.NotFound):
        http.get_bytes(server.url + "/missing")
    assert sleeps == [] and len(server.requests) == 1


def test_timeout_is_retried(server, sleeps):
    server.static["/f"] = b"ok"
    server.faults["/f"].append(("sleep", 1.5))
    resp = http.request("GET", server.url + "/f", timeout=0.5)
    assert resp.body == b"ok"
    assert len(sleeps) == 1


def test_dns_failure_is_not_retried(sleeps):
    with pytest.raises(http.HTTPError, match="cannot resolve"):
        http.get_bytes("https://no-such-host.invalid/x")
    assert sleeps == []


def test_hf_token_sent_to_hf_only_and_dropped_on_cross_host_redirect(server, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert http._auth_headers_for("https://huggingface.co/x") == {"Authorization": "Bearer hf_secret"}
    assert http._auth_headers_for("https://example.org/x") == {}
    # Redirect from host A (127.0.0.1) to host B (localhost) must drop Authorization, keep Range.
    server.static["/target"] = b"0123456789"
    other = server.url.replace("127.0.0.1", "localhost")
    server.route("GET", "/redir", lambda r: (307, {"Location": other + "/target"}, b""))
    body = http.get_bytes(server.url + "/redir", offset=2, length=3)
    assert body == b"234"
    final = server.requests[-1]
    assert final.path == "/target"
    assert "Authorization" not in final.headers
    assert final.headers.get("Range") == "bytes=2-4"


def test_local_files(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abcdef")
    assert http.get_bytes(str(p), offset=1, length=3) == b"bcd"
    assert http.get_bytes(p.as_uri()) == b"abcdef"
    with pytest.raises(http.HTTPError):
        http.get_bytes(str(p), offset=4, length=5)
    with pytest.raises(http.NotFound):
        http.get_bytes(str(tmp_path / "nope"))
