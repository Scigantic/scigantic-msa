import email.utils
import time

import pytest

from scigantic_msa import http


@pytest.fixture
def sleeps(monkeypatch):
    got = []
    monkeypatch.setattr(http, "_sleep", lambda s: got.append(s))
    return got


def get(url, **kw):
    return http.request("GET", url, **kw).body


def test_user_agent_names_the_tool(server):
    server.static["/f"] = b"ok"
    assert get(server.url + "/f") == b"ok"
    r = server.requests[-1]
    assert r.headers["User-Agent"].startswith("scigantic-msa/")
    assert "github.com/Scigantic/scigantic-msa" in r.headers["User-Agent"]


def test_truncated_response_is_retried_then_fails(server, sleeps):
    server.static["/f"] = b"x" * 1000
    for _ in range(5):
        server.faults["/f"].append(("truncate", 10))
    with pytest.raises(http.HTTPError):
        get(server.url + "/f")


def test_truncated_once_then_ok(server, sleeps):
    server.static["/f"] = b"x" * 1000
    server.faults["/f"].append(("truncate", 10))
    assert get(server.url + "/f") == b"x" * 1000


def test_429_honours_retry_after_seconds(server, sleeps):
    server.static["/f"] = b"ok"
    server.faults["/f"].append(("status", 429, {"Retry-After": "7"}, b"slow down"))
    assert get(server.url + "/f") == b"ok"
    assert sleeps == [7.0]


def test_429_retry_after_http_date(server, sleeps):
    server.static["/f"] = b"ok"
    when = email.utils.formatdate(time.time() + 30, usegmt=True)
    server.faults["/f"].append(("status", 429, {"Retry-After": when}, b""))
    assert get(server.url + "/f") == b"ok"
    assert 25 <= sleeps[0] <= 31


def test_retry_after_beyond_the_cap_gives_up(server, sleeps):
    server.static["/f"] = b"ok"
    server.faults["/f"].append(("status", 429, {"Retry-After": "86400"}, b""))
    with pytest.raises(http.HTTPError, match="wait 86400"):
        get(server.url + "/f")
    assert sleeps == []


def test_5xx_backoff_then_success(server, sleeps):
    server.static["/f"] = b"ok"
    for code in (500, 502, 503):
        server.faults["/f"].append(("status", code, {}, b""))
    assert get(server.url + "/f") == b"ok"
    assert len(sleeps) == 3 and sleeps[0] < sleeps[2]


def test_persistent_5xx_fails(server, sleeps):
    server.static["/f"] = b"ok"
    for _ in range(10):
        server.faults["/f"].append(("status", 503, {}, b""))
    with pytest.raises(http.HTTPError, match="after 5 attempts"):
        get(server.url + "/f")


def test_404_is_not_retried(server, sleeps):
    with pytest.raises(http.NotFound):
        get(server.url + "/missing")
    assert sleeps == [] and len(server.requests) == 1


def test_timeout_is_retried(server, sleeps):
    server.static["/f"] = b"ok"
    server.faults["/f"].append(("sleep", 1.5))
    resp = http.request("GET", server.url + "/f", timeout=0.5)
    assert resp.body == b"ok"
    assert len(sleeps) == 1


def test_dns_failure_is_not_retried(sleeps):
    with pytest.raises(http.HTTPError, match="cannot resolve"):
        get("https://no-such-host.invalid/x")
    assert sleeps == []


def test_authorization_is_dropped_on_a_cross_host_redirect(server):
    # Redirect from host A (127.0.0.1) to host B (localhost).
    server.static["/target"] = b"0123456789"
    other = server.url.replace("127.0.0.1", "localhost")
    server.route("GET", "/redir", lambda r: (307, {"Location": other + "/target"}, b""))
    # An Authorization header the caller set is dropped on the cross-host hop...
    resp = http.request("GET", server.url + "/redir", headers={"Authorization": "Bearer secret"})
    assert resp.body == b"0123456789"
    first, final = server.requests[-2], server.requests[-1]
    assert first.headers.get("Authorization") == "Bearer secret"
    assert final.path == "/target" and "Authorization" not in final.headers
    # ...and kept on a same-host redirect.
    server.route("GET", "/redir-same", lambda r: (307, {"Location": server.url + "/target"}, b""))
    http.request("GET", server.url + "/redir-same", headers={"Authorization": "Bearer secret"})
    assert server.requests[-1].headers.get("Authorization") == "Bearer secret"
