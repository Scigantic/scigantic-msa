import json

import pytest

from conftest import Q_LIVE
from scigantic_msa.remote import NvidiaClient, RemoteError


def a3m(q, n=3):
    return f">query\n{q}\n" + "".join(f">hit{i}\n{q[:-1]}A\n" for i in range(n))


def client(server, **kw):
    return NvidiaClient(server.url + "/predict", api_key="k", status_url=server.url + "/status/{reqid}",
                        sleep=lambda s: None, **kw)


def test_merged_alignment(server):
    body = {"alignments": {"uniref30": {"a3m": {"alignment": a3m(Q_LIVE, 1)}},
                           "colabfold": {"a3m": {"alignment": a3m(Q_LIVE, 4)}}}}
    server.route("POST", "/predict", lambda r: (200, {}, json.dumps(body).encode()))
    res = client(server).search(Q_LIVE)
    assert len(res.rows) == 5 and res.provenance["alignment_key"] == "colabfold"
    req = server.requests[0]
    assert req.headers["Authorization"] == "Bearer k"
    assert req.json()["sequence"] == Q_LIVE and req.json()["output_alignment_formats"] == ["a3m"]


def test_202_is_polled(server):
    server.route("POST", "/predict", lambda r: (202, {"NVCF-REQID": "abc"}, b""))
    answers = [(202, {"NVCF-REQID": "abc"}, b""),
               (200, {}, json.dumps({"alignments": {"db": {"a3m": {"alignment": a3m(Q_LIVE)}}}}).encode())]
    server.route("GET", "/status/abc", lambda r: answers.pop(0))
    assert len(client(server).search(Q_LIVE).rows) == 4


@pytest.mark.parametrize("status,body,msg", [
    (401, b"", "rejected the API key"),
    (500, b"x", "HTTP 500"),
    (200, b"{}", "unexpected response"),
    (200, json.dumps({"alignments": {"db": {"a3m": {"alignment": ">query\nMKV\n"}}}}).encode(), "failed validation"),
])
def test_errors(server, status, body, msg):
    server.route("POST", "/predict", lambda r: (status, {}, body))
    with pytest.raises(RemoteError, match=msg):
        client(server).search(Q_LIVE)


def test_needs_a_key():
    with pytest.raises(RemoteError, match="NVIDIA_API_KEY"):
        NvidiaClient("http://127.0.0.1:9/p").search(Q_LIVE)
