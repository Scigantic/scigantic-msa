import io
import json
import tarfile
import urllib.parse

import pytest

from conftest import FIXTURES, Q_LIVE
from scigantic_msa.remote.colabfold import ColabFoldClient, RateLimited, RemoteError, query_text, ticket_id

LIVE_TAR = (FIXTURES / "colabfold-live1.tar.gz").read_bytes()
LIVE_ID = "dwio757X99SN0gJJWTevrvWcDWsX4CA6hguAgw"


def test_ticket_id_matches_the_live_server():
    assert ticket_id(query_text(Q_LIVE), "env-taxonomy") == LIVE_ID
    assert ticket_id(query_text(Q_LIVE.lower() + "*"), "env-taxonomy") == LIVE_ID
    assert ticket_id(query_text(Q_LIVE), "env") != LIVE_ID
    assert ticket_id("x", "m", ["b", "a"]) == ticket_id("x", "m", ["a", "b"])


class FakeColabFold:
    """Scripted status answers; records submissions."""

    def __init__(self, server, statuses, tar=LIVE_TAR, submit_answers=None):
        self.statuses = list(statuses)
        self.submits = []
        self.submit_answers = list(submit_answers or [])
        self.tar = tar
        server.route("POST", "/ticket/msa", self.submit)
        server.route_prefix("GET", "/ticket/", self.status)
        server.route_prefix("GET", "/result/download/", lambda r: (200, {}, self.tar))

    def submit(self, req):
        form = urllib.parse.parse_qs(req.body.decode())
        self.submits.append(form)
        if self.submit_answers:
            code, body = self.submit_answers.pop(0)
            return code, {"Content-Type": "application/json"}, json.dumps(body).encode()
        tid = ticket_id(form["q"][0], form["mode"][0])
        return 200, {}, json.dumps({"id": tid, "status": "PENDING"}).encode()

    def status(self, req):
        s = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(s, tuple):
            return s[0], {}, s[1]
        return 200, {}, json.dumps({"id": req.path_only.rsplit("/", 1)[1], "status": s}).encode()


def client(server, **kw):
    sleeps = []
    t = [0.0]

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    c = ColabFoldClient(server.url + "/", sleep=sleep, clock=lambda: t[0], **kw)
    return c, sleeps


def test_full_job_uses_the_real_result(server):
    fake = FakeColabFold(server, ["UNKNOWN", "PENDING", "RUNNING", "COMPLETE"])
    c, sleeps = client(server)
    res = c.search(Q_LIVE)
    assert len(fake.submits) == 1
    assert fake.submits[0]["q"] == [f">101\n{Q_LIVE}\n"] and fake.submits[0]["mode"] == ["env-taxonomy"]
    assert res.rows[0].seq == Q_LIVE
    assert res.provenance["ticket"] == LIVE_ID
    # uniref.a3m has 1,123 records and the env file 4,312; the env query repeat is dropped.
    assert res.provenance["uniref_rows"] == 1122
    assert res.provenance["env_rows"] > 4000
    assert res.provenance["rows_with_taxid"] >= 1100
    assert res.templates and "pdb70_m8" in res.templates
    assert all(5 <= s <= 10 for s in sleeps)
    ua = server.requests[0].headers["User-Agent"]
    assert ua.startswith("scigantic-msa/") and "@" not in ua
    # The trailing slash on the host was stripped (no //ticket).
    assert all(not r.path.startswith("//") for r in server.requests)


def test_resume_downloads_without_submitting(server):
    fake = FakeColabFold(server, ["COMPLETE"])
    c, _ = client(server)
    c.search(Q_LIVE)
    assert fake.submits == []


def test_rate_limit_backs_off_and_then_gives_up(server):
    fake = FakeColabFold(server, ["UNKNOWN"], submit_answers=[(429, {"status": "RATELIMIT"})] * 20)
    c, sleeps = client(server, ratelimit_retries=3, ratelimit_initial=60)
    with pytest.raises(RateLimited, match="rate limited"):
        c.search(Q_LIVE)
    assert sleeps == [60, 120, 240]
    assert len(fake.submits) == 4


def test_rate_limit_then_success(server):
    fake = FakeColabFold(server, ["UNKNOWN", "COMPLETE"],
                         submit_answers=[(429, {"status": "RATELIMIT"}), (200, {"id": LIVE_ID, "status": "PENDING"})])
    c, sleeps = client(server)
    res = c.search(Q_LIVE)
    assert sleeps[0] == 60 and len(fake.submits) == 2 and res.rows


def test_ratelimit_status_in_a_200_body(server):
    FakeColabFold(server, ["UNKNOWN", "COMPLETE"],
                  submit_answers=[(200, {"status": "RATELIMIT", "reason": "x"}), (200, {"id": LIVE_ID, "status": "PENDING"})])
    c, sleeps = client(server)
    assert c.search(Q_LIVE).rows
    assert sleeps[0] == 60


def test_unknown_forever_resubmits_once_then_fails(server):
    fake = FakeColabFold(server, ["UNKNOWN"])
    c, _ = client(server, unknown_polls=3)
    with pytest.raises(RemoteError, match="even after submitting it again"):
        c.search(Q_LIVE)
    assert len(fake.submits) == 2


def test_error_status(server):
    FakeColabFold(server, ["UNKNOWN", "ERROR"])
    c, _ = client(server)
    with pytest.raises(RemoteError, match="reported an error"):
        c.search(Q_LIVE)


def test_deadline(server):
    FakeColabFold(server, ["UNKNOWN", "PENDING"])
    c, sleeps = client(server, deadline=60)
    with pytest.raises(RemoteError, match="gave up"):
        c.search(Q_LIVE)
    assert sum(sleeps) <= 60


def test_non_json_and_http_errors(server):
    FakeColabFold(server, [(200, b"<html>maintenance page</html>")])
    c, _ = client(server)
    with pytest.raises(RemoteError, match="not JSON"):
        c.search(Q_LIVE)


def test_http_400_is_an_error(server):
    FakeColabFold(server, [(400, b"bad")])
    c, _ = client(server)
    with pytest.raises(RemoteError, match="HTTP 400"):
        c.search(Q_LIVE)


def test_maintenance_waits_then_gives_up(server):
    FakeColabFold(server, ["MAINTENANCE"])
    c, sleeps = client(server, ratelimit_retries=2)
    with pytest.raises(RemoteError, match="maintenance"):
        c.search(Q_LIVE)
    assert sleeps == [60, 120]


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.parametrize("tar,msg", [
    (b"", "empty body"),
    (b"not a tarball", "not a valid tar.gz"),
    (_tar({"msa.sh": b"x"}), "no uniref.a3m"),
    (_tar({"uniref.a3m": f">101\n{Q_LIVE}\n".encode()}), "no bfd.mgnify30"),
    (_tar({"uniref.a3m": b">101\nMKV\n", "bfd.mgnify30.metaeuk30.smag30.a3m": b">101\nMKV\n"}), "failed validation"),
])
def test_bad_downloads(server, tar, msg):
    FakeColabFold(server, ["COMPLETE"], tar=tar)
    c, _ = client(server)
    with pytest.raises(RemoteError, match=msg):
        c.search(Q_LIVE)


def test_taxids_come_from_uniref_tax(server):
    FakeColabFold(server, ["COMPLETE"])
    c, _ = client(server)
    res = c.search(Q_LIVE)
    tax = {}
    with tarfile.open(fileobj=io.BytesIO(LIVE_TAR)) as tf:
        for line in tf.extractfile("uniref_tax.tsv").read().decode().splitlines():
            t, ident = line.split("\t")[:2]
            tax[ident] = t
    for r in res.rows[1:50]:
        ident = r.header.split("\t")[0]
        assert r.taxid == tax.get(ident)
    assert all(r.taxid is None for r in res.rows if r.db == "env")
