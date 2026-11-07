import gzip
import os
import stat

import pytest

from conftest import Q_6T33, Q_LIVE, key
from fake_backend import CODE, FakeBackend
from scigantic_msa.a3m import Row
from scigantic_msa.account import AccountClient, AccountError, NotLoggedIn, config_path, load_config
from scigantic_msa.msa import MSA


def sample_msa(q=Q_6T33, source="remote"):
    rows = [Row("101", q, "query"), Row("UniRef100_X\t1", q[:-1] + "A", "uniref", "9606"), Row("SRR1_2", "-" + q[1:], "env")]
    return MSA(query=q, rows=rows, source=source, provenance={"from": "remote", "backend": "colabfold", "mode": "env-taxonomy"},
               attribution=[{"id": "colabfold-api"}])


@pytest.fixture
def backend(server, monkeypatch):
    monkeypatch.setenv("SCIGANTIC_API_URL", server.url)
    return FakeBackend(server)


def logged_in(backend, email="old@lab.edu"):
    c = AccountClient()
    c.login_start(email)
    c.login_finish(email, CODE)
    return AccountClient()


def test_login_creates_an_account_in_one_round_trip(backend):
    c = AccountClient()
    sent = c.login_start("new@lab.edu")
    assert sent["accountExists"] is False and backend.codes_sent == ["new@lab.edu"]
    with pytest.raises(AccountError) as e:
        c.login_finish("new@lab.edu", CODE)
    assert e.value.code == "terms_required"
    d = c.login_finish("new@lab.edu", CODE, accept_terms=True)
    assert d["accountCreated"] is True
    cfg = load_config()
    assert cfg["token"] == d["token"] and cfg["email"] == "new@lab.edu"
    mode = stat.S_IMODE(os.stat(config_path()).st_mode)
    assert mode == 0o600
    assert AccountClient().account()["email"] == "new@lab.edu"


def test_wrong_code(backend):
    c = AccountClient()
    with pytest.raises(AccountError) as e:
        c.login_finish("old@lab.edu", "AAAA-BBBB-CCCC")
    assert e.value.status == 401
    assert "token" not in load_config()


def test_not_logged_in(backend):
    with pytest.raises(NotLoggedIn, match="scigantic-msa login"):
        AccountClient().account()


def test_save_list_get_download_delete(backend):
    c = logged_in(backend)
    m = sample_msa()
    aid = c.save(m, label="my run")
    it = backend.items[aid]
    assert it["status"] == "ready" and it["seqSha256"] == key(Q_6T33) and it["label"] == "my run"
    assert it["source"] == "colabfold-api" and it["kind"] == "unpaired"
    assert "_owner" not in c.get(aid)
    put = [r for r in backend.srv.requests if r.method == "PUT"][0]
    assert "Authorization" not in put.headers  # the presigned URL carries its own signature
    assert int(put.headers["Content-Length"]) == it["sizeBytes"]
    got = c.download(aid)
    assert got.source == "account" and got.account_id == aid
    assert [(r.header, r.seq, r.taxid) for r in got.rows] == [(r.header, r.seq, r.taxid) for r in m.rows]
    assert got.provenance["backend"] == "colabfold"
    assert [a["alignmentId"] for a in c.list()] == [aid]
    assert c.lookup([key(Q_6T33), key(Q_LIVE)]) == {key(Q_6T33): [c.get(aid)]}
    assert c.delete(aid) is True
    assert list(c.list()) == []


def test_pagination(backend):
    c = logged_in(backend)
    ids = {c.save(sample_msa()) for _ in range(5)}
    assert {a["alignmentId"] for a in c.list()} == ids


def test_other_users_get_404(backend):
    a = logged_in(backend)
    aid = a.save(sample_msa())
    b = AccountClient()
    b.login_start("new@lab.edu")
    b.login_finish("new@lab.edu", CODE, accept_terms=True)
    b = AccountClient()
    for call in (lambda: b.get(aid), lambda: b.download(aid), lambda: b.delete(aid)):
        with pytest.raises(AccountError) as e:
            call()
        assert e.value.status == 404
    assert b.lookup([key(Q_6T33)]) == {}


def test_server_rejects_a_mislabelled_upload(backend, monkeypatch):
    c = logged_in(backend)
    m = sample_msa()
    real_bundle = m.to_bundle
    other = sample_msa(Q_LIVE)
    monkeypatch.setattr(m, "to_bundle", other.to_bundle)
    with pytest.raises(AccountError) as e:
        c.save(m)
    assert e.value.code == "sequence_mismatch"
    monkeypatch.setattr(m, "to_bundle", real_bundle)


def test_download_checksum_mismatch_is_refused(backend):
    c = logged_in(backend)
    aid = c.save(sample_msa())
    blob = bytearray(backend.blobs[aid])
    blob[-5] ^= 1
    backend.blobs[aid] = bytes(blob)
    with pytest.raises(AccountError, match="checksum"):
        c.download(aid)


def test_bundle_is_gzip_a3m_the_backend_can_check():
    m = sample_msa()
    text = gzip.decompress(m.to_bundle()).decode()
    assert text.startswith("#scigantic-msa ")
    from scigantic_msa.a3m import a3m_query

    assert a3m_query(text) == Q_6T33
    back = MSA.from_bundle(m.to_bundle(), source="cache")
    assert back.rows[1].taxid == "9606" and back.provenance == m.provenance


def test_bundle_row_count_mismatch_is_caught():
    m = sample_msa()
    text = gzip.decompress(m.to_bundle()).decode()
    text = text.rsplit(">", 1)[0]  # drop the last record
    from scigantic_msa.a3m import A3MError

    with pytest.raises(A3MError, match="metadata says"):
        MSA.from_bundle(gzip.compress(text.encode()), source="cache")


def test_notebook_zero_step_and_remint_on_401(server, monkeypatch):
    monkeypatch.setenv("SCIGANTIC_API_URL", server.url)
    be = FakeBackend(server, notebooks={"nb1": {"live": "L1", "email": "old@lab.edu"}})
    monkeypatch.setenv("SCIGANTIC_NOTEBOOK_ID", "nb1")
    monkeypatch.setenv("SCIGANTIC_LIVENESS_TOKEN", "L1")
    c = AccountClient()
    assert c.logged_in and c.auth_method() == "notebook"
    aid = c.save(sample_msa())
    assert be.notebook_mints == 1
    mint = [r for r in server.requests if r.path == "/api/msa/token/notebook"][0]
    assert mint.headers["X-Scigantic-Liveness"] == "L1" and mint.json() == {"notebookId": "nb1"}
    be.expire_next.add(c._token)
    assert c.get(aid)["alignmentId"] == aid
    assert be.notebook_mints == 2
    assert not config_path().exists()  # the notebook token is never written to disk


@pytest.mark.parametrize("nb,code", [
    ({"live": "L1", "email": "x@lab.edu", "trial": True}, "trial_notebook"),
    ({"live": "L1", "email": "x@lab.edu", "status": "stopped"}, "notebook_not_running"),
    ({"live": "OTHER", "email": "x@lab.edu"}, None),
])
def test_notebook_refusals(server, monkeypatch, nb, code):
    monkeypatch.setenv("SCIGANTIC_API_URL", server.url)
    FakeBackend(server, notebooks={"nb1": nb})
    monkeypatch.setenv("SCIGANTIC_NOTEBOOK_ID", "nb1")
    monkeypatch.setenv("SCIGANTIC_LIVENESS_TOKEN", "L1")
    with pytest.raises(AccountError) as e:
        AccountClient().account()
    assert e.value.code == code


def test_env_token_wins_and_expired_saved_token_is_ignored(backend, monkeypatch):
    from scigantic_msa.account import save_config

    save_config({"token": "old", "expiresAt": 1})
    assert not AccountClient().logged_in
    monkeypatch.setenv("SCIGANTIC_MSA_TOKEN", "envtok")
    assert AccountClient().auth_method() == "env"


def test_refresh_slides_the_saved_token(backend):
    c = logged_in(backend)
    old = load_config()["token"]
    c.refresh()
    assert load_config()["token"] != old


def test_lookup_slides_a_token_close_to_expiry(backend, monkeypatch):
    from scigantic_msa.account import save_config

    c = logged_in(backend)
    cfg = load_config()
    cfg["expiresAt"] = __import__("time").time() + 86400  # one day left
    save_config(cfg)
    monkeypatch.setattr(AccountClient, "_refreshed", False)
    c = AccountClient()
    old = c._token
    c.lookup([key(Q_6T33)])
    assert load_config()["token"] != old
    assert backend.srv.count(lambda r: r.path == "/api/msa/token/refresh") == 1


def test_unreachable_api_is_an_account_error(monkeypatch):
    monkeypatch.setenv("SCIGANTIC_API_URL", "http://127.0.0.1:9")
    c = AccountClient(token="t")
    with pytest.raises(AccountError) as e:
        c.lookup([key(Q_6T33)])
    assert e.value.code == "unreachable" and "could not reach" in e.value.message


def test_deep_bundle_keeps_the_query_inside_the_api_check_window():
    # The API reads only the first 256 KiB of a stored file to find the query.
    # A taxid map for 60k rows is about 1 MB, so it must not come first.
    import zlib

    from scigantic_msa.a3m import Row, a3m_query

    q = Q_6T33
    rows = [Row("q", q, "query")] + [
        Row(f"UniRef100_X{i}\t1", "-" * (len(q) - 3) + q[-3:], "uniref", str(9606 + i)) for i in range(60000)
    ]
    m = MSA(query=q, rows=rows, source="remote", provenance={"from": "remote"})
    data = m.to_bundle()
    assert len(data) > 256 * 1024
    head = zlib.decompressobj(wbits=31).decompress(data[: 256 * 1024])
    text = head.decode("utf-8", "replace")
    first_two = text.split("\n>", 2)
    assert len(first_two) == 3, "the query record does not end inside the first 256 KiB"
    assert a3m_query(first_two[0] + "\n>" + first_two[1]) == q
    back = MSA.from_bundle(data, source="cache")
    assert back.depth == m.depth
    assert back.rows[1].taxid == "9606" and back.rows[-1].taxid == str(9606 + 59999)


@pytest.mark.parametrize("prov,derived", [
    ({"from": "remote", "backend": "nvidia", "databases": [f"db{i}" for i in range(17)]}, None),
    ({"from": "remote", "backend": "nvidia", "databases": ["colabfold/uniref30"]}, None),
    ({"from": "index", "source_id": "x" * 300}, None),
    ({"from": "index"}, {"parent_seq_sha256": "a" * 64, "identity": None}),
])
def test_metadata_the_api_would_refuse_is_dropped_not_sent(backend, prov, derived):
    c = logged_in(backend)
    m = sample_msa()
    m.provenance = prov
    m.derived = derived
    aid = c.save(m)
    assert c.get(aid)["status"] == "ready"


def test_index_results_are_labelled_index():
    m = sample_msa()
    m.provenance = {"from": "index"}
    from scigantic_msa.account import _source_label

    assert _source_label(m) == "index"
