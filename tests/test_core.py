import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, Q_LITEFOLD, Q_LIVE, key
from fake_backend import CODE, FakeBackend
from test_colabfold import FakeColabFold
from scigantic_msa import MSANotFound, get_msa, get_msas, where
from scigantic_msa.account import AccountClient, NotLoggedIn
from scigantic_msa.index import IndexIntegrityError


@pytest.fixture
def env(index_setup, server, monkeypatch):
    idx, base, root = index_setup
    monkeypatch.setenv("SCIGANTIC_MSA_INDEX_URL", base)
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", server.url)
    monkeypatch.setenv("SCIGANTIC_API_URL", server.url)
    from scigantic_msa.remote import colabfold

    monkeypatch.setattr(colabfold.time, "sleep", lambda s: None)
    return server


def remote_calls(server):
    return server.count(lambda r: r.path.startswith(("/ticket", "/result")))


def test_exact_hit(env, capsys):
    m = get_msa(Q_6T33)
    assert m.source == "exact" and m.provenance["index_source"] == "openfold3-pdb"
    assert "CC-BY-4.0" in capsys.readouterr().err
    assert remote_calls(env) == 0


def test_near_hit_for_a_point_mutant(env):
    mut = Q_LITEFOLD[:30] + ("W" if Q_LITEFOLD[30] != "W" else "Y") + Q_LITEFOLD[31:]
    m = get_msa(mut)
    assert m.source == "near" and m.rows[0].seq == mut
    assert m.derived["parent_seq_sha256"] == key(Q_LITEFOLD)
    assert m.derived["mismatches"] == [[31, Q_LITEFOLD[30], mut[30]]]
    assert "near-identical" in m.summary()
    assert remote_calls(env) == 0
    # The mutant's own sequence never reached any server.
    assert env.count(lambda r: mut in r.path or mut.encode() in r.body) == 0


def test_no_remote_by_default(env):
    with pytest.raises(MSANotFound, match="remote='colabfold'") as e:
        get_msa(Q_LIVE)
    assert remote_calls(env) == 0
    assert [s.name for s in e.value.steps] == ["cache", "account", "exact", "near", "remote"]


def test_remote_when_asked_then_cache(env, capsys):
    fake = FakeColabFold(env, ["UNKNOWN", "COMPLETE"])
    m = get_msa(Q_LIVE, remote="colabfold")
    assert m.source == "remote" and len(fake.submits) == 1
    err = capsys.readouterr().err
    assert err.count("sending sequences with no stored MSA to") == 1 and "anyone who knows the sequence" in err
    n = remote_calls(env)
    again = get_msa(Q_LIVE, remote="colabfold")
    assert again.source == "cache" and remote_calls(env) == n
    assert again.provenance["backend"] == "colabfold"
    assert [(r.seq, r.taxid) for r in again.rows] == [(r.seq, r.taxid) for r in m.rows]


def test_announce_once_per_process(env, capsys):
    FakeColabFold(env, ["COMPLETE"])
    get_msa(Q_LIVE, remote="colabfold", cache=False)
    get_msa(Q_LIVE, remote="colabfold", cache=False)
    assert capsys.readouterr().err.count("sending sequences") == 1


def test_env_kill_switch_beats_explicit_remote(env, monkeypatch):
    FakeColabFold(env, ["COMPLETE"])
    monkeypatch.setenv("SCIGANTIC_MSA_NO_REMOTE", "1")
    with pytest.raises(MSANotFound, match="SCIGANTIC_MSA_NO_REMOTE"):
        get_msa(Q_LIVE, remote="colabfold")
    assert remote_calls(env) == 0


def test_config_turns_remote_on(env, monkeypatch):
    from scigantic_msa.account import save_config

    FakeColabFold(env, ["COMPLETE"])
    save_config({"remote": "colabfold"})
    assert get_msa(Q_LIVE).source == "remote"
    with pytest.raises(MSANotFound):
        get_msa(Q_LIVE, remote=False, cache=False)


def test_corrupt_index_stops_before_any_remote_call(env, index_setup):
    _, _, _ = index_setup
    path = "/idx/v1/builds/test-build-1/shards/" + key(Q_LIVE)[:3] + ".jsonl.gz"
    env.static[path] = env.static[path][:10]
    FakeColabFold(env, ["COMPLETE"])
    with pytest.raises(IndexIntegrityError):
        get_msa(Q_LIVE, remote="colabfold")
    assert remote_calls(env) == 0


def test_unreachable_index_falls_through_to_remote(env, monkeypatch):
    monkeypatch.setenv("SCIGANTIC_MSA_INDEX_URL", "http://127.0.0.1:9/")
    FakeColabFold(env, ["COMPLETE"])
    assert get_msa(Q_LIVE, remote="colabfold").source == "remote"


def test_bad_first_entry_falls_back_to_the_next(env, capsys):
    from scigantic_msa.index import Index

    idx = Index()
    e = idx.lookup(key(Q_6T33))[0]
    path = e.file("cfdb_uniref30").url.split(str(env.port), 1)[1]
    env.static[path] = b">x\n" + b"A" * (e.file("cfdb_uniref30").length - 4) + b"\n"
    m = get_msa(Q_6T33)
    assert m.source == "exact" and m.provenance["index_source"] == "openproteinset-pdb"
    assert "did not check out" in capsys.readouterr().err


def test_account_hit_and_save(env):
    be = FakeBackend(env)
    with pytest.raises(NotLoggedIn):
        get_msa(Q_LIVE, remote="colabfold", save=True)
    assert remote_calls(env) == 0
    c = AccountClient()
    c.login_start("old@lab.edu")
    c.login_finish("old@lab.edu", CODE)
    FakeColabFold(env, ["COMPLETE"])
    m = get_msa(Q_LIVE, remote="colabfold", save=True, label="x")
    assert m.account_id in be.items
    # A fresh machine: no cache, signed in, the account answers first.
    got = get_msa(Q_LIVE, cache=False)
    assert got.source == "account" and got.account_id == m.account_id
    assert be.srv.count(lambda r: r.path.startswith("/api/msa/lookup")) >= 1


def test_where_sends_nothing(env):
    FakeColabFold(env, ["COMPLETE"])
    steps = where(Q_LIVE, remote="colabfold")
    assert steps[-1].outcome == "would-run" and "127.0.0.1" in steps[-1].detail
    assert remote_calls(env) == 0
    steps = where(Q_6T33)
    assert any(s.name == "exact" and s.outcome == "hit" for s in steps)
    # where() reads only index files, never alignment bytes.
    assert env.count(lambda r: r.path.startswith("/src/")) == 0


def test_complex_pairs_real_chains(env, capsys):
    cx = get_msas([Q_7M22A, Q_7M22C])
    assert [c.source for c in cx] == ["exact", "exact"]
    assert cx.pairing.n_paired == 8
    assert all(c.uniprot_rows for c in cx)
    assert "paired 8 rows" in capsys.readouterr().err


def test_homodimer_is_one_lookup(env):
    cx = get_msas([Q_6T33, Q_6T33])
    assert cx[0] is cx[1]
    assert cx.pairing.message.startswith("one chain")


def test_signed_in_with_the_api_down_still_uses_the_index(env, monkeypatch):
    monkeypatch.setenv("SCIGANTIC_MSA_TOKEN", "saved-token")
    monkeypatch.setenv("SCIGANTIC_API_URL", "http://127.0.0.1:9")
    m = get_msa(Q_6T33)
    assert m.source == "exact"
    steps = where(Q_6T33)
    assert steps[1].name == "account" and steps[1].outcome == "error"


def test_an_unreachable_remote_is_a_miss_not_a_crash(env, monkeypatch):
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", "http://127.0.0.1:9")
    with pytest.raises(MSANotFound) as e:
        get_msa(Q_LIVE, remote="colabfold")
    assert e.value.steps[-1].name == "remote" and e.value.steps[-1].outcome == "error"


def test_near_parent_shard_unreachable_is_reported_not_raised(env):
    mut = Q_LITEFOLD[:30] + ("W" if Q_LITEFOLD[30] != "W" else "Y") + Q_LITEFOLD[31:]
    parent = key(Q_LITEFOLD)
    assert key(mut)[:3] != parent[:3]
    env.faults[f"/idx/v1/builds/test-build-1/shards/{parent[:3]}.jsonl.gz"] = [("status", 503, {}, b"")] * 10
    with pytest.raises(MSANotFound) as e:
        get_msa(mut)
    near = [s for s in e.value.steps if s.name == "near"]
    assert near and near[0].outcome == "error" and "not reachable" in near[0].detail


def _creates(server):
    return server.count(lambda r: r.method == "POST" and r.path_only == "/api/msa/alignments")


def test_save_on_a_cache_hit_does_not_upload_the_same_alignment_again(env):
    fb = FakeBackend(env)
    c = AccountClient()
    c.login_start("old@lab.edu")
    c.login_finish("old@lab.edu", CODE)
    FakeColabFold(env, ["COMPLETE"])
    first = get_msa(Q_LIVE, remote="colabfold", save=True)
    assert first.source == "remote" and _creates(env) == 1
    again = get_msa(Q_LIVE, save=True)
    assert again.source == "cache" and again.account_id == first.account_id
    assert _creates(env) == 1
    # Deleted from the account: the next save=True stores it again.
    AccountClient().delete(first.account_id)
    third = get_msa(Q_LIVE, save=True)
    assert _creates(env) == 2 and third.account_id != first.account_id
    assert len(fb.items) == 1


def test_no_upload_without_save_and_only_hashes_reach_the_api(env):
    FakeBackend(env)
    c = AccountClient()
    c.login_start("old@lab.edu")
    c.login_finish("old@lab.edu", CODE)
    FakeColabFold(env, ["COMPLETE"])
    get_msa(Q_LIVE, remote="colabfold")
    get_msa(Q_6T33)
    get_msas([Q_7M22A, Q_7M22C])
    assert _creates(env) == 0 and env.count(lambda r: r.method == "PUT") == 0
    for r in env.requests:
        if r.path.startswith("/api/"):
            for q in (Q_LIVE, Q_6T33, Q_7M22A, Q_7M22C):
                assert q.encode() not in r.body and q not in r.path


def test_remote_false_talks_only_to_the_index(env):
    seq = "MKVLAAGIVGLLLAACSPEQKTEEKKWWWWHHHHPPPP"
    with pytest.raises(MSANotFound):
        get_msa(seq, remote=False)
    assert [r.path for r in env.requests if not r.path.startswith("/idx/")] == []
    assert not any(seq.encode() in r.body or seq in r.path for r in env.requests)
