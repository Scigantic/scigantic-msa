import gzip
import json

import pytest

from conftest import Q_1HVW, Q_6T33, Q_LITEFOLD, build_index, key, serve_dir, serve_real_files, standard_entries
from scigantic_msa.index import Index, IndexIntegrityError, IndexUnavailable


def test_lookup_returns_entries_in_source_priority(index_setup):
    idx, _, _ = index_setup
    hits = idx.lookup(key(Q_6T33))
    assert [h.source for h in hits] == ["openfold3-pdb", "openproteinset-pdb"]
    assert hits[0].file("uniprot").format == "sto"
    assert idx.lookup(key("MKVLAAGIVALLLAAGCS")) == []


def test_lookup_touches_only_manifest_sums_and_one_shard(index_setup, server):
    idx, _, _ = index_setup
    idx.lookup(key(Q_6T33))
    paths = [r.path for r in server.requests]
    assert paths[0].endswith("/v1/manifest.json")
    assert sum(1 for p in paths if "/shards/" in p) == 1
    assert sum(1 for p in paths if p.endswith("SHA256SUMS")) == 1


def test_shards_are_cached_per_build(index_setup, server, tmp_path):
    idx, base, _ = index_setup
    idx.lookup(key(Q_6T33))
    n = len(server.requests)
    again = Index(base, cache_root=tmp_path / "cache")
    assert again.lookup(key(Q_6T33))
    assert len(server.requests) == n  # manifest within TTL, sums and shard from disk


def test_offline_mode_uses_only_the_cache(index_setup, server, tmp_path):
    idx, base, _ = index_setup
    idx.lookup(key(Q_6T33))
    off = Index(base, cache_root=tmp_path / "cache", offline=True)
    n = len(server.requests)
    assert off.lookup(key(Q_6T33))
    with pytest.raises(IndexUnavailable, match="offline"):
        off.lookup(key(Q_1HVW)) if key(Q_1HVW)[:3] != key(Q_6T33)[:3] else off.shard("fff")
    assert len(server.requests) == n


def test_local_directory_index(index_setup):
    _, _, root = index_setup
    idx = Index(str(root))
    assert [h.source for h in idx.lookup(key(Q_LITEFOLD))] == ["litefold-human-colabfold"]


def _corrupt_shard(server, base, root, sha, transform):
    prefix = sha[:3]
    path = "/idx/v1/builds/test-build-1/shards/" + prefix + ".jsonl.gz"
    server.static[path] = transform(server.static[path])
    return path


@pytest.mark.parametrize("name,transform,msg", [
    ("truncated", lambda b: b[: len(b) // 2], "does not match its checksum"),
    ("empty", lambda b: b"", "does not match its checksum"),
])
def test_bad_shard_bytes_are_rejected(index_setup, server, name, transform, msg):
    idx, base, root = index_setup
    _corrupt_shard(server, base, root, key(Q_6T33), transform)
    with pytest.raises(IndexIntegrityError, match=msg):
        idx.lookup(key(Q_6T33))


def _rebuild_with_shard(tmp_path, server, sha, edit):
    """Rewrite one shard's content AND its checksum, so only the content checks can catch it."""
    urls = serve_real_files(server)
    root = build_index(tmp_path / "ix2", standard_entries(urls))
    broot = root / "v1" / "builds" / "test-build-1"
    sp = broot / "shards" / (sha[:3] + ".jsonl.gz")
    lines = gzip.decompress(sp.read_bytes()).decode().splitlines()
    lines = edit(lines)
    sp.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))
    import hashlib

    sums = []
    for line in (broot / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        if rel == f"shards/{sha[:3]}.jsonl.gz":
            h = hashlib.sha256(sp.read_bytes()).hexdigest()
        sums.append(f"{h}  {rel}")
    sb = ("\n".join(sums) + "\n").encode()
    (broot / "SHA256SUMS").write_bytes(sb)
    m = json.loads((root / "v1" / "manifest.json").read_text())
    m["sha256sums_sha256"] = hashlib.sha256(sb).hexdigest()
    (root / "v1" / "manifest.json").write_text(json.dumps(m))
    base = serve_dir(server, root, "/ix2")
    return Index(base, cache_root=tmp_path / "c2")


def _hdr(lines, **kw):
    h = json.loads(lines[0])
    h.update(kw)
    return [json.dumps(h)] + lines[1:]


@pytest.mark.parametrize("edit,msg", [
    (lambda ls: ls[1:], "header"),
    (lambda ls: _hdr(ls, _shard="zzz"), "header names"),
    (lambda ls: _hdr(ls, build_id="other"), "from build other"),
    (lambda ls: _hdr(ls, entries=json.loads(ls[0])["entries"] + 1), "header says"),
    (lambda ls: ls[:1] + [json.dumps(dict(json.loads(ls[1]), seq_sha256="f" * 64))] + ls[2:], "wrong shard"),
    (lambda ls: ls[:1] + ["{not json"] + ls[2:], "not JSON"),
    (lambda ls: ls[:1] + [json.dumps({k: v for k, v in json.loads(ls[1]).items() if k != "files"})] + ls[2:], "bad index entry"),
])
def test_shard_content_checks(tmp_path, server, edit, msg):
    sha = key(Q_6T33)
    idx = _rebuild_with_shard(tmp_path, server, sha, edit)
    with pytest.raises(IndexIntegrityError, match=msg):
        idx.lookup(sha)


def test_wrong_sha256sums_is_rejected(index_setup, server):
    idx, _, _ = index_setup
    server.static["/idx/v1/builds/test-build-1/SHA256SUMS"] += b"0" * 64 + b"  extra\n"
    with pytest.raises(IndexIntegrityError, match="SHA256SUMS does not match"):
        idx.lookup(key(Q_6T33))


@pytest.mark.parametrize("body,msg", [
    (b"", "empty"),
    (b"<html>parked domain</html>", "not JSON"),
    (json.dumps({"format": "other"}).encode(), "not a scigantic-msa"),
    (json.dumps({"format": "scigantic-msa-index", "format_version": 2}).encode(), "upgrade"),
    (json.dumps({"format": "scigantic-msa-index", "format_version": 1}).encode(), "lacks build_id"),
])
def test_bad_manifest(server, tmp_path, body, msg):
    server.static["/m/v1/manifest.json"] = body
    with pytest.raises(IndexIntegrityError, match=msg):
        Index(server.url + "/m/", cache_root=tmp_path / "c").manifest()


def test_unreachable_index_is_unavailable_not_a_miss(tmp_path):
    with pytest.raises(IndexUnavailable):
        Index("http://127.0.0.1:9/", cache_root=tmp_path / "c").lookup(key(Q_6T33))


def test_stale_manifest_used_when_network_fails(index_setup, server, tmp_path):
    idx, base, _ = index_setup
    idx.lookup(key(Q_6T33))
    stale = Index(base, cache_root=tmp_path / "cache", manifest_ttl=0)
    server.faults["/idx/v1/manifest.json"] += [("status", 503, {}, b"")] * 10
    assert stale.lookup(key(Q_6T33))


def test_new_build_replaces_cached_shards(server, tmp_path):
    urls = serve_real_files(server)
    root = build_index(tmp_path / "a", standard_entries(urls)[:1], build_id="b1")
    base = serve_dir(server, root, "/r")
    idx = Index(base, cache_root=tmp_path / "c", manifest_ttl=0)
    assert len(idx.lookup(key(Q_6T33))) == 1
    root2 = build_index(tmp_path / "b", standard_entries(urls)[:2], build_id="b2")
    serve_dir(server, root2, "/r")
    idx2 = Index(base, cache_root=tmp_path / "c", manifest_ttl=0)
    assert len(idx2.lookup(key(Q_6T33))) == 2


def test_fetch_checks_length_and_uses_file_cache(index_setup, server):
    idx, _, _ = index_setup
    f = idx.lookup(key(Q_6T33))[1].file("uniref90")
    data = idx.fetch(f)
    assert len(data) == f.length
    n = len(server.requests)
    idx.fetch(f)
    assert len(server.requests) == n


def test_mount_rewrite(index_setup, tmp_path):
    from scigantic_msa.index import FileRef

    idx, _, _ = index_setup
    mnt = tmp_path / "mnt" / "openfold"
    (mnt / "pdb" / "x").mkdir(parents=True)
    (mnt / "pdb" / "x" / "f.a3m").write_text(">q\nMKV\n")
    idx.mounts = {"s3://openfold/": str(mnt)}
    url = "https://openfold.s3.amazonaws.com/pdb/x/f.a3m"
    assert idx.resolve(url) == str(mnt / "pdb" / "x" / "f.a3m")
    assert idx.fetch(FileRef("uniref90", "a3m", url, 0, 7), use_cache=False) == b">q\nMKV\n"
    assert idx.resolve("https://openfold.s3.amazonaws.com/pdb/y/missing.a3m").startswith("https://")


def test_deflate_raw_entries(tmp_path):
    import zlib

    from scigantic_msa.index import FileRef

    raw = b">q\nMKV\n>h\nMRV\n"
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    comp = c.compress(raw) + c.flush()
    blob = b"PADDING" + comp + b"TAIL"
    p = tmp_path / "z.bin"
    p.write_bytes(blob)
    idx = Index(str(tmp_path), cache_root=tmp_path / "c")
    f = FileRef("uniclust30", "a3m", str(p), 7, len(comp), "deflate-raw", len(raw))
    assert idx.fetch(f, use_cache=False) == raw
    bad = FileRef("uniclust30", "a3m", str(p), 7, len(comp), "deflate-raw", len(raw) + 1)
    with pytest.raises(IndexIntegrityError, match="decompressed"):
        idx.fetch(bad, use_cache=False)


def test_by_length_and_sequences(index_setup):
    idx, _, _ = index_setup
    got = idx.by_length(len(Q_6T33))
    assert (key(Q_6T33), Q_6T33) in got
    assert idx.by_length(99999) == []
    assert idx.sequences_fasta().exists()
