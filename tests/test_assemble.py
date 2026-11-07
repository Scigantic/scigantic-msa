import pytest

from conftest import Q_1HVW, Q_6T33, Q_LITEFOLD, key, real, uniclust_query
from scigantic_msa import assemble
from scigantic_msa.a3m import match_length, parse_a3m, parse_stockholm
from scigantic_msa.assemble import EntryUnusable, load_entry, order_entries


def _entry(idx, seq, source):
    return next(e for e in idx.lookup(key(seq)) if e.source == source)


def test_a3m_purpose_prefers_of3_cfdb_over_ops(index_setup):
    idx, _, _ = index_setup
    ents = order_entries(idx.lookup(key(Q_6T33)), "a3m")
    assert ents[0].source == "openfold3-pdb"
    msa = load_entry(idx, ents[0], Q_6T33)
    assert msa.provenance["roles"] == ["cfdb_uniref30"]
    assert msa.depth == len(parse_a3m(real("of3-6t33_A/cfdb_uniref30.a3m")))
    assert msa.source == "exact"
    assert msa.attribution[0]["license"] == "CC-BY-4.0"


def test_af3_purpose_merges_cfdb_uniref90_mgnify(index_setup):
    idx, _, _ = index_setup
    msa = load_entry(idx, _entry(idx, Q_6T33, "openfold3-pdb"), Q_6T33, purpose="af3")
    assert msa.provenance["roles"] == ["cfdb_uniref30", "uniref90", "mgnify"]
    n_cfdb = len(parse_a3m(real("of3-6t33_A/cfdb_uniref30.a3m")))
    n_u90 = len(parse_stockholm(real("of3-6t33_A/uniref90_hits.sto").split(b"\n")))
    n_mg = len(parse_stockholm(real("of3-6t33_A/mgnify_hits.sto").split(b"\n")))
    assert n_cfdb < msa.depth <= n_cfdb + n_u90 + n_mg - 2
    assert msa.rows[0].seq == Q_6T33
    # Rows are unique after the merge.
    from scigantic_msa.a3m import ungapped

    keys = [ungapped(r.seq) for r in msa.rows]
    assert len(keys) == len(set(keys))


def test_ops_entry_merges_three_files_query_first(index_setup):
    idx, _, _ = index_setup
    msa = load_entry(idx, _entry(idx, Q_6T33, "openproteinset-pdb"), Q_6T33)
    assert msa.provenance["roles"] == ["uniref90", "bfd_uniclust", "mgnify"]
    assert msa.rows[0].seq == Q_6T33
    assert {r.db for r in msa.rows[1:]} >= {"uniref90", "mgnify"}
    assert all(match_length(r.seq) == len(Q_6T33) for r in msa.rows)


def test_uniprot_rows_for_pairing(index_setup):
    idx, _, _ = index_setup
    msa = load_entry(idx, _entry(idx, Q_1HVW, "openfold3-pdb"), Q_1HVW, want_uniprot=True)
    assert msa.uniprot_rows and msa.uniprot_rows[0].seq == Q_1HVW
    assert all(r.db == "uniprot" for r in msa.uniprot_rows[1:])
    assert sum(1 for r in msa.uniprot_rows if r.taxid) > 10


def test_litefold_and_uniclust(index_setup):
    idx, _, _ = index_setup
    m = load_entry(idx, idx.lookup(key(Q_LITEFOLD))[0], Q_LITEFOLD)
    assert m.provenance["roles"] == ["colabfold"] and m.depth > 300
    q = uniclust_query()
    u = load_entry(idx, idx.lookup(key(q))[0], q)
    assert u.provenance["roles"] == ["uniclust30"] and u.depth == 200


def test_wrong_bytes_behind_an_entry_are_unusable_not_a_hit(index_setup, server):
    idx, _, _ = index_setup
    e = _entry(idx, Q_6T33, "openproteinset-pdb")
    f = e.file("uniref90")
    path = f.url.split(str(server.port), 1)[1]
    other = real("of3-1hvw_A/cfdb_uniref30.a3m")
    server.static[path] = (other * 10)[: f.length]  # right length, wrong alignment
    with pytest.raises(EntryUnusable, match="does not match the query"):
        load_entry(idx, e, Q_6T33)


def test_short_file_behind_an_entry_is_unusable(index_setup, server):
    idx, _, _ = index_setup
    e = _entry(idx, Q_6T33, "openproteinset-pdb")
    path = e.file("mgnify").url.split(str(server.port), 1)[1]
    server.static[path] = server.static[path][:-10]
    with pytest.raises(EntryUnusable, match="wanted"):
        load_entry(idx, e, Q_6T33)


def test_big_stockholm_path_matches_the_small_path(index_setup, monkeypatch):
    idx, _, _ = index_setup
    e = _entry(idx, Q_6T33, "openfold3-pdb")
    small = load_entry(idx, e, Q_6T33, want_uniprot=True).uniprot_rows
    calls = []
    orig = assemble._first_line_after

    def spy(*a, **kw):
        calls.append(a[2])
        return orig(*a, **kw)

    monkeypatch.setattr(assemble, "STREAM_THRESHOLD", 100)
    monkeypatch.setattr(assemble, "_first_line_after", spy)
    # Make the binary search run at a 512 byte resolution on this small file.
    monkeypatch.setattr(assemble, "_big_stockholm_lines", _with_resolution(assemble._big_stockholm_lines, 512))
    idx.files.root = idx.files.root.parent / "files-big"  # no cached bytes
    big = load_entry(idx, e, Q_6T33, want_uniprot=True).uniprot_rows
    assert calls, "binary search was not used"
    assert [(r.header, r.seq, r.taxid) for r in big] == [(r.header, r.seq, r.taxid) for r in small]


def _with_resolution(fn, res):
    import inspect

    src = inspect.getsource(fn).replace("65536", str(res))
    ns = dict(fn.__globals__)
    exec(compile(src, "<patched>", "exec"), ns)
    return ns[fn.__name__]


def interleave(sto: bytes, cols: int) -> bytes:
    """Re-block a one-block Stockholm file the way jackhmmer writes wide ones:
    fixed-width lines, a '#=GC RF' line and a blank line after each block."""
    lines = sto.decode().split("\n")
    gs = [l for l in lines if l.startswith("#=GS")]
    rows = [l.split() for l in lines if l and not l.startswith(("#", "//"))]
    width = max(len(n) for n, _ in rows) + 2
    total = len(rows[0][1])
    out = ["# STOCKHOLM 1.0", ""] + gs + [""]
    for c in range(0, total, cols):
        for n, a in rows:
            out.append(n.ljust(width) + a[c : c + cols])
        out.append("#=GC RF".ljust(width) + "x" * len(rows[0][1][c : c + cols]))
        out.append("")
    out.append("//")
    return ("\n".join(out) + "\n").encode()


def test_interleaved_stockholm_small_and_big_paths_agree(index_setup, server, monkeypatch):
    idx, _, _ = index_setup
    e = _entry(idx, Q_6T33, "openfold3-pdb")
    f = e.file("uniprot")
    path = f.url.split(str(server.port), 1)[1]
    one_block = server.static[path]
    want = parse_stockholm(one_block.split(b"\n"), Q_6T33)
    blocked = interleave(one_block, 10)
    assert blocked.count(b"#=GC RF") == 5
    server.static[path] = blocked
    f.length = len(blocked)
    got_small = load_entry(idx, e, Q_6T33, want_uniprot=True).uniprot_rows
    assert [(r.header, r.seq) for r in got_small] == [(r.header, r.seq) for r in want]
    capped = parse_stockholm(blocked.split(b"\n"), Q_6T33, max_rows=7)
    assert [(r.header, r.seq) for r in capped] == [(r.header, r.seq) for r in want[:7]]
    monkeypatch.setattr(assemble, "STREAM_THRESHOLD", 100)
    monkeypatch.setattr(assemble, "ROW_CAPS", dict(assemble.ROW_CAPS, uniprot=9))
    monkeypatch.setattr(assemble, "_big_stockholm_lines", _with_resolution(assemble._big_stockholm_lines, 256))
    idx.files.root = idx.files.root.parent / "files-big2"
    got_big = load_entry(idx, e, Q_6T33, want_uniprot=True).uniprot_rows
    assert [(r.header, r.seq, r.taxid) for r in got_big] == [(r.header, r.seq, r.taxid) for r in want[:9]]


def test_broken_uniprot_file_keeps_the_msa(index_setup, server):
    idx, _, _ = index_setup
    e = _entry(idx, Q_6T33, "openfold3-pdb")
    path = e.file("uniprot").url.split(str(server.port), 1)[1]
    server.static[path] = b"x" * len(server.static[path])
    m = load_entry(idx, e, Q_6T33, want_uniprot=True)
    assert m.uniprot_rows is None and "STOCKHOLM" in m.provenance["uniprot_error"]
    assert m.depth > 1
