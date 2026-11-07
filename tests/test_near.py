import random
import shutil

import pytest

from conftest import Q_6T33, Q_LITEFOLD, key, real
from scigantic_msa.a3m import match_length, parse_a3m
from scigantic_msa.near import align, find_near, remap_rows


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def path_cost(q, p, path):
    cost = 0
    for qi, pj in path:
        if qi is None or pj is None:
            cost += 1
        elif q[qi] != p[pj]:
            cost += 1
    return cost


def mutate(s, rng, n):
    s = list(s)
    for _ in range(n):
        op = rng.choice("sid")
        i = rng.randrange(len(s))
        if op == "s":
            s[i] = rng.choice("ACDEFGHIKLMNPQRSTVWY")
        elif op == "i":
            s.insert(i, rng.choice("ACDEFGHIKLMNPQRSTVWY"))
        elif len(s) > 5:
            del s[i]
    return "".join(s)


def test_align_is_optimal_and_consistent():
    rng = random.Random(7)
    for _ in range(300):
        p = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(rng.randint(8, 60)))
        q = mutate(p, rng, rng.randint(0, 4))
        d = edit_distance(q, p)
        res = align(q, p, 4)
        if d > 4:
            assert res is None
            continue
        assert res is not None
        cost, path = res
        assert [i for i, _ in path if i is not None] == list(range(len(q)))
        assert [j for _, j in path if j is not None] == list(range(len(p)))
        assert path_cost(q, p, path) == cost
        if len(q) == len(p) and sum(a != b for a, b in zip(q, p)) <= 4:
            continue  # substitution-only path is accepted even when an indel pair would be cheaper
        assert cost == d


def test_remap_point_mutant_keeps_columns():
    rows = parse_a3m(real("ops-6t33_A/uniref90_hits.a3m"), Q_6T33)
    mut = Q_6T33[:5] + "W" + Q_6T33[6:]
    _, path = align(mut, Q_6T33, 2)
    new = remap_rows(rows, mut, path)
    assert new[0].seq == mut
    assert [r.seq for r in new[1:]] == [r.seq for r in rows[1:]]


def test_remap_insertion_and_deletion():
    rows = parse_a3m(">p\nMKTAYV\n>h1\nMKTAYV\n>h2\nMkkR-AYV\n>h3\nM-TAY-\n", "MKTAYV")
    q_ins = "MKTGAYV"  # G inserted after T
    _, path = align(q_ins, "MKTAYV", 2)
    new = remap_rows(rows, q_ins, path)
    assert new[0].seq == q_ins
    assert all(match_length(r.seq) == len(q_ins) for r in new)
    assert new[1].seq == "MKT-AYV"
    q_del = "MKAYV"  # T deleted
    _, path = align(q_del, "MKTAYV", 2)
    new = remap_rows(rows, q_del, path)
    assert all(match_length(r.seq) == len(q_del) for r in new)
    # The parent's T column becomes an insertion in the hits.
    assert new[1].seq == "MKtAYV"
    assert new[3].seq == "M-tAY-"
    assert new[2].seq == "MkkRAYV"  # a gap in a removed column just disappears


def test_remap_drops_rows_that_become_empty():
    rows = parse_a3m(">p\nMKV\n>h\n--V\n", "MKV")
    _, path = align("MK", "MKV", 1)
    new = remap_rows(rows, "MK", path)
    assert [r.header for r in new] == ["p"]


def test_find_near_by_length_point_mutant(index_setup, server):
    idx, _, _ = index_setup
    mut = Q_6T33[:20] + ("A" if Q_6T33[20] != "A" else "G") + Q_6T33[21:]
    hits = find_near(idx, mut, min_identity=0.95, method="by_length")
    assert hits and hits[0].parent_seq_sha256 == key(Q_6T33)
    assert hits[0].mismatches == [(21, Q_6T33[20], mut[20])]
    # Only by_length files were read: the server never saw the sequence.
    assert all(mut not in (r.path + r.body.decode("latin-1")) for r in server.requests)
    assert any("/by_length/" in r.path for r in server.requests)


def test_find_near_with_indel(index_setup):
    idx, _, _ = index_setup
    q = Q_LITEFOLD[:40] + Q_LITEFOLD[41:]
    hits = find_near(idx, q, min_identity=0.95, max_length_diff=3, method="by_length")
    assert hits and hits[0].parent_seq_sha256 == key(Q_LITEFOLD)
    assert hits[0].deletions == 1


def test_find_near_respects_identity_threshold(index_setup):
    idx, _, _ = index_setup
    far = "".join("A" if c != "A" else "G" for c in Q_6T33[:6]) + Q_6T33[6:]
    assert find_near(idx, far, min_identity=0.95, method="by_length") == []


@pytest.mark.mmseqs
@pytest.mark.skipif(not shutil.which("mmseqs"), reason="mmseqs not installed")
def test_find_near_mmseqs(index_setup):
    idx, _, _ = index_setup
    mut = Q_LITEFOLD[:50] + ("A" if Q_LITEFOLD[50] != "A" else "G") + Q_LITEFOLD[51:]
    hits = find_near(idx, mut, min_identity=0.95, method="mmseqs")
    assert hits and hits[0].parent_seq_sha256 == key(Q_LITEFOLD)
    assert hits[0].method == "mmseqs"
