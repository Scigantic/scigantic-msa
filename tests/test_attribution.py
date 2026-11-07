"""The attribution file keeps every write's credit when writes share a
folder."""
import json
import threading

import pytest

from conftest import REAL
from scigantic_msa import MSA
from scigantic_msa.adapters import AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix
from scigantic_msa.cli import MODELS, main
from scigantic_msa.writers import ATTRIBUTION_FILE

A = REAL / "of3-6t33_A" / "cfdb_uniref30.a3m"
B = REAL / "of3-1hvw_A" / "cfdb_uniref30.a3m"


def credited(seq_a3m, cid):
    m = MSA().from_a3m(seq_a3m)
    m.attribution = [{"id": cid, "license": f"{cid} terms"}]
    return m


def side(d):
    return json.loads((d / ATTRIBUTION_FILE).read_text())


def shas(doc):
    return [sorted(c["seq_sha256"] for c in e["chains"]) for e in doc["outputs"]]


@pytest.mark.parametrize("first,second", [
    (lambda m, d: Boltz.prepare(m, out_dir=d, quiet=True), lambda m, d: Chai.prepare(m, out_dir=d, quiet=True)),
    (lambda m, d: AlphaFold3.prepare(m, name="x", out_dir=d, quiet=True),
     lambda m, d: ColabFold.prepare(m, out_dir=d, name="y", quiet=True)),
    (lambda m, d: OpenFold3.prepare(m, out_dir=d, quiet=True), lambda m, d: Protenix.prepare(m, out_dir=d, quiet=True)),
    (lambda m, d: Protenix.prepare(m, out_dir=d, name="x", quiet=True),
     lambda m, d: Protenix.prepare(m, out_dir=d, name="y", quiet=True)),
])
def test_two_adapter_calls_into_one_folder_keep_both_credits(tmp_path, first, second):
    pytest.importorskip("pyarrow")
    a, b = credited(A, "src-a"), credited(B, "src-b")
    first(a, tmp_path)
    second(b, tmp_path)
    doc = side(tmp_path)
    assert shas(doc) == [[a.seq_sha256], [b.seq_sha256]]
    assert {s["id"] for s in doc["sources"]} == {"src-a", "src-b"}


@pytest.mark.parametrize("model", MODELS)
def test_convert_twice_with_other_names_keeps_both(tmp_path, model):
    pytest.importorskip("pyarrow")
    for f, n in ((A, "one"), (B, "two")):
        assert main(["convert", str(f), "-m", model, "-o", str(tmp_path), "--name", n, "-q"]) == 0
    sidecars = list(tmp_path.rglob(ATTRIBUTION_FILE))
    assert len(sidecars) == 1
    assert len(json.loads(sidecars[0].read_text())["outputs"]) == 2


@pytest.mark.parametrize("model", MODELS)
def test_the_same_write_twice_is_one_entry(tmp_path, model):
    pytest.importorskip("pyarrow")
    for _ in range(2):
        assert main(["convert", str(A), "-m", model, "-o", str(tmp_path), "--name", "one", "-q"]) == 0
    (f,) = tmp_path.rglob(ATTRIBUTION_FILE)
    assert len(json.loads(f.read_text())["outputs"]) == 1


def test_a_write_that_leaves_an_older_file_keeps_its_credit(tmp_path):
    a, b = credited(A, "src-a"), credited(B, "src-b")
    Boltz.prepare([a, b], out_dir=tmp_path, quiet=True)
    # A rerun of chain a alone rewrites only a's file; b's is still there.
    Boltz.prepare(credited(A, "src-a2"), out_dir=tmp_path, quiet=True)
    doc = side(tmp_path)
    assert len(doc["outputs"]) == 2
    assert {s["id"] for s in doc["sources"]} == {"src-a", "src-b", "src-a2"}


def test_an_attribution_file_from_0_2_3_is_kept(tmp_path):
    old = {"written_by": "scigantic-msa", "chains": [{"seq_sha256": "x", "attribution": [{"id": "old"}]}],
           "files": {"x": "x.a3m"}, "notes": ["n"]}
    (tmp_path / ATTRIBUTION_FILE).write_text(json.dumps(old))
    Boltz.prepare(credited(A, "new"), out_dir=tmp_path, quiet=True)
    doc = side(tmp_path)
    assert doc["outputs"][0] == {"chains": old["chains"], "files": old["files"], "notes": ["n"]}
    assert [s["id"] for s in doc["sources"]] == ["old", "new"]


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", "\xff"])
def test_an_unreadable_attribution_file_is_replaced(tmp_path, text):
    (tmp_path / ATTRIBUTION_FILE).write_text(text, encoding="latin-1")
    Boltz.prepare(credited(A, "new"), out_dir=tmp_path, quiet=True)
    assert len(side(tmp_path)["outputs"]) == 1


def test_writes_at_the_same_time_all_land(tmp_path):
    rows = [ln for ln in A.read_text().splitlines()]
    query = rows[1]
    jobs = []
    for i in range(8):
        # Eight distinct sequences, each its own alignment.
        q = query[:-1] + "ACDEFGHI"[i]
        jobs.append(credited(f">101\n{q}\n", f"s{i}"))
    errs = []

    def go(m):
        try:
            Boltz.prepare(m, out_dir=tmp_path, quiet=True)
        except Exception as e:  # pragma: no cover - reported below
            errs.append(e)

    ts = [threading.Thread(target=go, args=(m,)) for m in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    doc = side(tmp_path)
    assert len(doc["outputs"]) == 8 and len(doc["sources"]) == 8
    assert not list(tmp_path.glob("*.lock"))
