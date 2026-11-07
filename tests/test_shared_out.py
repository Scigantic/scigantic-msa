"""Several complexes written into one folder, and complexes whose chains
share species with only some of the others.

Pair keys and paired row positions mean something only within one complex.
A second write into the same ``--out`` that shares a chain (A+B, then A+C,
or then A alone) must not rewrite a file the first complex points at. And a
chain that shares no species with the others (a viral partner, a designed
binder with one stray hit) must not cost the others their pairing.
"""
import json
from pathlib import Path

import pytest

from scigantic_msa import write_model
from scigantic_msa.a3m import Row, parse_a3m, parse_colabfold_complex
from scigantic_msa.adapters import AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix
from scigantic_msa.alignment import Alignment
from scigantic_msa.pairing import pair_by_species
from scigantic_msa.write import MODELS
from scigantic_msa.writers import chai_expected_basename, write_af3_json

A, B, C = "MKVLAGWERTLLPQSDE", "GHSEQKLLPAWMNRTYV", "PPWWGGYYSTRKDEQNL"


def syn(q, hits, tax):
    """An alignment whose hit i carries taxid tax[i] (or no species)."""
    rows = [Row("101", q, "query")]
    rows += [Row(f"UniRef100_{q[:3]}{i}", s, "uniref" if i in tax else "env", tax.get(i)) for i, s in enumerate(hits)]
    return Alignment(query=q, rows=rows, source="remote", provenance={})


def mut(q, k):
    return q[:k] + ("A" if q[k] != "A" else "G") + q[k + 1:]


@pytest.fixture
def abc():
    """A and B share 9606, 10090, 4932. C shares 9606 and 7227 with A only."""
    a = syn(A, [mut(A, i) for i in range(1, 6)], {0: "9606", 1: "10090", 2: "7227", 3: "562", 4: "4932"})
    b = syn(B, [mut(B, i) for i in range(1, 5)], {0: "10090", 1: "9606", 2: "4932", 3: "83333"})
    c = syn(C, [mut(C, i) for i in range(1, 4)], {0: "7227", 1: "9606"})
    return a, b, c


@pytest.fixture
def stray():
    """A and B share 9606, 10090, 4932; C has one hit with a taxid nobody
    shares, and two without species."""
    a = syn(A, [mut(A, i) for i in range(1, 6)], {0: "9606", 1: "10090", 2: "7227", 3: "562", 4: "4932"})
    b = syn(B, [mut(B, i) for i in range(1, 5)], {0: "10090", 1: "9606", 2: "4932", 3: "83333"})
    c = syn(C, [mut(C, i) for i in range(1, 4)], {2: "11111"})
    return a, b, c


def _taxa(*ms):
    return {r.seq: r.taxid for m in ms for r in m.rows}


def _yaml_files(path):
    return [Path(json.loads(l.split("msa: ", 1)[1])) for l in Path(path).read_text().splitlines() if "msa: " in l]


def _referenced(model, out, name, files):
    """Every file the job ``name`` points at, with its content."""
    d = out / model
    paths = [Path(files[0])]
    if model == "boltz":
        paths += _yaml_files(files[0])
    elif model == "chai":
        seqs = [l for l in Path(files[0]).read_text().splitlines() if not l.startswith(">")]
        paths += [Path(files[1]) / chai_expected_basename(s) for s in seqs]
    elif model == "openfold3":
        for ch in json.loads(Path(files[0]).read_text())["queries"][name]["chains"]:
            for k in ("main_msa_file_paths", "paired_msa_file_paths"):
                if k in ch:
                    paths += sorted(Path(ch[k]).iterdir())
    elif model == "protenix":
        for ent in json.loads(Path(files[0]).read_text())[0]["sequences"]:
            pc = ent["proteinChain"]
            paths += [Path(pc[k]) for k in ("unpairedMsaPath", "pairedMsaPath") if k in pc]
    assert all(p.is_file() for p in paths), d
    return {str(p): p.read_bytes() for p in paths}


# 1: every writer. The A+B job reads the same bytes after A+C, A alone and
# B+A are written into the same --out under other names.
@pytest.mark.parametrize("model", MODELS)
def test_later_writes_into_one_out_leave_the_first_complex_alone(model, abc, tmp_path):
    if model == "chai":
        pytest.importorskip("pyarrow")
    a, b, c = abc
    files, _ = write_model(model, [a, b], tmp_path, name="ab", quiet=True)
    before = _referenced(model, tmp_path, "ab", files)
    for name, chains in (("ac", [a, c]), ("a", [a]), ("ba", [b, a]), ("abc", [a, b, c])):
        write_model(model, chains, tmp_path, name=name, quiet=True)
    assert _referenced(model, tmp_path, "ab", files) == before


# 1a: Boltz pair keys are pair indexes local to one complex. Writing A+B,
# then A+C, rewrote A's CSV with A+C keys, so the A+B YAML paired A's 9606
# row with B's 10090 row, silently.
def test_boltz_second_complex_sharing_a_chain_keeps_first_pairing(abc, tmp_path):
    a, b, c = abc
    files, _ = write_model("boltz", [a, b], tmp_path, name="ab", quiet=True)
    write_model("boltz", [a, c], tmp_path, name="ac", quiet=True)
    tax = _taxa(a, b, c)

    def keys(f):
        lines = f.read_text().splitlines()[1:]
        return {k: tax[s] for k, s in (line.split(",", 1) for line in lines) if k != "-1"}

    ka, kb = (keys(f) for f in _yaml_files(files[0]))
    assert kb and {k: ka.get(k) for k in kb} == kb, f"A+B YAML pairs A {ka} with B {kb}"


# 1b: Chai-1. A complex A+B, then the monomer A: A's .aligned.pqt was
# rewritten without pairing keys, and Chai-1 then paired only the query row.
def test_chai_monomer_write_keeps_complex_pairing(abc, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    a, b, _ = abc
    files, _ = write_model("chai", [a, b], tmp_path, name="ab", quiet=True)
    write_model("chai", [a], tmp_path, name="a", quiet=True)

    def keys(m):
        t = pq.read_table(Path(files[1]) / chai_expected_basename(m.query)).to_pydict()
        return {k for k in t["pairing_key"] if k}

    assert keys(a) & keys(b), "the A+B Chai job lost every pairing key"


def test_unpaired_files_are_still_shared(abc, tmp_path):
    """A file whose content depends only on its sequence keeps the plain
    name, so writes share it."""
    a, b, c = abc
    a2 = syn(A, [mut(A, i) for i in range(1, 4)], {})
    write_model("boltz", [a2, c], tmp_path, name="x", quiet=True)
    write_model("boltz", [a2, b], tmp_path, name="y", quiet=True)
    csvs = sorted(p.name for p in (tmp_path / "boltz" / "msas").glob("*.csv"))
    assert csvs == sorted({f"{m.seq_sha256}.csv" for m in (a2, b, c)})
    pytest.importorskip("pyarrow")
    assert Chai.prepare([a2, c], out_dir=tmp_path / "chai", quiet=True) == (tmp_path / "chai").resolve()


def test_af3_files_paired_for_one_complex_get_their_own_name(abc, tmp_path):
    a, b, c = abc
    p1, p2 = tmp_path / "ab.json", tmp_path / "ac.json"
    write_af3_json([a, b], p1, inline=False)
    first = {e["protein"]["sequence"]: (tmp_path / e["protein"]["unpairedMsaPath"]).read_text()
             for e in json.loads(p1.read_text())["sequences"]}
    write_af3_json([a, c], p2, inline=False)
    write_af3_json([a], tmp_path / "a.json", inline=False)
    doc = json.loads(p1.read_text())
    for e in doc["sequences"]:
        assert (tmp_path / e["protein"]["unpairedMsaPath"]).read_text() == first[e["protein"]["sequence"]]
    mono = json.loads((tmp_path / "a.json").read_text())["sequences"][0]["protein"]
    assert mono["unpairedMsaPath"] == f"{a.seq_sha256}.unpaired.a3m"


# 2: a species in every chain was required, so a third chain from another
# organism removed all pairing for A and B.
def test_unshared_third_chain_does_not_cost_the_others_their_pairing(stray):
    a, b, c = stray
    assert pair_by_species([a, b]).n_paired == 3
    pr = pair_by_species([a, b, c], names=["A", "B", "C"])
    assert pr.n_paired == 3 and pr.rows[2] == [None] * 3
    assert "across chains A, B" in pr.message
    assert "chain C shares no species with another chain" in pr.message and "not paired" in pr.message


def test_species_shared_by_some_chains_are_paired(abc):
    a, b, c = abc
    pr = pair_by_species([a, b, c], names=["A", "B", "C"])
    sp = [[r.taxid if r else None for r in rows] for rows in pr.rows]
    # The species all three share first, then those two chains share.
    assert sp[0][0] == sp[1][0] == sp[2][0] == "9606"
    assert set(list(zip(*sp))[1:]) == {("10090", "10090", None), ("4932", "4932", None), ("7227", None, "7227")}
    assert "1 with chains A, B, C" in pr.message and "2 with chains A, B" in pr.message
    assert "1 with chains A, C" in pr.message


def _species_rows(model, chains, tmp_path):
    """For each paired row a model would read, the species of each chain's
    row in it (None for a gap or a row of the chain's own)."""
    tax = _taxa(*chains)
    if model == "colabfold":
        p = ColabFold.prepare(list(chains), out_dir=tmp_path, quiet=True)
        lens, _, _, records = parse_colabfold_complex(p.read_text())
        out = []
        for _, s in records[1:]:
            segs, k = [], 0
            for n in lens:
                segs.append(s[k:k + n])
                k += n
            if sum(any(x.isalpha() for x in seg) for seg in segs) > 1:
                out.append(tuple(tax.get(seg) for seg in segs))
        return out
    if model == "openfold3":
        q = OpenFold3.prepare(list(chains), out_dir=tmp_path, quiet=True)
        cols = [open(ch["paired_msa_file_paths"] + "/colabfold_paired.a3m").read().splitlines()[3::2]
                for ch in q["queries"]["query"]["chains"]]
        return [tuple(tax.get(s) for s in row) for row in zip(*cols)]
    if model == "af3":
        # A chain's own row standing in for a pair is not a species of the
        # pair: count only species two chains have.
        held = [{r.taxid for r in m.rows if r.taxid} for m in chains]
        tax = {s: t for s, t in tax.items() if sum(t in h for h in held) > 1}
        doc = AlphaFold3.prepare(list(chains), quiet=True)
        cols = [[r.seq for r in parse_a3m(e["protein"]["unpairedMsa"], e["protein"]["sequence"])[1:]]
                for e in doc["sequences"]]
        n = pair_by_species(list(chains)).n_paired
        return [tuple(tax.get(col[k]) if k < len(col) else None for col in cols) for k in range(n)]
    if model == "boltz":
        files = [Path(json.loads(l.split("msa: ", 1)[1])) for l in
                 Boltz.prepare(list(chains), out_dir=tmp_path, quiet=True).splitlines() if "msa: " in l]
        keyed = [{k: s for k, s in (l.split(",", 1) for l in f.read_text().splitlines()[2:]) if k != "-1"}
                 for f in files]
        keys = sorted({k for d in keyed for k in d}, key=int)
        return [tuple(tax.get(d[k]) if k in d else None for d in keyed) for k in keys]
    if model == "chai":
        pq = pytest.importorskip("pyarrow.parquet")
        d = Chai.prepare(list(chains), out_dir=tmp_path, quiet=True)
        tabs = [pq.read_table(d / chai_expected_basename(m.query)).to_pydict() for m in chains]
        keyed = [{k: s for k, s in zip(t["pairing_key"], t["sequence"]) if k} for t in tabs]
        # Chai pairs a key only when every chain has it.
        assert len({frozenset(d) for d in keyed}) == 1
        return [tuple(tax.get(d[k]) for d in keyed) for k in sorted(keyed[0], key=int)]
    raise AssertionError(model)


@pytest.mark.parametrize("model", ["af3", "boltz", "chai", "colabfold", "openfold3"])
def test_every_writer_pairs_the_two_chains_next_to_a_stray_one(model, stray, tmp_path):
    rows = _species_rows(model, stray, tmp_path)
    assert sorted(r[:2] for r in rows) == [("10090", "10090"), ("4932", "4932"), ("9606", "9606")]
    assert all(r[2] is None for r in rows)


@pytest.mark.parametrize("model", ["af3", "boltz", "chai", "colabfold", "openfold3"])
def test_every_writer_pairs_species_some_chains_share(model, abc, tmp_path):
    rows = _species_rows(model, abc, tmp_path)
    assert len(rows) == 4
    for r in rows:
        # One species per row, in at least two chains.
        assert len(set(r) - {None}) == 1 and sum(x is not None for x in r) >= 2
    assert {next(x for x in r if x) for r in rows} == {"9606", "10090", "4932", "7227"}


def test_af3_stands_in_own_rows_never_gap_rows(abc):
    """AlphaFold 3 joins unpaired rows by index, so a chain without a pair's
    species gets one of its own rows there, not a gap row that would count
    in its profile."""
    a, b, c = abc
    doc = AlphaFold3.prepare([a, b, c], quiet=True)
    for e, m in zip(doc["sequences"], (a, b, c)):
        rows = parse_a3m(e["protein"]["unpairedMsa"], e["protein"]["sequence"])
        assert all(any(x.isalpha() for x in r.seq) for r in rows)
        # Every row of the chain is still there, once.
        assert sorted(r.seq for r in rows) == sorted(r.seq for r in m.rows)


@pytest.mark.parametrize("adapter", [AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix])
def test_every_adapter_says_which_chains_paired(adapter, stray, tmp_path, capsys):
    if adapter is Chai:
        pytest.importorskip("pyarrow")
    kw = {} if adapter is AlphaFold3 else {"out_dir": tmp_path}
    adapter.prepare(list(stray), **kw)
    err = capsys.readouterr().err
    line = [l for l in err.splitlines() if "chain C" in l]
    assert len(line) == 1, err
    if adapter is Protenix:
        assert "no pairing.a3m" in line[0]
    else:
        assert "paired 3 rows by species across chains A, B" in line[0] and "not paired" in line[0]
