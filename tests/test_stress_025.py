"""Regression tests for defects found by a stress test of 0.2.5.

Self-contained and offline. Each test names the defect it covers.
"""
import json
import shlex
from pathlib import Path

import pytest

from scigantic_msa import write_model
from scigantic_msa.a3m import Row, parse_a3m
from scigantic_msa.adapters import AlphaFold3, Boltz, _chain_names, _distinct
from scigantic_msa.alignment import Alignment
from scigantic_msa.cli import main
from scigantic_msa.pairing import pair_by_species
from scigantic_msa.remote.colabfold import DEFAULT_HOST, ColabFoldClient, public_host
from scigantic_msa.write import MODELS
from scigantic_msa.writers import BOLTZ_MAX_PAIRED, af3_layout, write_af3_json, write_boltz

QA = "MKVLAGWERTLLPQSDEHHKAYLLPWQ"
QB = "GHSEQKLLPAWMNRTYVKKDDEFGMLA"
QC = "PPWWGGYYSTRKDEQNLMAGHKKLWEE"


def mut(q, k, to="A"):
    return q[:k] + (to if q[k] != to else "G") + q[k + 1:]


def aln(q, taxids, tag, extra=0):
    """Hit i has taxid taxids[i]; ``extra`` more hits have no species."""
    rows = [Row("101", q, "query")]
    for i, t in enumerate(taxids):
        rows.append(Row(f"UniRef100_{tag}{i}\t{t}", mut(q, 2 + i), "uniref", t))
    for j in range(extra):
        rows.append(Row(f"env_{tag}{j}", mut(q, 2 + len(taxids) + j, "W"), "env"))
    return Alignment(query=q, rows=rows, source="remote", provenance={})


def af3_rows(text):
    """The rows of an a3m as AlphaFold 3 reads a user MSA: every record in
    order, gap-only rows included (parsers.parse_fasta, deduplicate=False)."""
    return [rec.split("\n", 1)[1].replace("\n", "") for rec in text.split(">")[1:]]


def check_af3_alignment(doc, chains, pr):
    """Every paired row of every chain is at one row index, the same in
    every chain that has that pair."""
    rows = {p["protein"]["sequence"]: af3_rows(p["protein"]["unpairedMsa"]) for p in doc["sequences"]}
    for k in range(pr.n_paired):
        at = set()
        for ci, m in enumerate(chains):
            r = pr.rows[ci][k]
            if r is not None:
                assert rows[m.query].count(r.seq) == 1
                at.add(rows[m.query].index(r.seq))
        assert len(at) == 1, f"pair {k} sits at rows {sorted(at)}"
    return rows


# ---------------------------------------------------------------- defect 1


def test_af3_shallow_chain_paired_row_stays_at_its_index(tmp_path):
    """Defect 1. A chain without a row of the first pair's species and no
    rows of its own left got its later paired rows right after the query, so
    they stacked against other chains' rows of other species."""
    a = aln(QA, ["1", "2"], "a")
    b = aln(QB, ["1", "2", "3"], "b")
    c = aln(QC, ["3"], "c")  # shares only species 3, with B
    chains = [a, b, c]
    pr = pair_by_species(chains)
    assert pr.keys == ["tx:1", "tx:2", "tx:3"]
    write_model("af3", chains, tmp_path, name="j", quiet=True)
    doc = json.loads((tmp_path / "af3" / "j.json").read_text())
    rows = check_af3_alignment(doc, chains, pr)
    tax_of = {r.seq: r.taxid for m in chains for r in m.rows[1:]}
    paired = [{r.seq for r in pr.rows[ci] if r is not None} for ci in range(3)]
    for i in range(1, max(len(v) for v in rows.values())):
        species = {tax_of[rows[m.query][i]] for ci, m in enumerate(chains)
                   if i < len(rows[m.query]) and rows[m.query][i] in paired[ci]}
        assert len(species) <= 1, f"row {i} stacks paired rows of species {species}"
    # Every row of every chain is still there.
    for m in chains:
        assert {r.seq for r in m.rows} <= set(rows[m.query])


def test_af3_needs_as_few_gap_rows_as_it_can_and_says_so(capsys):
    """Defect 1. Here one gap row is the fewest any order needs (A and C
    have no rows to spare, and each lacks a pair the other has), and the
    notice names the chain and says every pair stays in place."""
    a, b, c = aln(QA, ["1", "2"], "a"), aln(QB, ["1", "2", "3"], "b"), aln(QC, ["3"], "c")
    lay = af3_layout([a, b, c], pair_by_species([a, b, c]), ["A", "B", "C"])
    assert sum(lay.gaps.values()) == 1 and lay.n_paired == 3
    doc = AlphaFold3.prepare([a, b, c])
    err = capsys.readouterr().err
    assert "AlphaFold 3 gets the 3 paired rows at the same row" in err
    assert "gap rows instead (chain A: 1)" in err and "All 3 paired rows stay at their own row" in err
    gaps = [s for e in doc["sequences"] for s in af3_rows(e["protein"]["unpairedMsa"]) if set(s) == {"-"}]
    assert len(gaps) == 1


def test_af3_reorders_pairs_so_a_shallow_chain_needs_no_gap_rows(tmp_path):
    """Defect 1. A and B are deep and share many species; C shares one with
    B and has nothing to spare. Its pair goes first, so C needs no stand-in
    and A, which has rows to spare, stands in once."""
    a = aln(QA, [str(t) for t in range(1, 9)], "a", extra=3)
    b = aln(QB, [str(t) for t in range(1, 9)] + ["99"], "b", extra=3)
    c = aln(QC, ["99"], "c")
    chains = [a, b, c]
    pr = pair_by_species(chains)
    lay = af3_layout(chains, pr)
    assert lay.gaps == {} and lay.note() == ""
    doc = write_af3_json(chains, tmp_path / "j.json")
    check_af3_alignment(doc, chains, pr)
    side = json.loads((tmp_path / "scigantic-msa-attribution.json").read_text())
    assert any("the 9 rows paired by taxid" in n for n in side["outputs"][0]["notes"])


def test_af3_path_files_keep_the_alignment(tmp_path):
    """Defect 1, inline=False: the unpaired files hold the same layout."""
    a, b, c = aln(QA, ["1", "2"], "a"), aln(QB, ["1", "2", "3"], "b"), aln(QC, ["3"], "c")
    doc = write_af3_json([a, b, c], tmp_path / "j.json", inline=False)
    for e in doc["sequences"]:
        e["protein"]["unpairedMsa"] = (tmp_path / e["protein"].pop("unpairedMsaPath")).read_text()
    check_af3_alignment(doc, [a, b, c], pair_by_species([a, b, c]))


def test_af3_keeps_rows_that_differ_only_in_insertions():
    """AlphaFold 3 reads a user MSA with deduplicate=False, so a row that
    matches another once insertions are removed is not a repeat to it. Such
    rows were dropped from a chain with paired rows."""
    ins = QA[:5] + "k" + QA[5:]
    ins = mut(ins, 9)
    plain = mut(QA, 8)
    assert ins.replace("k", "") == plain
    a = Alignment(query=QA, rows=[Row("101", QA, "query"), Row("e", ins, "env"),
                                  Row("UniRef100_a\t9606", plain, "uniref", "9606")], source="remote", provenance={})
    b = aln(QB, ["9606"], "b")
    doc = AlphaFold3.prepare([a, b], quiet=True)
    rows = af3_rows(doc["sequences"][0]["protein"]["unpairedMsa"])
    assert sorted(rows) == sorted(r.seq for r in a.rows)
    check_af3_alignment(doc, [a, b], pair_by_species([a, b]))


# ---------------------------------------------------------------- defect 2


def second(q):
    return Alignment(query=q, rows=[Row("101", q, "query"), Row("other", mut(q, 10, "W"), "uniref")],
                     source="remote", provenance={})


def _job_files(model, files, nxt):
    """Every file a job's input points at, with its content."""
    paths = []
    if model == "boltz":
        paths = [Path(json.loads(l.split("msa: ", 1)[1])) for l in Path(files[0]).read_text().splitlines()
                 if "msa: " in l]
    elif model == "chai":
        d = Path(shlex.split(nxt)[shlex.split(nxt).index("--msa-directory") + 1])
        assert d == Path(files[1])
        paths = sorted(d.glob("*.pqt"))
    elif model == "openfold3":
        for q in json.loads(Path(files[0]).read_text())["queries"].values():
            for ch in q["chains"]:
                for k in ("main_msa_file_paths", "paired_msa_file_paths"):
                    if k in ch:
                        paths += sorted(Path(ch[k]).iterdir())
    elif model == "protenix":
        for ent in json.loads(Path(files[0]).read_text())[0]["sequences"]:
            pc = ent["proteinChain"]
            paths += [Path(pc[k]) for k in ("unpairedMsaPath", "pairedMsaPath") if k in pc]
    paths.append(Path(files[0]))
    assert paths and all(p.is_file() for p in paths)
    return {str(p): p.read_bytes() for p in paths}


@pytest.mark.parametrize("model", MODELS)
def test_second_alignment_of_same_sequence_keeps_first_jobs_msa(tmp_path, model):
    """Defect 2. A second job in the same --out with another alignment of
    the same sequence rewrote the first job's MSA for Boltz, Chai-1,
    OpenFold3 and Protenix."""
    if model == "chai":
        pytest.importorskip("pyarrow")
    files_x, nxt_x = write_model(model, [aln(QA, ["1"], "first")], tmp_path, name="x", quiet=True)
    before = {p: p.read_bytes() for p in Path(tmp_path, model).rglob("*")
              if p.is_file() and "attribution" not in p.name}
    x = _job_files(model, files_x, nxt_x)
    files_y, nxt_y = write_model(model, [second(QA)], tmp_path, name="y", quiet=True)
    changed = [str(p.relative_to(tmp_path)) for p, b in before.items() if p.read_bytes() != b]
    assert not changed, f"job x's files rewritten by job y: {changed}"
    assert _job_files(model, files_x, nxt_x) == x
    y = _job_files(model, files_y, nxt_y)
    assert set(x) != set(y)


@pytest.mark.parametrize("model", ["boltz", "chai", "openfold3", "protenix"])
def test_identical_alignments_still_share_their_files(tmp_path, model):
    """Defect 2. Content-addressed names: the same content is one file."""
    if model == "chai":
        pytest.importorskip("pyarrow")
    files_x, nxt_x = write_model(model, [aln(QA, ["1"], "a"), aln(QB, ["1"], "b")], tmp_path, name="x", quiet=True)
    files_y, nxt_y = write_model(model, [aln(QA, ["1"], "a"), aln(QB, ["1"], "b")], tmp_path, name="y", quiet=True)
    x, y = _job_files(model, files_x, nxt_x), _job_files(model, files_y, nxt_y)
    assert set(x) - {files_x[0]} == set(y) - {files_y[0]}


def test_convert_twice_into_one_out_keeps_both_msas(tmp_path, capsys):
    """Defect 2 from the command line: convert one.a3m --name x, then
    convert two.a3m --name y. Each printed next command reads its own MSA."""
    one, two = tmp_path / "one.a3m", tmp_path / "two.a3m"
    one.write_text(aln(QA, ["1"], "a").a3m())
    two.write_text(second(QA).a3m())
    out = tmp_path / "out"
    for f, name in ((one, "x"), (two, "y")):
        assert main(["convert", "-m", "boltz", "-o", str(out), "--name", name, str(f)]) == 0
    err = capsys.readouterr().err
    yamls = [Path(shlex.split(l.split("next: ", 1)[1])[2]) for l in err.splitlines() if "next: boltz predict" in l]
    assert len(yamls) == 2
    msas = [Path(json.loads(l.split("msa: ", 1)[1])) for y in yamls for l in y.read_text().splitlines()
            if "msa: " in l]
    assert msas[0] != msas[1]
    assert msas[0].read_text() == one.read_text() and msas[1].read_text() == two.read_text()


# ---------------------------------------------------------------- defect 3


@pytest.mark.parametrize("host", ["https://api.colabfold.com//.", "http://api.colabfold.com//.",
                                  "https://api.colabfold.com//", "https://api.colabfold.com/.//./"])
def test_repeated_slashes_are_the_public_server(host):
    """Defect 3. The server answers //./ticket/msa with a redirect to
    /ticket/msa, so this spelling is the public server."""
    assert public_host(host)
    assert ColabFoldClient(host=host).host == DEFAULT_HOST


def test_repeated_slashes_before_a_path_are_not_the_public_server():
    assert not public_host("https://api.colabfold.com//proxy")


# ---------------------------------------------------------------- defect 4


def test_pairing_notice_counts_chains_not_distinct_sequences():
    """Defect 4. A, B, A, C (chain C repeats A) said 'across 3 chains' and
    named chains A, C, B, D."""
    a, b, c = aln(QA, ["1", "2"], "a"), aln(QB, ["1", "2"], "b"), aln(QC, ["1"], "c")
    chains = [a, b, a, c]
    msas = [m for m, _ in _distinct(chains)]
    msg = pair_by_species(msas, names=_chain_names(chains)).message
    assert "across 4 chains" in msg, msg
    assert "1 with chains A, B, C, D" in msg and "1 with chains A, B, C" in msg, msg


# ---------------------------------------------------------------- Boltz cap


def test_boltz_gets_at_most_8191_pair_keys(tmp_path):
    """Boltz pairs at most 8192 rows with the query (construct_paired_msa),
    and drops a keyed row it does not pair, so rows past 8191 pairs are
    written unkeyed."""
    n = BOLTZ_MAX_PAIRED + 5
    a = Alignment(query=QA, rows=[Row("101", QA, "query")] + [
        Row(f"UniRef100_a{i}\t{i}", "".join("ACDEFGHIKLMNPQRSTVWY"[(i // 20 ** j) % 20] for j in range(len(QA))),
            "uniref", str(i)) for i in range(n)], source="remote", provenance={})
    b = Alignment(query=QB, rows=[Row("101", QB, "query")] + [
        Row(f"UniRef100_b{i}\t{i}", "".join("ACDEFGHIKLMNPQRSTVWY"[(i // 20 ** j + 3) % 20] for j in range(len(QB))),
            "uniref", str(i)) for i in range(n)], source="remote", provenance={})
    for line in write_boltz([a, b], tmp_path):
        rows = Path(line.split("msa: ", 1)[1]).read_text().splitlines()[1:]
        keys = [r.split(",", 1)[0] for r in rows]
        assert keys[0] == "-1" and keys[1:BOLTZ_MAX_PAIRED] == [str(k) for k in range(1, BOLTZ_MAX_PAIRED)]
        assert set(keys[BOLTZ_MAX_PAIRED:]) == {"-1"} and len(rows) == n + 1
    assert Boltz.prepare([a, b], out_dir=tmp_path / "p", quiet=True)
