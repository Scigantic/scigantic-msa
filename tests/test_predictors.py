"""Load our outputs with the predictors' own code.

Needs interpreters that have the predictors installed, for example:
    uv venv vboltz && uv pip install -p vboltz/bin/python boltz
    uv venv vchai && uv pip install -p vchai/bin/python chai_lab
    SCIGANTIC_MSA_TEST_BOLTZ_PYTHON=vboltz/bin/python \\
    SCIGANTIC_MSA_TEST_CHAI_PYTHON=vchai/bin/python pytest -m predictors

For Protenix, SCIGANTIC_MSA_TEST_PROTENIX_PYTHON is an interpreter that can
import protenix's data pipeline (torch, biotite, rdkit, pdbeccdutils and the
like; the CUDA packages are not needed).
"""
import json
import os
import string
import subprocess
from pathlib import Path

import pytest

from conftest import REAL, Q_6T33, Q_7M22A, Q_7M22C, Q_LITEFOLD, make_result_tar, real_alignment
from scigantic_msa.adapters import Boltz, Chai, Protenix, protenix_species
from scigantic_msa.cli import main
from scigantic_msa.pairing import pair_by_species

HERE = Path(__file__).parent / "predictors"
pytestmark = pytest.mark.predictors


def _py(name):
    # isolated_env removes SCIGANTIC_* variables, so read them at import time.
    return _ENV.get(name)


_ENV = {k: v for k, v in os.environ.items() if k.startswith("SCIGANTIC_MSA_TEST_")}


def _run(python, script, *args, env=None):
    r = subprocess.run([python, "-I", str(HERE / script), *map(str, args)], capture_output=True, text=True, timeout=600,
                       env=None if env is None else dict(os.environ, **env))
    assert r.returncode == 0, r.stderr[-2000:]
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]


@pytest.fixture
def chains():
    def get(seq, want_uniprot=False):
        return real_alignment(seq, with_uniprot=want_uniprot)

    return get


def _msa_paths(yaml_text):
    return [json.loads(l.split("msa: ", 1)[1]) for l in yaml_text.splitlines() if l.strip().startswith("msa: ")]


def test_boltz_reads_our_files(chains, tmp_path):
    py = _py("SCIGANTIC_MSA_TEST_BOLTZ_PYTHON")
    if not py:
        pytest.skip("set SCIGANTIC_MSA_TEST_BOLTZ_PYTHON to an interpreter with boltz installed")
    mono = chains(Q_LITEFOLD)
    a, b = chains(Q_7M22A, want_uniprot=True), chains(Q_7M22C, want_uniprot=True)
    files = _msa_paths(Boltz.prepare(mono, out_dir=tmp_path / "mono"))
    files += _msa_paths(Boltz.prepare([a, b], out_dir=tmp_path / "cx", quiet=True))
    res = _run(py, "check_boltz.py", *files)
    # Boltz drops rows whose residues repeat; our A3M has none to drop.
    assert res[0]["n"] == len({r.seq.replace("-", "").upper() for r in mono.rows})
    assert res[0]["widths"] and all(w == len(Q_LITEFOLD) for w in res[0]["widths"])
    assert set(res[0]["taxonomy"]) == {-1}
    n_pair = pair_by_species([a, b]).n_paired
    ta, tb = res[1]["taxonomy"], res[2]["taxonomy"]
    shared = (set(ta) & set(tb)) - {-1}
    # Every pair we wrote survives Boltz's parser (it drops repeated rows).
    assert n_pair > 0 and len(shared) == n_pair
    assert res[1]["widths"] == [len(Q_7M22A)] and res[2]["widths"] == [len(Q_7M22C)]


def test_chai_reads_and_pairs_our_files(chains, tmp_path):
    py = _py("SCIGANTIC_MSA_TEST_CHAI_PYTHON")
    if not py:
        pytest.skip("set SCIGANTIC_MSA_TEST_CHAI_PYTHON to an interpreter with chai_lab installed")
    pytest.importorskip("pyarrow")
    mono = chains(Q_6T33)
    Chai.prepare(mono, out_dir=tmp_path / "m")
    res = _run(py, "check_chai.py", tmp_path / "m", Q_6T33)[0]
    assert res["files"][0]["depth"] == mono.depth and res["files"][0]["length"] == len(Q_6T33)
    a, b = chains(Q_7M22A, want_uniprot=True), chains(Q_7M22C, want_uniprot=True)
    Chai.prepare([a, b], out_dir=tmp_path / "c", quiet=True)
    res = _run(py, "check_chai.py", tmp_path / "c", Q_7M22A, Q_7M22C)[0]
    assert [f["length"] for f in res["files"]] == [len(Q_7M22A), len(Q_7M22C)]
    assert res["merged_depth"] > 1


def _live_result():
    """The real ColabFold server result in fixtures, read offline."""
    from conftest import FIXTURES, Q_LIVE
    from scigantic_msa.alignment import Alignment
    from scigantic_msa.remote.colabfold import ColabFoldClient

    res = ColabFoldClient("http://127.0.0.1:9").parse_result((FIXTURES / "colabfold-live1.tar.gz").read_bytes(), Q_LIVE, "t")
    return Alignment(query=Q_LIVE, rows=res.rows, source="remote", provenance=res.provenance)


def test_boltz_and_chai_read_a_real_colabfold_result(tmp_path):
    boltz_py, chai_py = _py("SCIGANTIC_MSA_TEST_BOLTZ_PYTHON"), _py("SCIGANTIC_MSA_TEST_CHAI_PYTHON")
    if not boltz_py or not chai_py:
        pytest.skip("set SCIGANTIC_MSA_TEST_BOLTZ_PYTHON and SCIGANTIC_MSA_TEST_CHAI_PYTHON")
    pytest.importorskip("pyarrow")
    m = _live_result()
    res = _run(boltz_py, "check_boltz.py", *_msa_paths(Boltz.prepare(m, out_dir=tmp_path / "b")))[0]
    assert res["n"] == len({r.seq.replace("-", "").upper() for r in m.rows}) and res["n"] > 1000
    Chai.prepare(m, out_dir=tmp_path / "c")
    res = _run(chai_py, "check_chai.py", tmp_path / "c", m.query)[0]
    assert res["files"][0]["length"] == len(m.query) and res["files"][0]["depth"] > 1000


def _server_dimer():
    """Two chains shaped like ColabFold server results (taxids from
    uniref_tax.tsv) that share 9606 and 10090; chain 1 has two 9606 rows."""
    from scigantic_msa.alignment import Alignment
    from scigantic_msa.remote.colabfold import ColabFoldClient

    def server(q, hits):
        def hit(i):
            s = list(q)
            s[i], s[i + 3] = ("G" if s[i] == "A" else "A"), ("L" if s[i + 3] == "W" else "W")
            return "".join(s)

        tar = make_result_tar(q, [(acc, hit(i), t) for acc, i, t in hits])
        res = ColabFoldClient("http://127.0.0.1:9").parse_result(tar, q, "t")
        return Alignment(query=q, rows=res.rows, source="remote", provenance=res.provenance)

    return (server(Q_7M22A, [("UniRef100_P0CG48", 2, "9606"), ("UniRef100_UPI000C299439", 10, "9606"),
                             ("UniRef100_Q9D1J1", 20, "10090"), ("UniRef100_A0A0B4J2F0", 30, "562"),
                             ("UniRef100_X1", 40, None)]),
            server(Q_7M22C, [("UniRef100_P06654", 2, "9606"), ("UniRef100_Q8BQ37", 10, "10090"),
                             ("UniRef100_P11111", 20, "7227"), ("UniRef100_Y2", 30, None)]))


def _pairing_species(d):
    """Species Protenix reads in each pairing.a3m under d, with row counts."""
    out = []
    for p in sorted(d.rglob("pairing.a3m")):
        ids = [protenix_species(l[1:]) for l in p.read_text().splitlines() if l.startswith(">")][1:]
        out.append({s: ids.count(s) for s in ids})
    return out


def test_protenix_reads_and_pairs_our_files(tmp_path, capsys):
    py = _py("SCIGANTIC_MSA_TEST_PROTENIX_PYTHON")
    if not py:
        pytest.skip("set SCIGANTIC_MSA_TEST_PROTENIX_PYTHON to an interpreter that can import protenix")
    a, c, h = (REAL / d / "cfdb_uniref30.a3m" for d in ("of3-7m22_A", "of3-7m22_C", "of3-1hvw_A"))
    runs = {
        "mono": [REAL / "litefold-A0A096LP01" / "A0A096LP01.a3m"],
        "het": [a, c],
        "het_uniprot": [a, c, "--uniprot", a.with_name("uniprot_hits.sto"), "--uniprot", c.with_name("uniprot_hits.sto")],
        "unshared": [a, h],
        "a2c": [a, a, c],
    }
    jsons = []
    for name, args in runs.items():
        assert main(["convert", "-q", "-m", "protenix", "-o", str(tmp_path / name), "--name", name, *map(str, args)]) == 0
        jsons.append(tmp_path / name / "protenix" / f"{name}.json")
    server = Protenix.prepare(list(_server_dimer()), out_dir=tmp_path / "server", name="server", quiet=True)
    jsons.append(tmp_path / "server.json")
    jsons[-1].write_text(json.dumps(server))
    capsys.readouterr()
    res = {r["name"]: r for r in _run(py, "check_protenix.py", *jsons, env={"PROTENIX_ROOT_DIR": str(tmp_path / "px")})}
    assert set(res) == set(runs) | {"server"}

    for r in res.values():
        assert all(ch["query_row_matches"] for ch in r["chains"])
        # Protenix keeps every pairing.a3m row; it reads them without dedup.
        assert all(n == kept for ch in r["chains"] for n, kept in [ch["files"].get("pairing.a3m", (0, 0))])
        # Each paired row is one species in every chain that has a row in it.
        for row in zip(*(r["paired_species"] or [])):
            assert len(set(row) - {None}) == 1

    # The only unpaired rows Protenix drops repeat once insertions are removed.
    rows = next((tmp_path / "mono").rglob("non_pairing.a3m")).read_text().splitlines()[1::2]
    table = str.maketrans("", "", string.ascii_lowercase)
    assert res["mono"]["chains"][0]["files"]["non_pairing.a3m"] == [len(rows), len({s.translate(table) for s in rows})]
    assert res["mono"]["paired_species"] is None
    assert res["unshared"]["paired_depth"] == 1 and all(len(ch["files"]) == 1 for ch in res["unshared"]["chains"])

    for name, uni in (("het", False), ("het_uniprot", True)):
        # Protenix pairs every species both files share, as many rows as the
        # shorter file has, and these are the species the library pairs on.
        first, second = _pairing_species(tmp_path / name)
        shared = set(first) & set(second)
        assert set(res[name]["paired_species"][0][1:]) == shared
        assert res[name]["paired_depth"] == 1 + sum(min(first[s], second[s]) for s in shared)
        ours = pair_by_species([real_alignment(Q_7M22A, with_uniprot=uni), real_alignment(Q_7M22C, with_uniprot=uni)])
        assert {k[3:] for k in ours.keys} == shared
    assert res["server"]["paired_species"] == [["", "9606", "10090"]] * 2
    assert res["server"]["unpaired_depth"] == 3  # the second 9606 row is kept, unpaired
    assert len(res["a2c"]["chains"]) == 3 and res["a2c"]["paired_depth"] > 1
