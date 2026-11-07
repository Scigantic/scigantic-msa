"""Load our outputs with the predictors' own code.

Needs interpreters that have the predictors installed, for example:
    uv venv vboltz && uv pip install -p vboltz/bin/python boltz
    uv venv vchai && uv pip install -p vchai/bin/python chai_lab
    SCIGANTIC_MSA_TEST_BOLTZ_PYTHON=vboltz/bin/python \\
    SCIGANTIC_MSA_TEST_CHAI_PYTHON=vchai/bin/python pytest -m predictors
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, Q_LITEFOLD, real_alignment
from scigantic_msa.adapters import Boltz, Chai
from scigantic_msa.pairing import pair_by_species

HERE = Path(__file__).parent / "predictors"
pytestmark = pytest.mark.predictors


def _py(name):
    # isolated_env removes SCIGANTIC_* variables, so read them at import time.
    return _ENV.get(name)


_ENV = {k: v for k, v in os.environ.items() if k.startswith("SCIGANTIC_MSA_TEST_")}


def _run(python, script, *args):
    r = subprocess.run([python, "-I", str(HERE / script), *map(str, args)], capture_output=True, text=True, timeout=600)
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
