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

from conftest import Q_6T33, Q_7M22A, Q_7M22C, Q_LITEFOLD, key
from scigantic_msa.assemble import load_entry
from scigantic_msa.pairing import pair_by_species
from scigantic_msa.writers import write_boltz, write_chai

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
def chains(index_setup):
    idx, _, _ = index_setup

    def get(seq, **kw):
        return load_entry(idx, idx.lookup(key(seq))[0], seq, **kw)

    return get


def test_boltz_reads_our_files(chains, tmp_path):
    py = _py("SCIGANTIC_MSA_TEST_BOLTZ_PYTHON")
    if not py:
        pytest.skip("set SCIGANTIC_MSA_TEST_BOLTZ_PYTHON to an interpreter with boltz installed")
    mono = chains(Q_LITEFOLD)
    a, b = chains(Q_7M22A, want_uniprot=True), chains(Q_7M22C, want_uniprot=True)
    m_line = write_boltz([mono], tmp_path / "mono")[0]
    c_lines = write_boltz([a, b], tmp_path / "cx")
    files = [l.split("msa: ", 1)[1] for l in [m_line] + c_lines]
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
    write_chai([mono], tmp_path / "m")
    res = _run(py, "check_chai.py", tmp_path / "m", Q_6T33)[0]
    assert res["files"][0]["depth"] == mono.depth and res["files"][0]["length"] == len(Q_6T33)
    a, b = chains(Q_7M22A, want_uniprot=True), chains(Q_7M22C, want_uniprot=True)
    write_chai([a, b], tmp_path / "c")
    res = _run(py, "check_chai.py", tmp_path / "c", Q_7M22A, Q_7M22C)[0]
    assert [f["length"] for f in res["files"]] == [len(Q_7M22A), len(Q_7M22C)]
    assert res["merged_depth"] > 1
