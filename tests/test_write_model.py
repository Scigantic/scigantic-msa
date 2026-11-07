"""The uniform way to write inputs, and the errors a caller gets when wrong.

The six adapters do not share a signature: five require ``out_dir``, one does
not, two take no ``name``, and they return four different types. The README's
only example is AlphaFold 3, the one adapter whose arguments are all optional,
so the obvious generalisation from it is a TypeError on the other five. Found
by driving all six from a stress script against the published wheel.

``write_model`` is the one call that works for every model, which is what
anything calling this generically (an agent, a notebook, a pipeline) needs.
"""
from __future__ import annotations

import pytest

from scigantic_msa import MODELS, MSA, write_model
from scigantic_msa.a3m import A3MError

A3M = ">query\nMKTAYIAKQRQISFVKSHFSRQ\n>hit\nMKTAYIAKQRQISFVKSHFSRQ\n"


@pytest.fixture
def chains():
    return [MSA().from_a3m(A3M)]


@pytest.mark.parametrize("model", MODELS)
def test_every_model_writes_files_and_a_next_command(model, chains, tmp_path):
    """One call per model, no per-adapter keywords to learn."""
    if model == "chai":
        # Chai-1 files are parquet. The stdlib-only CI job installs without
        # the [chai] extra on purpose, to hold the "core has no dependencies"
        # claim honest, and a test that ignores that breaks the claim rather
        # than checking it.
        pytest.importorskip("pyarrow")
    files, nxt = write_model(model, chains, tmp_path, "job", quiet=True)
    assert files, f"{model} reported no files"
    for f in files:
        assert (tmp_path / f).exists() or f.startswith(str(tmp_path)), f
    # The next command is the point: it tells the caller what to run, so a
    # caller that cannot read six sets of model docs still gets somewhere.
    assert nxt.strip(), f"{model} gave no next command"


@pytest.mark.parametrize("model", MODELS)
def test_every_model_handles_a_two_chain_complex(model, chains, tmp_path):
    if model == "chai":
        pytest.importorskip("pyarrow")
    files, nxt = write_model(model, chains * 2, tmp_path, "job", quiet=True)
    assert files and nxt.strip()


def test_unknown_model_names_the_ones_that_work(chains, tmp_path):
    with pytest.raises(ValueError) as e:
        write_model("alphafold2", chains, tmp_path, "job", quiet=True)
    assert "alphafold2" in str(e.value)


def test_it_is_importable_from_the_package_root():
    # It used to live in cli.py, where `from scigantic_msa import ...` could
    # not reach it and dir(scigantic_msa) did not show it.
    import scigantic_msa

    assert "write_model" in scigantic_msa.__all__
    assert "MODELS" in scigantic_msa.__all__


class TestFromA3mTellsYouWhichBranchItTook:
    """A str is read as a PATH when it has no newline and no '>' header.

    That heuristic is right for real input and baffling when it is wrong: the
    old message for an empty string was "``.`` does not exist", which names
    the working directory and claims it is missing.
    """

    def test_empty_string_says_it_was_empty(self):
        with pytest.raises(A3MError, match="empty string"):
            MSA().from_a3m("")

    def test_a_directory_is_not_reported_as_missing(self, tmp_path):
        with pytest.raises(A3MError, match="is a directory"):
            MSA().from_a3m(str(tmp_path))

    def test_a_missing_file_explains_the_heuristic(self):
        with pytest.raises(FileNotFoundError) as e:
            MSA().from_a3m("not-here.a3m")
        msg = str(e.value)
        assert "read as a file path" in msg
        assert "'>'" in msg, "it should say what A3M text would need"

    def test_real_a3m_text_is_still_parsed(self):
        assert MSA().from_a3m(A3M).query == "MKTAYIAKQRQISFVKSHFSRQ"


def test_readme_signature_table_matches_the_adapters():
    """The table exists because the adapters disagree; a stale table is worse
    than none, since the whole point is that a reader cannot guess."""
    import inspect
    import re
    from pathlib import Path

    from scigantic_msa import adapters as A

    text = (Path(__file__).parent.parent / "README.md").read_text()
    for name in ("AlphaFold3", "Boltz", "Chai", "ColabFold", "OpenFold3", "Protenix"):
        sig = inspect.signature(getattr(A, name).prepare)
        row = re.search(rf"^\| `{name}` \| ([^|]+) \| ([^|]+) \|", text, re.M)
        assert row, f"{name} is missing from the README table"
        documented_required = "**required**" in row.group(1)
        actually_required = sig.parameters["out_dir"].default is inspect.Parameter.empty
        assert documented_required == actually_required, f"{name}: out_dir"
        assert (row.group(2).strip() == "yes") == ("name" in sig.parameters), f"{name}: name"


def test_the_package_reports_the_version_it_was_built_as():
    """0.2.1 shipped reporting itself as 0.2.0.

    There were two hand-maintained version strings, in pyproject.toml and in
    __init__.py, with nothing tying them together, so bumping one and not the
    other published a package whose own `version` command lied. pyproject now
    reads the module attribute, and this checks the two cannot drift again.
    """
    import re
    from pathlib import Path

    import scigantic_msa

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    # The version must NOT be written out a second time here.
    assert not re.search(r"^version\s*=\s*[\"']", pyproject, re.M), (
        "pyproject declares its own version again; it should stay dynamic"
    )
    assert 'version = {attr = "scigantic_msa.__version__"}' in pyproject
    assert re.fullmatch(r"\d+\.\d+\.\d+", scigantic_msa.__version__), scigantic_msa.__version__


def test_readme_call_works_as_written(chains, tmp_path):
    # The README calls write_model without quiet and with a str out_dir.
    files, nxt = write_model("boltz", chains, str(tmp_path), name="job")
    assert files and files[0].startswith(str(tmp_path / "boltz"))
