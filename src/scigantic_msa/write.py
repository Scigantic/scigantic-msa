"""One call that writes the input files for any supported predictor.

The six adapters do not share a signature: five of them require ``out_dir``
and one does not, two take no ``name``, and they return four different types
(a dict, a YAML string, a Path, a list of dicts). That is fine for a human
reading one adapter's docstring, and hostile to anything calling them
generically: the obvious pattern from the README's AlphaFold 3 example is a
``TypeError`` on the other five.

``write_model`` is the uniform way in. It takes the same model names as the
``--model`` flag, writes the files, and returns what it wrote along with the
command to run next, so a caller never has to know which adapter wanted which
keyword. It lived inside cli.py, where only the command line could reach it.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import List, Tuple

from .adapters import AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix
from .alignment import Alignment

#: The model names ``write_model`` accepts, same strings as ``--model``.
MODELS = ("af3", "boltz", "chai", "colabfold", "openfold3", "protenix")


def write_model(model: str, chains: List[Alignment], out: "str | Path", name: str = "query",
                quiet: bool = False) -> Tuple[List[str], str]:
    """Write the input for one model into ``out/<model>/``. Returns (files
    written, next command).

    Each model gets its own folder so several models can share one ``out``
    without overwriting each other's input, MSA files, attribution file or
    predictions."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    out = Path(out) / model
    out.mkdir(parents=True, exist_ok=True)
    if model == "af3":
        AlphaFold3.prepare(chains, name=name, out_dir=out, quiet=quiet)
        p = out / f"{_fname(name)}.json"
        flag = " --resolve_msa_overlaps=false" if len({c.query for c in chains}) > 1 else ""
        return [str(p)], (f"python run_alphafold.py --json_path={_q(p)} --model_dir=WEIGHTS "
                          f"--output_dir={_q(out / 'pred')} --norun_data_pipeline{flag}")
    if model == "boltz":
        yaml = Boltz.prepare(chains, out_dir=out / "msas", quiet=quiet)
        p = out / f"{_fname(name)}.yaml"
        p.write_text(yaml)
        return [str(p), str(out / "msas")], f"boltz predict {_q(p)} --out_dir {_q(out / 'pred')}"
    if model == "chai":
        d = Chai.prepare(chains, out_dir=out / "msas", quiet=quiet)
        p = out / f"{_fname(name)}.fasta"
        p.write_text(Chai.fasta(chains))
        return [str(p), str(d)], f"chai-lab fold --msa-directory {_q(d)} {_q(p)} {_q(out / 'pred')}"
    if model == "colabfold":
        p = ColabFold.prepare(chains, out_dir=out, name=_fname(name), quiet=quiet)
        return [str(p)], f"colabfold_batch {_q(p)} {_q(out / 'pred')}"
    if model == "openfold3":
        q = OpenFold3.prepare(chains, out_dir=out / "msas", name=name, quiet=quiet)
        p = out / f"{_fname(name)}.json"
        p.write_text(json.dumps(q, indent=2) + "\n")
        return [str(p), str(out / "msas")], f"run_openfold predict --query_json {_q(p)} --use_msa_server false --output_dir {_q(out / 'pred')}"
    if model == "protenix":
        jobs = Protenix.prepare(chains, out_dir=out / "msas", name=name, quiet=quiet)
        p = out / f"{_fname(name)}.json"
        p.write_text(json.dumps(jobs, indent=2) + "\n")
        return [str(p), str(out / "msas")], f"protenix pred -i {_q(p)} -o {_q(out / 'pred')}"
    raise ValueError(f"unknown model {model}")


def _q(path: "str | Path") -> str:
    """A path as one shell word, so the printed command can be pasted as is
    when --out has spaces or quotes in it."""
    return shlex.quote(str(path))


def _fname(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip(".")
    return s[:100] or "query"
