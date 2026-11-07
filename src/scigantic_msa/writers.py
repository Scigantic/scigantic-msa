"""Write MSAs in the input format each structure predictor reads.

Every writer that produces files also writes ``scigantic-msa-attribution.json``
next to them: which source each alignment came from, its license, whether
rows were copied from a near-identical sequence, and notes about pairing.

Format rules are from the predictors' own parsers (see FORMATS notes in the
repository history): AlphaFold 3 ``folding_input.py``, Boltz ``parse/csv.py``
and ``main.py``, Chai-1 ``aligned_pqt.py``, ColabFold ``input.py`` and
OpenFold3 ``dataset_config_components.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import string
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .a3m import Row, format_a3m
from .msa import MSA, dedupe_rows
from .pairing import Pairing, pair_by_species
from .seqkey import canonical_sequence

__all__ = [
    "write_a3m",
    "write_colabfold",
    "write_af3_json",
    "write_boltz",
    "write_chai",
    "write_openfold3",
    "entities",
    "ATTRIBUTION_FILE",
]

ATTRIBUTION_FILE = "scigantic-msa-attribution.json"
AF3_PROTEIN_KEYS = frozenset(
    {"id", "sequence", "modifications", "description", "unpairedMsa", "unpairedMsaPath",
     "pairedMsa", "pairedMsaPath", "templates"}
)
CHAI_SOURCES = ("query", "uniprot", "uniref90", "bfd_uniclust", "mgnify")


def _atomic_write(path: Path, data: "str | bytes") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    try:
        if isinstance(data, bytes):
            with os.fdopen(fd, "wb") as fb:
                fb.write(data)
        else:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as ft:
                ft.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def entities(chains: Sequence[MSA]) -> List[Tuple[MSA, int, List[int]]]:
    """Group chains with the same sequence: (msa, copies, chain indexes)."""
    out: Dict[str, Tuple[MSA, List[int]]] = {}
    for i, m in enumerate(chains):
        if m.query in out:
            out[m.query][1].append(i)
        else:
            out[m.query] = (m, [i])
    return [(m, len(idx), idx) for m, idx in out.values()]


def _pairing(ents: Sequence[MSA], pairing: Optional[Pairing]) -> Pairing:
    if pairing is not None:
        return pairing
    return pair_by_species(ents)


def _sidecar(directory: Path, chains: Sequence[MSA], notes: List[str], files: Dict[str, Any]) -> None:
    data = {
        "written_by": "scigantic-msa",
        "chains": [
            {
                "seq_sha256": m.seq_sha256,
                "length": len(m.query),
                "source": m.source,
                "summary": m.summary(),
                "depth": m.depth,
                "attribution": m.attribution,
                "derived": m.derived,
                "provenance": m.provenance,
            }
            for m in chains
        ],
        "files": files,
        "notes": notes,
    }
    _atomic_write(directory / ATTRIBUTION_FILE, json.dumps(data, indent=1, sort_keys=True) + "\n")


def _unpaired_with_prepaired(m: MSA, paired: List[Row]) -> List[Row]:
    """Query, then the paired rows in order, then the remaining rows."""
    used = {id(r) for r in paired}
    rest = [r for r in m.rows[1:] if id(r) not in used]
    return [m.rows[0]] + list(paired) + dedupe_rows([m.rows[0]] + rest)[1:]


# --------------------------------------------------------------------------- A3M


def write_a3m(msa: MSA, path: "str | Path | None" = None, *, max_rows: Optional[int] = None) -> str:
    text = msa.a3m(max_rows=max_rows)
    if path is not None:
        p = Path(path)
        _atomic_write(p, text)
        _sidecar(p.parent, [msa], [], {msa.seq_sha256: p.name})
    return text


# --------------------------------------------------------------------------- ColabFold


def write_colabfold(chains: Sequence[MSA], path: "str | Path", *, pairing: Optional[Pairing] = None) -> Path:
    """One A3M for ``colabfold_batch``. A monomer is a plain A3M; a
    homo-oligomer gets ``#<len>\\t<copies>``; a complex gets the ColabFold
    complex layout with species-paired rows first (positional pairing) and
    each chain's unpaired rows padded with gaps."""
    p = Path(path)
    ents = entities(chains)
    notes: List[str] = []
    if len(ents) == 1:
        m, copies, _ = ents[0]
        text = m.a3m()
        if copies > 1:
            text = f"#{len(m.query)}\t{copies}\n" + text
    else:
        msas = [e[0] for e in ents]
        pr = _pairing(msas, pairing)
        notes.append(pr.message)
        lens = [len(m.query) for m in msas]
        cards = [e[1] for e in ents]
        names = [str(101 + i) for i in range(len(msas))]
        out = [f"#{','.join(map(str, lens))}\t{','.join(map(str, cards))}",
               ">" + "\t".join(names), "".join(m.query for m in msas)]
        for k in range(pr.n_paired):
            out.append(">" + "\t".join(pr.rows[c][k].header.split("\t")[0] for c in range(len(msas))))
            out.append("".join(pr.rows[c][k].seq for c in range(len(msas))))
        total = sum(lens)
        offset = 0
        for c, m in enumerate(msas):
            pre, post = "-" * offset, "-" * (total - offset - lens[c])
            for r in m.rows:
                hdr = names[c] if r is m.rows[0] else r.header
                out.append(">" + hdr)
                out.append(pre + r.seq + post)
            offset += lens[c]
        text = "\n".join(out) + "\n"
    _atomic_write(p, text)
    _sidecar(p.parent, chains, notes, {"colabfold_a3m": p.name})
    return p


# --------------------------------------------------------------------------- AlphaFold 3


def write_af3_json(
    chains: Sequence[MSA],
    path: "str | Path | None" = None,
    *,
    input: "dict | str | Path | None" = None,  # noqa: A002 - matches AF3 wording
    name: Optional[str] = None,
    inline: bool = True,
    pairing: Optional[Pairing] = None,
) -> dict:
    """Write or merge an AlphaFold 3 input JSON.

    With ``input`` (a dict or a path to an existing AF3 JSON), protein entries
    whose sequence matches one of ``chains`` get ``unpairedMsa`` and
    ``pairedMsa`` set; everything else is kept. Without it, a new input is
    made with one protein entry per distinct sequence.

    Pairing: when every distinct chain has UniProt rows, they go into
    ``pairedMsa`` and AlphaFold 3 pairs them by species itself. Otherwise rows
    paired by taxid are put first in ``unpairedMsa`` (AF3 joins unpaired rows
    across chains by row index), ``pairedMsa`` is "", and the note says to run
    with ``--resolve_msa_overlaps=false``.

    ``inline=False`` writes ``<seq_sha256>.unpaired.a3m`` (and ``.paired.a3m``)
    next to the JSON and uses the ``*Path`` fields, which need ``path``.
    """
    if not inline and path is None:
        raise ValueError("inline=False needs a path for the JSON")
    out_path = Path(path) if path is not None else None
    if isinstance(input, (str, Path)):
        doc = json.loads(Path(input).read_text())
    elif isinstance(input, dict):
        doc = copy.deepcopy(input)
    else:
        doc = None
    ents = entities(chains)
    msas = [e[0] for e in ents]
    notes: List[str] = []
    paired_text: Dict[str, str] = {}
    unpaired_rows: Dict[str, List[Row]] = {m.query: dedupe_rows(m.rows) for m in msas}
    if len(msas) > 1:
        if all(m.uniprot_rows and len(m.uniprot_rows) > 1 for m in msas):
            for m in msas:
                paired_text[m.query] = format_a3m(m.uniprot_rows or [])
            notes.append("pairedMsa holds UniProt hits; AlphaFold 3 pairs them by species")
        else:
            pr = _pairing(msas, pairing)
            notes.append(pr.message)
            if pr.n_paired:
                for c, m in enumerate(msas):
                    unpaired_rows[m.query] = _unpaired_with_prepaired(m, pr.rows[c])
                notes.append(
                    "rows paired by taxid are first in unpairedMsa; run AlphaFold 3 with "
                    "--resolve_msa_overlaps=false so it keeps them"
                )
    if doc is None:
        letters = iter(_chain_ids())
        seqs = []
        for m, copies, _ in ents:
            ids = [next(letters) for _ in range(copies)]
            seqs.append({"protein": {"id": ids if len(ids) > 1 else ids[0], "sequence": m.query}})
        doc = {
            "name": name or f"msa_{msas[0].seq_sha256[:12]}",
            "modelSeeds": [1],
            "sequences": seqs,
            "dialect": "alphafold3",
            "version": 2,
        }
    if doc.get("dialect") != "alphafold3":
        raise ValueError("input is not an AlphaFold 3 JSON (dialect must be 'alphafold3')")
    if not inline:
        doc["version"] = max(int(doc.get("version", 2)), 2)
    by_seq = {m.query: m for m in msas}
    matched = set()
    files: Dict[str, Any] = {}
    for entry in doc.get("sequences", []):
        prot = entry.get("protein")
        if not isinstance(prot, dict):
            continue
        bad = set(prot) - AF3_PROTEIN_KEYS
        if bad:
            raise ValueError(f"protein entry has keys AlphaFold 3 does not accept: {sorted(bad)}")
        try:
            q = canonical_sequence(prot.get("sequence", ""))
        except ValueError:
            continue
        found = by_seq.get(q)
        if found is None:
            continue
        m = found
        matched.add(q)
        for k in ("unpairedMsa", "unpairedMsaPath", "pairedMsa", "pairedMsaPath"):
            prot.pop(k, None)
        up = format_a3m(unpaired_rows[q])
        pa = paired_text.get(q, "")
        if inline:
            prot["unpairedMsa"] = up
            prot["pairedMsa"] = pa
        else:
            assert out_path is not None
            fn = f"{m.seq_sha256}.unpaired.a3m"
            _atomic_write(out_path.parent / fn, up)
            prot["unpairedMsaPath"] = fn
            files[m.seq_sha256] = {"unpaired": fn}
            if pa:
                pfn = f"{m.seq_sha256}.paired.a3m"
                _atomic_write(out_path.parent / pfn, pa)
                prot["pairedMsaPath"] = pfn
                files[m.seq_sha256]["paired"] = pfn
            else:
                prot["pairedMsa"] = ""
        if prot.get("templates") is None:
            prot["templates"] = []
    missing = set(by_seq) - matched
    if missing:
        raise ValueError(
            f"{len(missing)} alignment(s) match no protein sequence in the AlphaFold 3 input"
        )
    for entry in doc.get("sequences", []):
        prot = entry.get("protein")
        if isinstance(prot, dict) and prot.get("unpairedMsa") is None and "unpairedMsaPath" not in prot:
            notes.append(
                f"protein {prot.get('id')} has no MSA from scigantic-msa; AlphaFold 3 will run its own search for it"
            )
    if out_path is not None:
        _atomic_write(out_path, json.dumps(doc, indent=2) + "\n")
        files["af3_json"] = out_path.name
        _sidecar(out_path.parent, chains, notes, files)
    return doc


def _chain_ids():
    for c in string.ascii_uppercase:
        yield c
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            yield a + b


# --------------------------------------------------------------------------- Boltz


def write_boltz(chains: Sequence[MSA], directory: "str | Path", *, pairing: Optional[Pairing] = None) -> List[str]:
    """Write Boltz MSA files and return one YAML ``msa:`` line per chain.

    A single distinct sequence gets ``<seq_sha256>.a3m``. Several get
    ``<seq_sha256>.csv`` (header ``key,sequence``): the query has key -1,
    rows paired by species get the pair index (Boltz pairs equal keys), the
    rest -1. Boltz refuses ``.a3m.gz``, so files are never compressed.
    """
    d = Path(directory)
    ents = entities(chains)
    msas = [e[0] for e in ents]
    notes: List[str] = []
    paths: Dict[str, Path] = {}
    if len(msas) == 1:
        m = msas[0]
        p = d / f"{m.seq_sha256}.a3m"
        _atomic_write(p, m.a3m())
        paths[m.query] = p
    else:
        pr = _pairing(msas, pairing)
        notes.append(pr.message)
        for c, m in enumerate(msas):
            lines = ["key,sequence", f"-1,{m.query}"]
            for k, r in enumerate(pr.rows[c], 1):
                lines.append(f"{k},{r.seq}")
            used = {id(r) for r in pr.rows[c]}
            for r in m.rows[1:]:
                if id(r) not in used:
                    lines.append(f"-1,{r.seq}")
            p = d / f"{m.seq_sha256}.csv"
            _atomic_write(p, "\n".join(lines) + "\n")
            paths[m.query] = p
    _sidecar(d, chains, notes, {m.seq_sha256: paths[m.query].name for m in msas})
    return [f"msa: {paths[m.query]}" for m in chains]


# --------------------------------------------------------------------------- Chai-1


def chai_source(row: Row) -> str:
    if row.db == "query":
        return "query"
    if row.db in ("uniref", "uniref90"):
        return "uniref90"
    if row.db == "uniprot":
        return "uniprot"
    if row.db == "mgnify":
        return "mgnify"
    return "bfd_uniclust"


def chai_expected_basename(sequence: str) -> str:
    """The file name Chai-1 looks for (``expected_basename`` in chai_lab)."""
    return hashlib.sha256(sequence.upper().encode()).hexdigest() + ".aligned.pqt"


def write_chai(chains: Sequence[MSA], directory: "str | Path", *, pairing: Optional[Pairing] = None) -> List[Path]:
    """Write ``<seq_sha256>.aligned.pqt`` per distinct sequence, the file
    Chai-1 looks up in ``--msa-directory``. Needs pyarrow
    (``pip install 'scigantic-msa[chai]'``)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("writing Chai-1 files needs pyarrow: pip install 'scigantic-msa[chai]'") from None
    d = Path(directory)
    ents = entities(chains)
    msas = [e[0] for e in ents]
    notes: List[str] = []
    pr = _pairing(msas, pairing) if len(msas) > 1 else None
    if pr is not None:
        notes.append(pr.message)
    out: Dict[str, Path] = {}
    for c, m in enumerate(msas):
        name = chai_expected_basename(m.query)
        if name != m.seq_sha256 + ".aligned.pqt":
            raise AssertionError("Chai-1 file name does not match seq_sha256")
        keyed: Dict[int, str] = {}
        if pr is not None:
            for k, r in enumerate(pr.rows[c], 1):
                keyed[id(r)] = str(k)
        rows = [m.rows[0]]
        if pr is not None:
            rows += pr.rows[c]
        used = {id(r) for r in rows}
        rows += [r for r in m.rows[1:] if id(r) not in used]
        seqs, srcs, keys, comments = [], [], [], []
        for i, r in enumerate(rows):
            seqs.append(r.seq)
            srcs.append("query" if i == 0 else chai_source(r))
            keys.append(keyed.get(id(r), ""))
            comments.append(r.header)
        table = pa.table(
            {
                "sequence": pa.array(seqs, pa.string()),
                "source_database": pa.array(srcs, pa.string()),
                "pairing_key": pa.array(keys, pa.string()),
                "comment": pa.array(comments, pa.string()),
            }
        )
        p = d / name
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / ("." + name + ".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, p)
        out[m.query] = p
    _sidecar(d, chains, notes, {m.seq_sha256: out[m.query].name for m in msas})
    return [out[m.query] for m in chains]


# --------------------------------------------------------------------------- OpenFold3


def write_openfold3(chains: Sequence[MSA], directory: "str | Path", *, pairing: Optional[Pairing] = None) -> List[dict]:
    """Write OpenFold3 MSA directories and return, per chain, the fields to
    put in that chain of the query JSON.

    ``<dir>/main/<seq_sha256>/colabfold_main.a3m`` holds the unpaired rows,
    plus ``uniprot_hits.a3m`` when UniProt rows exist (OpenFold3 pairs those
    by species itself). If some chain has no UniProt rows, rows paired by
    taxid go to ``<dir>/paired/<complex>/<seq_sha256>/colabfold_paired.a3m``.
    """
    d = Path(directory)
    ents = entities(chains)
    msas = [e[0] for e in ents]
    notes: List[str] = []
    all_uniprot = all(m.uniprot_rows and len(m.uniprot_rows) > 1 for m in msas)
    fields: Dict[str, dict] = {}
    files: Dict[str, Any] = {}
    paired_dirs: Dict[str, Path] = {}
    if len(msas) > 1 and not all_uniprot:
        pr = _pairing(msas, pairing)
        notes.append(pr.message)
        if pr.n_paired:
            cid = hashlib.sha256("|".join(m.seq_sha256 for m in msas).encode()).hexdigest()[:16]
            for c, m in enumerate(msas):
                pdir = d / "paired" / cid / m.seq_sha256
                _atomic_write(pdir / "colabfold_paired.a3m", format_a3m([m.rows[0]] + pr.rows[c]))
                paired_dirs[m.query] = pdir
    elif len(msas) > 1:
        notes.append("uniprot_hits.a3m written for every chain; OpenFold3 pairs them by species")
    for m in msas:
        mdir = d / "main" / m.seq_sha256
        _atomic_write(mdir / "colabfold_main.a3m", m.a3m())
        if m.uniprot_rows and len(m.uniprot_rows) > 1 and (all_uniprot or len(msas) == 1):
            _atomic_write(mdir / "uniprot_hits.a3m", m.uniprot_a3m())
        f = {"main_msa_file_paths": str(mdir)}
        if m.query in paired_dirs:
            f["paired_msa_file_paths"] = str(paired_dirs[m.query])
        fields[m.query] = f
        files[m.seq_sha256] = {k: os.path.relpath(v, d) for k, v in f.items()}
    _sidecar(d, chains, notes, files)
    return [fields[m.query] for m in chains]
