"""Model adapters: turn alignments into the input files a structure predictor
reads.

    from scigantic_msa import MSA
    from scigantic_msa.adapters import AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix

    result = MSA().get(sequence, remote=True)
    af3_input = AlphaFold3.prepare(result)               # dict ready for json.dump
    boltz_yaml = Boltz.prepare(result, out_dir="boltz/")  # writes files, returns YAML text
    chai_dir = Chai.prepare(result, out_dir="chai_msas/")

Every ``prepare`` takes one result or a list of results (one per chain, in
chain order; repeat a result for a homo-oligomer). The results carry their
sequences. If you also pass ``sequence=`` (a string, or a list for a
complex), it must match, or ``prepare`` raises ValueError.

Each adapter checks what it wrote: every MSA file it names exists, is not
empty, parses as an alignment whose first row is the chain's sequence, and
has the name the model looks for. A file that fails any of these is an
error, never a silent fallback.

Format rules come from the models' own code; see the notes on each class.
"""
from __future__ import annotations

import json
import os
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .a3m import A3MError, Row, format_a3m, match_length, parse_a3m, parse_colabfold_complex
from .alignment import Alignment, species_key
from .pairing import Pairing, pair_by_species
from .seqkey import InvalidSequence, canonical_sequence, seq_sha256
from . import writers

__all__ = ["AlphaFold3", "Boltz", "Chai", "ColabFold", "OpenFold3", "Protenix", "PrepareError"]

Results = Union[Alignment, Sequence[Alignment]]


class PrepareError(ValueError):
    """An input could not be prepared, or a written file failed its check."""


# --------------------------------------------------------------------------- shared


def _chains(results: Any, sequence: Any = None) -> List[Alignment]:
    if isinstance(results, Alignment):
        chains = [results]
    elif isinstance(results, (list, tuple)) and results and all(isinstance(r, Alignment) for r in results):
        chains = list(results)
    else:
        raise TypeError("prepare() takes a result of MSA().get() or MSA().from_a3m(), or a list of them, one per chain")
    for i, c in enumerate(chains, 1):
        _check_rows(c, f"chain {i}")
    if sequence is not None:
        seqs = [sequence] if isinstance(sequence, str) else list(sequence)
        if len(seqs) != len(chains):
            raise PrepareError(f"{len(seqs)} sequence(s) passed for {len(chains)} chain result(s)")
        for i, (s, c) in enumerate(zip(seqs, chains), 1):
            try:
                q = canonical_sequence(s)
            except InvalidSequence as e:
                raise PrepareError(f"chain {i}: {e}") from None
            if q != c.query:
                raise PrepareError(
                    f"chain {i}: the sequence passed does not match the alignment's query "
                    f"(length {len(q)} vs {len(c.query)}, seq_sha256 {seq_sha256(q)[:12]} vs {c.seq_sha256[:12]})"
                )
    return chains


def _check_rows(m: Alignment, what: str) -> None:
    """The result may have been edited after it was made; check it again."""
    if not m.rows or m.rows[0].seq != m.query:
        raise PrepareError(f"{what}: the first row is not the query")
    n = len(m.query)
    for k, r in enumerate(m.rows):
        if match_length(r.seq) != n:
            raise PrepareError(f"{what}: row {k + 1} has {match_length(r.seq)} match columns, the query has {n}")


def _distinct(chains: Sequence[Alignment]) -> List[Tuple[Alignment, List[int]]]:
    """Distinct sequences in first-seen order, with the chain indexes of each."""
    return [(m, idx) for m, _, idx in writers.entities(chains)]


def chain_id(i: int) -> str:
    """A, B, ..., Z, AA, AB, ..."""
    letters = string.ascii_uppercase
    if i < 26:
        return letters[i]
    i -= 26
    return letters[i // 26] + letters[i % 26]


def _say(msg: str, quiet: bool) -> None:
    if not quiet:
        print(f"scigantic-msa: {msg}", file=sys.stderr, flush=True)


def _pairing_note(msas: Sequence[Alignment], quiet: bool) -> Optional[Pairing]:
    if len(msas) < 2:
        return None
    pr = pair_by_species(msas)
    _say(pr.message, quiet)
    return pr


def _out(out_dir: Any) -> Path:
    if out_dir is None:
        raise TypeError("out_dir is required: this adapter writes files")
    d = Path(out_dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _verify_a3m(path: Path, query: str, *, what: str) -> int:
    """The written file exists, is not empty and its first row is the query.
    Returns the number of rows."""
    if not path.is_file():
        raise PrepareError(f"{what}: {path} was not written")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PrepareError(f"{what}: {path} is empty")
    try:
        return len(parse_a3m(text, query, what=str(path.name)))
    except A3MError as e:
        raise PrepareError(f"{what}: {e}") from None


# --------------------------------------------------------------------------- AlphaFold 3


class AlphaFold3:
    """AlphaFold 3 input JSON with the MSAs inline.

    Rules from ``alphafold3/common/folding_input.py``: ``dialect`` is
    ``alphafold3``, protein entries take only id, sequence, modifications,
    description, unpairedMsa(Path), pairedMsa(Path) and templates. A custom
    MSA is ``unpairedMsa`` with ``pairedMsa`` set ("" when there is none).

    Complexes: AlphaFold 3 pairs ``pairedMsa`` rows itself, but only by the
    species in UniProt headers. ColabFold server rows are UniRef100 and carry
    no species in the header, so rows are paired here by NCBI taxid and put
    first in ``unpairedMsa``. Run AlphaFold 3 with
    ``--resolve_msa_overlaps=false`` so it keeps them; ``prepare`` says so.
    """

    @staticmethod
    def prepare(results: Results, sequence: Any = None, *, name: Optional[str] = None,
                input: Any = None, out_dir: Any = None, quiet: bool = False) -> dict:  # noqa: A002
        """Return the AF3 input as a dict. With ``input`` (a dict or a path to
        an AF3 JSON), fill in the matching protein entries and keep the rest.
        With ``out_dir``, also write ``<name>.json`` there."""
        chains = _chains(results, sequence)
        path = None
        if out_dir is not None:
            d = _out(out_dir)
            path = d / f"{_safe_name(name or 'input')}.json"
        doc = writers.write_af3_json(chains, path, input=input, name=name)
        msas = [m for m, _ in _distinct(chains)]
        if len(msas) > 1:
            if all(m.uniprot_rows and len(m.uniprot_rows) > 1 for m in msas):
                _say("pairedMsa holds UniProt rows; AlphaFold 3 pairs them by species", quiet)
            else:
                pr = _pairing_note(msas, quiet)
                if pr is not None and pr.n_paired:
                    _say("run AlphaFold 3 with --resolve_msa_overlaps=false so it keeps the paired rows", quiet)
        for entry in doc.get("sequences", []):
            prot = entry.get("protein")
            if isinstance(prot, dict) and isinstance(prot.get("unpairedMsa"), str) and prot["unpairedMsa"]:
                try:
                    parse_a3m(prot["unpairedMsa"], prot["sequence"], what=f"protein {prot.get('id')} unpairedMsa")
                except A3MError as e:
                    raise PrepareError(str(e)) from None
        return doc


# --------------------------------------------------------------------------- Boltz


class Boltz:
    """Boltz MSA files and the YAML that points at them.

    Rules from ``boltz/main.py`` and ``boltz/data/parse/{a3m,csv}.py``: the
    ``msa:`` value must end in ``.a3m`` or ``.csv`` (``.a3m.gz`` is refused).
    An ``.a3m`` is read with no taxonomy, so it never pairs. A complex gets
    one ``.csv`` per chain with header ``key,sequence``; rows with the same
    key are paired across chains, and -1 means unpaired. Boltz does not check
    that the first row is the chain sequence; ``prepare`` does. The paths in
    the YAML are absolute, because Boltz reads them relative to the directory
    it is run from.
    """

    @staticmethod
    def prepare(results: Results, sequence: Any = None, *, out_dir: Any, quiet: bool = False) -> str:
        """Write the MSA files into ``out_dir``; return the YAML text (a full
        ``version: 1`` document with one protein entry per distinct sequence).
        Add ligands or constraints to it as you need."""
        chains = _chains(results, sequence)
        d = _out(out_dir)
        dist = _distinct(chains)
        msas = [m for m, _ in dist]
        pr = _pairing_note(msas, quiet)
        lines = writers.write_boltz(chains, d, pairing=pr)
        paths = {m.query: Path(lines[idx[0]].split("msa: ", 1)[1]) for m, idx in dist}
        for m, _ in dist:
            p = paths[m.query]
            if p.suffix not in (".a3m", ".csv"):
                raise PrepareError(f"Boltz reads only .a3m or .csv MSA files, not {p.name}")
            if p.suffix == ".a3m":
                _verify_a3m(p, m.query, what="Boltz")
            else:
                _verify_boltz_csv(p, m.query)
        out = ["version: 1", "sequences:"]
        for m, idx in dist:
            ids = [chain_id(i) for i in idx]
            out += [
                "  - protein:",
                f"      id: {ids[0]}" if len(ids) == 1 else f"      id: [{', '.join(ids)}]",
                f"      sequence: {m.query}",
                f"      msa: {json.dumps(str(paths[m.query]))}",
            ]
        return "\n".join(out) + "\n"


def _verify_boltz_csv(path: Path, query: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise PrepareError(f"Boltz: {path} is missing or empty")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "key,sequence":
        raise PrepareError(f"Boltz: {path.name} does not start with the header key,sequence")
    if len(lines) < 2 or lines[1].split(",", 1)[-1] != query:
        raise PrepareError(f"Boltz: the first row of {path.name} is not the chain sequence")
    for k, line in enumerate(lines[1:], 2):
        key, _, row = line.partition(",")
        if match_length(row) != len(query):
            raise PrepareError(f"Boltz: line {k} of {path.name} has {match_length(row)} match columns, not {len(query)}")


# --------------------------------------------------------------------------- Chai-1


class Chai:
    """Chai-1 ``.aligned.pqt`` files, one per distinct sequence.

    Rules from ``chai_lab/data/parsing/msas/aligned_pqt.py``: columns
    sequence, source_database, pairing_key, comment; the first row is the
    query; source_database is one of query, uniprot, uniref90, bfd_uniclust,
    mgnify. Chai looks for ``sha256(sequence).aligned.pqt`` in
    ``--msa-directory`` and, if the file is missing, only logs a line and
    runs the chain without an MSA, so ``prepare`` checks every name. Rows
    paired by species share a ``pairing_key``. Needs pyarrow.
    """

    @staticmethod
    def prepare(results: Results, sequence: Any = None, *, out_dir: Any, quiet: bool = False) -> Path:
        """Write the parquet files into ``out_dir`` and return it: pass it to
        ``chai-lab fold --msa-directory``."""
        chains = _chains(results, sequence)
        d = _out(out_dir)
        msas = [m for m, _ in _distinct(chains)]
        pr = _pairing_note(msas, quiet)
        written = writers.write_chai(chains, d, pairing=pr)
        import pyarrow.parquet as pq

        for m, p in zip(chains, written):
            if p.name != writers.chai_expected_basename(m.query) or not p.is_file():
                raise PrepareError(f"Chai: {p.name} is not the file Chai-1 looks for")
            t = pq.read_table(p)
            if t.num_rows == 0:
                raise PrepareError(f"Chai: {p.name} has no rows")
            col = t.column("sequence").to_pylist()
            if col[0] != m.query or t.column("source_database").to_pylist()[0] != "query":
                raise PrepareError(f"Chai: the first row of {p.name} is not the query")
        return d

    @staticmethod
    def fasta(results: Results, sequence: Any = None) -> str:
        """The FASTA input for ``chai-lab fold`` (``>protein|name=A``)."""
        chains = _chains(results, sequence)
        return "".join(f">protein|name={chain_id(i)}\n{m.query}\n" for i, m in enumerate(chains))


# --------------------------------------------------------------------------- ColabFold


class ColabFold:
    """One A3M for ``colabfold_batch``.

    Rules from ``colabfold/input.py`` and ``colabfold/batch.py``: a monomer
    is a plain A3M; a homo-oligomer starts with ``#<len>\\t<copies>``; a
    complex uses the ``#<lens>\\t<copies>`` layout, with rows paired by
    species first and each chain's own rows padded with gaps.
    """

    @staticmethod
    def prepare(results: Results, sequence: Any = None, *, out_dir: Any, name: str = "query",
                quiet: bool = False) -> Path:
        """Write ``<name>.a3m`` into ``out_dir`` and return its path."""
        chains = _chains(results, sequence)
        d = _out(out_dir)
        msas = [m for m, _ in _distinct(chains)]
        pr = _pairing_note(msas, quiet)
        p = writers.write_colabfold(chains, d / f"{_safe_name(name)}.a3m", pairing=pr)
        if len(msas) == 1:
            _verify_a3m(p, msas[0].query, what="ColabFold")
        else:
            try:
                _, _, queries, _ = parse_colabfold_complex(p.read_text(encoding="utf-8"))
            except A3MError as e:
                raise PrepareError(f"ColabFold: {e}") from None
            if queries != [m.query for m in msas]:
                raise PrepareError("ColabFold: the complex file's queries are not the chain sequences")
        return p


# --------------------------------------------------------------------------- OpenFold3


class OpenFold3:
    """OpenFold3 MSA directories and the query JSON.

    Rules from ``openfold3`` 0.5.1 (``inference_query_format.py`` and
    ``dataset_config_components.py``): each chain names a directory in
    ``main_msa_file_paths``; a file there is used only if its stem is a
    known name, so the unpaired rows go in ``colabfold_main.a3m``. Rows
    paired by species go in ``colabfold_paired.a3m`` under
    ``paired_msa_file_paths``, the same number of rows in every chain.
    """

    @staticmethod
    def prepare(results: Results, sequence: Any = None, *, out_dir: Any, name: str = "query",
                quiet: bool = False) -> dict:
        """Write the MSA directories into ``out_dir``; return the query JSON
        as a dict (``{"queries": {name: {"chains": [...]}}}``)."""
        chains = _chains(results, sequence)
        d = _out(out_dir)
        dist = _distinct(chains)
        msas = [m for m, _ in dist]
        all_uniprot = all(m.uniprot_rows and len(m.uniprot_rows) > 1 for m in msas)
        pr = _pairing_note(msas, quiet) if not all_uniprot else None
        fields = writers.write_openfold3(chains, d, pairing=pr)
        by_seq = {m.query: f for m, f in zip(chains, fields)}
        out_chains = []
        n_paired = set()
        for m, idx in dist:
            f = by_seq[m.query]
            main = Path(f["main_msa_file_paths"])
            _verify_a3m(main / "colabfold_main.a3m", m.query, what="OpenFold3")
            entry: Dict[str, Any] = {
                "molecule_type": "protein",
                "chain_ids": [chain_id(i) for i in idx],
                "sequence": m.query,
                "main_msa_file_paths": str(main),
            }
            if "paired_msa_file_paths" in f:
                pdir = Path(f["paired_msa_file_paths"])
                n_paired.add(_verify_a3m(pdir / "colabfold_paired.a3m", m.query, what="OpenFold3"))
                entry["paired_msa_file_paths"] = str(pdir)
            out_chains.append(entry)
        if len(n_paired) > 1:
            raise PrepareError("OpenFold3: the paired files do not have the same number of rows")
        return {"queries": {name: {"chains": out_chains}}}


# --------------------------------------------------------------------------- Protenix


#: Header patterns Protenix reads species from, copied from protenix 2.0.0
#: ``protenix/data/msa/msa_utils.py`` (``_UNIPROT_REGEX``, ``_UNIREF_REGEX``).
PROTENIX_UNIPROT = re.compile(r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})")
PROTENIX_UNIREF = re.compile(r"^UniRef100_[^_]+_([^_/]+)")
PROTENIX_FILES = ("pairing.a3m", "non_pairing.a3m")


def protenix_species(header: str) -> str:
    """The species id Protenix reads from a header ("" for none), following
    ``MSAPairingEngine.get_species_ids`` in protenix 2.0.0."""
    d = header.strip()
    m = PROTENIX_UNIPROT.match(d) or PROTENIX_UNIREF.match(d)
    if not m:
        return ""
    return m.group("SpeciesId") if "SpeciesId" in m.groupdict() else m.group(1)


def _protenix_pairing_rows(m: Alignment) -> List[Row]:
    """Query, then the rows with a species, with headers Protenix can read.
    A header Protenix already reads a species from is kept. Otherwise a row
    with a taxid becomes ``UniRef100_<acc>_<taxid>/``, as Protenix's own
    ColabFold post-processing writes it."""
    src = m.uniprot_rows if m.uniprot_rows and len(m.uniprot_rows) > 1 else m.rows
    out = [Row("query", m.query, "query")]
    seen = {m.query}
    for r in src[1:]:
        k = species_key(r)
        if k is None:
            continue
        resid = r.seq.replace("-", "").upper()
        if resid in seen:
            continue
        if protenix_species(r.header):
            header = r.header
        elif k.startswith("tx:"):
            acc = r.header.split(None, 1)[0] if r.header.strip() else "hit"
            if acc.startswith("UniRef100_"):
                acc = acc[len("UniRef100_"):]
            acc = re.sub(r"[_/\s]", "-", acc) or "hit"
            header = f"UniRef100_{acc}_{k[3:]}/"
        else:
            header = r.header
        if not protenix_species(header):
            continue
        seen.add(resid)
        out.append(Row(header, r.seq, r.db, r.taxid))
    return out


class Protenix:
    """Protenix input JSON and per-chain MSA files.

    Rules from protenix 2.0.0 (``runner/msa_search.py``,
    ``protenix/data/msa/msa_featurizer.py``, ``msa_utils.py``): the input is a
    JSON list of jobs; each ``proteinChain`` has ``sequence``, ``count`` and
    ``unpairedMsaPath`` / ``pairedMsaPath``. The paths are opened as given,
    so relative paths depend on the directory Protenix is run from; they are
    written absolute here. Protenix pairs ``pairedMsaPath`` rows itself by
    the species in their headers (UniProt ``tr|ACC|NAME_SPECIES``, or
    ``UniRef100_<acc>_<taxid>``), and only for complexes of different chains.

    Ways Protenix goes wrong without saying so, which ``check`` turns into
    errors:

    - With the older ``msa: {precomputed_msa_dir}`` form it reads only
      ``pairing.a3m`` and ``non_pairing.a3m``; a file under another name is
      ignored.
    - A ``*MsaPath`` that does not exist makes it start a new MSA search for
      that chain, which sends the sequence to a server.
    - It checks only the length of the first row, not that it is the chain's
      sequence.
    """

    @staticmethod
    def prepare(results: Results, sequence: Any = None, *, out_dir: Any, name: str = "query",
                quiet: bool = False) -> List[dict]:
        """Write ``<out_dir>/<seq_sha256>/non_pairing.a3m`` (and
        ``pairing.a3m`` for a complex whose chains share species) and return
        the input as a list with one job, ready for ``json.dump``."""
        chains = _chains(results, sequence)
        d = _out(out_dir)
        dist = _distinct(chains)
        msas = [m for m, _ in dist]
        pr = _pairing_note(msas, quiet)
        pairing: Dict[str, List[Row]] = {}
        if pr is not None and pr.n_paired:
            pairing = {m.query: _protenix_pairing_rows(m) for m in msas}
            if any(len(rows) < 2 for rows in pairing.values()):
                pairing = {}
                _say("Protenix gets no pairing.a3m: some chain has no rows with a species Protenix can read", quiet)
        seqs = []
        for m, idx in dist:
            cdir = d / m.seq_sha256
            unpaired = cdir / "non_pairing.a3m"
            writers._atomic_write(unpaired, m.a3m())
            pc: Dict[str, Any] = {"sequence": m.query, "count": len(idx), "unpairedMsaPath": str(unpaired)}
            if m.query in pairing:
                paired = cdir / "pairing.a3m"
                writers._atomic_write(paired, format_a3m(pairing[m.query]))
                pc["pairedMsaPath"] = str(paired)
            seqs.append({"proteinChain": pc})
        writers._sidecar(d, chains, [pr.message] if pr is not None else [],
                         {m.seq_sha256: [n for n in PROTENIX_FILES if (d / m.seq_sha256 / n).is_file()] for m in msas})
        jobs = [{"name": name, "sequences": seqs}]
        Protenix.check(jobs)
        return jobs

    @staticmethod
    def check(jobs: Any) -> None:
        """Check a Protenix input (a list of jobs, one job dict, or a path to
        the JSON). Raises PrepareError listing every problem."""
        if isinstance(jobs, (str, os.PathLike)):
            jobs = json.loads(Path(jobs).read_text(encoding="utf-8"))
        if isinstance(jobs, dict):
            jobs = [jobs]
        if not isinstance(jobs, list) or not jobs:
            raise PrepareError("a Protenix input is a JSON list of jobs")
        problems: List[str] = []
        for j, job in enumerate(jobs, 1):
            seqs = job.get("sequences") if isinstance(job, dict) else None
            if not isinstance(seqs, list):
                problems.append(f"job {j}: no 'sequences' list")
                continue
            for e, ent in enumerate(seqs, 1):
                pc = ent.get("proteinChain") if isinstance(ent, dict) else None
                if not isinstance(pc, dict):
                    continue
                problems += _check_protenix_chain(pc, f"job {j} entity {e}")
        if problems:
            raise PrepareError("Protenix input problems:\n  " + "\n  ".join(problems))


def _check_protenix_chain(pc: Dict[str, Any], what: str) -> List[str]:
    out: List[str] = []
    try:
        q = canonical_sequence(str(pc.get("sequence", "")))
    except InvalidSequence as e:
        return [f"{what}: {e}"]
    files: List[Tuple[str, Path]] = []
    if "msa" in pc:
        info = pc.get("msa") or {}
        mdir = info.get("precomputed_msa_dir") if isinstance(info, dict) else None
        if not mdir or not Path(mdir).is_dir():
            return [f"{what}: msa.precomputed_msa_dir {mdir!r} is not a directory"]
        present = sorted(p.name for p in Path(mdir).iterdir() if p.is_file() and p.name.endswith((".a3m", ".a3m.gz", ".sto")))
        stray = [n for n in present if n not in PROTENIX_FILES]
        if stray:
            out.append(f"{what}: Protenix reads only pairing.a3m and non_pairing.a3m in {mdir}; "
                       f"it would ignore {', '.join(stray)}")
        found = [n for n in PROTENIX_FILES if n in present]
        if not found:
            out.append(f"{what}: {mdir} has neither pairing.a3m nor non_pairing.a3m")
        files += [("pairedMsaPath" if n == "pairing.a3m" else "unpairedMsaPath", Path(mdir) / n) for n in found]
    for key, want in (("pairedMsaPath", "pairing.a3m"), ("unpairedMsaPath", "non_pairing.a3m")):
        v = pc.get(key)
        if v is None:
            continue
        p = Path(v)
        other = PROTENIX_FILES[1] if want == PROTENIX_FILES[0] else PROTENIX_FILES[0]
        if p.name == other:
            out.append(f"{what}: {key} points at {p.name}; the paired and unpaired files are swapped")
        files.append((key, p))
    if not files and pc.get("unpairedMsa") is None and pc.get("pairedMsa") is None:
        out.append(f"{what}: no MSA given, so Protenix would run its own search and send the sequence to a server")
    for key, p in files:
        if not p.is_file():
            out.append(f"{what}: {key} {p} does not exist; Protenix would start a new MSA search for this chain")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            out.append(f"{what}: {key} {p} cannot be read: {e}")
            continue
        if not text.strip():
            out.append(f"{what}: {key} {p} is empty")
            continue
        try:
            parse_a3m(text, q, what=f"{key} {p.name}")
        except A3MError as e:
            out.append(f"{what}: {e}")
    for key in ("unpairedMsa", "pairedMsa"):
        v = pc.get(key)
        if isinstance(v, str) and v.strip():
            try:
                parse_a3m(v, q, what=key)
            except A3MError as e:
                out.append(f"{what}: {e}")
    return out


def _safe_name(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip(".")
    return s[:100] or "query"
