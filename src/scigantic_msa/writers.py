"""Low-level writers behind :mod:`scigantic_msa.adapters`.

Every writer that produces files also writes ``scigantic-msa-attribution.json``
next to them: where each alignment came from, its terms, and notes about
pairing.

Format rules are from the predictors' own parsers: AlphaFold 3
``folding_input.py``, Boltz ``parse/csv.py`` and ``main.py``, Chai-1
``aligned_pqt.py``, ColabFold ``input.py`` and OpenFold3
``dataset_config_components.py``.

A chain with no species information has ``None`` for every paired row (see
:mod:`scigantic_msa.pairing`). Each writer fills that in as its model does
for a chain missing from a species: gaps in the ColabFold paired rows and in
OpenFold3's ``colabfold_paired.a3m`` (both as ColabFold's own pairing writes
them), one gap row per key for Chai-1 (it pairs a key only when every chain
with an MSA has it), no key for Boltz (it fills the chain with its next
unpaired row), and nothing for AlphaFold 3 (its unpaired rows stay as they
are).

Every MSA file is named by its sequence and a hash of its content, or, where
the model fixes the file name (Chai-1, OpenFold3, Protenix), put in a folder
named for the content. So a second complex sharing a chain (A+B, then A+C or
A alone), or another alignment of the same sequence, written into the same
folder never rewrites a file the first one points at, and writes with the
same content share one file.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import secrets
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .a3m import Row, format_a3m, ungapped
from .alignment import Alignment, dedupe_rows
from .pairing import Pairing, pair_by_species
from .seqkey import canonical_sequence

__all__ = [
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
#: Chai-1 pairs at most this many rows, the query included
#: (``MAX_PAIRED_DEPTH`` in chai_lab's msas/preprocess.py).
CHAI_MAX_PAIRED = 8192
#: Boltz pairs at most this many rows, the query included
#: (``max_pairs`` of ``construct_paired_msa`` in boltz's featurizerv2.py).
BOLTZ_MAX_PAIRED = 8192


def _atomic_write(path: Path, data: "str | bytes") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not mkstemp: it always makes the file 0600, so a model running as
    # another user could not read it. os.open with 0666 lets the umask
    # decide, as for any other new file.
    while True:
        tmp = str(path.parent / f".{path.name}.{secrets.token_hex(6)}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
            break
        except FileExistsError:
            continue
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


def entities(chains: Sequence[Alignment]) -> List[Tuple[Alignment, int, List[int]]]:
    """Group chains with the same sequence: (msa, copies, chain indexes)."""
    out: Dict[str, Tuple[Alignment, List[int]]] = {}
    for i, m in enumerate(chains):
        if m.query in out:
            out[m.query][1].append(i)
        else:
            out[m.query] = (m, [i])
    return [(m, len(idx), idx) for m, idx in out.values()]


def _pairing(ents: Sequence[Alignment], pairing: Optional[Pairing]) -> Pairing:
    if pairing is not None:
        return pairing
    return pair_by_species(ents)


def _sidecar(directory: Path, chains: Sequence[Alignment], notes: List[str], files: Dict[str, Any]) -> None:
    """Add this write to the attribution file in ``directory``.

    Several writes can share a folder (two jobs, two models from Python, a
    rerun with another name), so the file keeps one entry per write. A new
    entry replaces an older one only when it rewrote every file the older
    one named; otherwise both stay, so no source is ever dropped while one
    of its files is still there. ``sources`` lists every distinct credit."""
    entry = {
        "chains": [
            {
                "seq_sha256": m.seq_sha256,
                "length": len(m.query),
                "source": m.source,
                "summary": m.summary(),
                "depth": m.depth,
                "attribution": m.attribution,
                "provenance": m.provenance,
            }
            for m in chains
        ],
        "files": files,
        "notes": notes,
    }
    path = directory / ATTRIBUTION_FILE
    # Imported here: cache.py is the only other user of the lock.
    from .cache import FileLock

    lock = FileLock(path, timeout=120, poll=0.05)
    lock.acquire()
    try:
        mine = _file_names(files)
        kept = [e for e in _read_sidecar(path) if not _file_names(e.get("files")) <= mine]
        outputs = kept + [entry]
        sources: List[Any] = []
        for e in outputs:
            # An entry from an older or hand-edited file can hold anything,
            # so take only the credits that have the shape this code writes.
            for c in _list(e.get("chains")):
                for a in _list(c.get("attribution") if isinstance(c, dict) else None):
                    if isinstance(a, dict) and a not in sources:
                        sources.append(a)
        data = {"written_by": "scigantic-msa", "format": 2, "sources": sources, "outputs": outputs}
        _atomic_write(path, json.dumps(data, indent=1, sort_keys=True) + "\n")
    finally:
        lock.release()


def _list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _read_sidecar(path: Path) -> List[Dict[str, Any]]:
    """The entries already in an attribution file. A file from 0.2.3 or
    earlier (one write, at the top level) becomes one entry. A missing file
    gives none. So does a file that cannot be read or is not one this code
    wrote, and since the write that follows replaces it, that is said on
    stderr: the credits in it are lost."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        why = getattr(e, "strerror", None) or ("not valid JSON" if isinstance(e, ValueError) else type(e).__name__)
        return _unreadable(path, why)
    if isinstance(data, dict) and isinstance(data.get("outputs"), list):
        return [e for e in data["outputs"] if isinstance(e, dict)]
    if isinstance(data, dict) and ("chains" in data or "files" in data):
        return [{k: data.get(k) for k in ("chains", "files", "notes")}]
    return _unreadable(path, "not an attribution file this tool wrote")


def _unreadable(path: Path, why: str) -> List[Dict[str, Any]]:
    print(f"scigantic-msa: warning: replacing {path} ({why}); any credits it held are not kept.",
          file=sys.stderr, flush=True)
    return []


def _file_names(files: Any, under: Tuple[str, ...] = ()) -> frozenset:
    """Every file in a ``files`` map, each with the keys that lead to it, so
    ``non_pairing.a3m`` of one chain is not taken for another chain's."""
    if isinstance(files, dict):
        return frozenset(f for k, v in files.items() for f in _file_names(v, under + (str(k),)))
    if isinstance(files, (list, tuple)):
        return frozenset(f for v in files for f in _file_names(v, under))
    return frozenset([under + (str(files),)]) if files is not None else frozenset()


def _tag(data: "str | bytes") -> str:
    """A short hash of a file's content, for the name of a file that only
    one complex can use."""
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:16]


def _gap_row(m: Alignment, k: int) -> Row:
    """A gap-only row standing in for a chain that has no row for pair ``k``."""
    return Row(f"gap_{k} no species information", "-" * len(m.query), "gap")


@dataclass
class AF3Layout:
    """Where each chain's rows go in AlphaFold 3's ``unpairedMsa``.

    ``rows`` maps a chain's sequence to its rows, query first. ``gaps``
    maps the label of a chain to the gap rows it needed. ``n_paired`` is
    how many rows pair at their own index in every chain that has them,
    which is every pair."""

    rows: Dict[str, List[Row]]
    gaps: Dict[str, int]
    n_paired: int

    def note(self) -> str:
        """What the layout costs, for the user; "" when nothing."""
        if not self.gaps:
            return ""
        many = len(self.gaps) > 1
        who = "; ".join(f"{'chains' if ', ' in c else 'chain'} {c}: {n}" for c, n in self.gaps.items())
        return (f"AlphaFold 3 joins unpaired rows by index, so a chain without a row of a paired species "
                f"needs a row there; {'these chains have' if many else 'this chain has'} too few rows of "
                f"{'their' if many else 'its'} own and {'get' if many else 'gets'} gap rows instead "
                f"({who}), which count in {'their' if many else 'its'} profile. "
                f"All {self.n_paired} paired rows stay at their own row")


def af3_layout(msas: Sequence[Alignment], pr: Pairing, names: Optional[Sequence[str]] = None) -> AF3Layout:
    """Lay out the paired rows of ``pr`` for AlphaFold 3.

    AlphaFold 3 joins the ``unpairedMsa`` rows of all chains by row index
    (``merge_msa_features`` in its ``merging_features.py``), and reads a
    user MSA as given (``deduplicate=False`` in ``features.py``). So pair
    k must be at the same row in every chain that has it. A chain without
    a row for pair k puts its next unpaired row there, as Boltz does in
    ``construct_paired_msa``, and a gap row only once it has none left, as
    Boltz also does: a gap row counts in that chain's profile, but a paired
    row at another pair's index would pair it with the wrong species.

    To need as few gap rows as possible, the pairs are reordered when the
    given order would need any: pairs are taken in groups of the chains
    that share them, and the order of the groups with the fewest gap rows
    is used, so the rows a shallow chain takes part in come first. A
    chain with no species keeps its rows as they are."""
    label = list(names) if names is not None else [str(i + 1) for i in range(len(msas))]
    n = pr.n_paired
    rows: Dict[str, List[Row]] = {m.query: dedupe_rows(m.rows) for m in msas}
    if not n:
        return AF3Layout(rows, {}, 0)
    live = [c for c in range(len(msas)) if c not in pr.chains_without_species]
    held = {c: [k for k in range(n) if pr.rows[c][k] is not None] for c in live}
    spare: Dict[int, List[Row]] = {}
    for c in live:
        m = msas[c]
        # A row with the residues of one of the chain's paired rows would
        # repeat it, so it never stands in.
        keys = {ungapped(r.seq) for r in pr.rows[c] if r is not None}
        spare[c] = [r for r in rows[m.query][1:] if ungapped(r.seq) not in keys]

    def cost(order: Sequence[int]) -> int:
        pos = {k: i for i, k in enumerate(order)}
        total = 0
        for c in live:
            if held[c]:
                need = max(pos[k] for k in held[c]) + 1 - len(held[c])
                total += max(0, need - len(spare[c]))
        return total

    order = list(range(n))
    now = cost(order)
    if now:
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for k in range(n):
            groups.setdefault(tuple(c for c in live if pr.rows[c][k] is not None), []).append(k)
        gs = list(groups.items())

        def group_cost(perm: Sequence[Tuple[Tuple[int, ...], List[int]]]) -> int:
            # The same as cost() of the keys group by group, without
            # walking every key.
            total = 0
            for c in live:
                last = max((i for i, (who, _) in enumerate(perm) if c in who), default=-1)
                need = sum(len(ks) for who, ks in perm[:last] if c not in who)
                total += max(0, need - len(spare[c]))
            return total

        if len(gs) <= 7:
            candidates: Any = itertools.permutations(gs)
        else:
            candidates = [gs, sorted(gs, key=lambda g: min(len(spare[c]) for c in g[0]))]
        best = None
        for perm in candidates:
            c_ = group_cost(perm)
            if best is None or c_ < best[0]:
                best = (c_, perm)
                if not c_:
                    break
        assert best is not None
        if best[0] < now:
            order = [k for _, ks in best[1] for k in ks]
    gaps: Dict[str, int] = {}
    for c in live:
        m = msas[c]
        out = [rows[m.query][0]]
        rest = iter(spare[c])
        last = max((i for i, k in enumerate(order) if pr.rows[c][k] is not None), default=-1)
        g = 0
        for i in range(last + 1):
            r = pr.rows[c][order[i]]
            if r is None:
                r = next(rest, None)
            if r is None:
                g += 1
                r = Row(f"gap_{i + 1} keeps paired rows aligned", "-" * len(m.query), "gap")
            out.append(r)
        out += list(rest)
        rows[m.query] = out
        if g:
            gaps[label[c]] = g
    return AF3Layout(rows, gaps, n)


# --------------------------------------------------------------------------- ColabFold


def write_colabfold(chains: Sequence[Alignment], path: "str | Path", *, pairing: Optional[Pairing] = None) -> Path:
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
            row = [pr.rows[c][k] or _gap_row(msas[c], k + 1) for c in range(len(msas))]
            out.append(">" + "\t".join(r.header.split("\t")[0] for r in row))
            out.append("".join(r.seq for r in row))
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
    chains: Sequence[Alignment],
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
    next to the JSON and uses the ``*Path`` fields, which need ``path``. An
    unpaired file with rows paired for this complex is
    ``<seq_sha256>.<hash>.unpaired.a3m``, so another complex cannot rewrite it.
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
                lay = af3_layout(msas, pr, _entity_names(ents))
                unpaired_rows = lay.rows
                notes.append(
                    f"the {lay.n_paired} rows paired by taxid are first in unpairedMsa, each at the same row in "
                    "every chain that has it; run AlphaFold 3 with --resolve_msa_overlaps=false so it keeps them"
                )
                if lay.note():
                    notes.append(lay.note())
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
        # AlphaFold 3 checks that the MSA's first row equals the sequence.
        prot["sequence"] = q
        for k in ("unpairedMsa", "unpairedMsaPath", "pairedMsa", "pairedMsaPath"):
            prot.pop(k, None)
        up = format_a3m(unpaired_rows[q])
        pa = paired_text.get(q, "")
        if inline:
            prot["unpairedMsa"] = up
            prot["pairedMsa"] = pa
        else:
            assert out_path is not None
            # Named for the content, so another alignment of this sequence,
            # or rows placed for another complex, get a file of their own.
            fn = f"{m.seq_sha256}.{_tag(up)}.unpaired.a3m"
            _atomic_write(out_path.parent / fn, up)
            prot["unpairedMsaPath"] = fn
            files[m.seq_sha256] = {"unpaired": fn}
            if pa:
                pfn = f"{m.seq_sha256}.{_tag(pa)}.paired.a3m"
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


def _entity_names(ents: Sequence[Tuple[Alignment, int, List[int]]]) -> List[str]:
    """For each distinct sequence, the chain ids that have it ("B, D")."""
    ids = list(itertools.islice(_chain_ids(), max(i for _, _, idx in ents for i in idx) + 1))
    return [", ".join(ids[i] for i in idx) for _, _, idx in ents]


def _chain_ids():
    for c in string.ascii_uppercase:
        yield c
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            yield a + b


# --------------------------------------------------------------------------- Boltz


def write_boltz(chains: Sequence[Alignment], directory: "str | Path", *, pairing: Optional[Pairing] = None) -> List[str]:
    """Write Boltz MSA files and return one YAML ``msa:`` line per chain.

    A single distinct sequence gets an ``.a3m``. Several get a ``.csv``
    (header ``key,sequence``): the query has key -1, rows paired by species
    get the pair index (Boltz pairs equal keys), the rest -1. Every file is
    ``<seq_sha256>.<hash>.<ext>``, the hash of its content: pair keys mean
    something only next to one complex's other files, and another alignment
    of the same sequence must not rewrite this one. Identical files share a
    name. Boltz refuses ``.a3m.gz``, so files are never compressed.
    """
    d = Path(directory)
    ents = entities(chains)
    msas = [e[0] for e in ents]
    notes: List[str] = []
    paths: Dict[str, Path] = {}
    if len(msas) == 1:
        m = msas[0]
        text = m.a3m()
        p = d / f"{m.seq_sha256}.{_tag(text)}.a3m"
        _atomic_write(p, text)
        paths[m.query] = p
    else:
        pr = _pairing(msas, pairing)
        notes.append(pr.message)
        for c, m in enumerate(msas):
            lines = ["key,sequence", f"-1,{m.query}"]
            # Pairs past Boltz's cap stay in the file as unpaired rows:
            # Boltz drops a keyed row it does not pair.
            keyed = [r for r in pr.rows[c][:BOLTZ_MAX_PAIRED - 1]]
            for k, r in enumerate(keyed, 1):
                if r is not None:
                    lines.append(f"{k},{r.seq}")
            used = {id(r) for r in keyed if r is not None}
            for r in m.rows[1:]:
                if id(r) not in used:
                    lines.append(f"-1,{r.seq}")
            text = "\n".join(lines) + "\n"
            p = d / f"{m.seq_sha256}.{_tag(text)}.csv"
            _atomic_write(p, text)
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


def write_chai(chains: Sequence[Alignment], directory: "str | Path", *, pairing: Optional[Pairing] = None) -> List[Path]:
    """Write ``<seq_sha256>.aligned.pqt`` per distinct sequence, the file
    Chai-1 looks up in ``--msa-directory``. Needs pyarrow
    (``pip install 'scigantic-msa[chai]'``).

    The files go in a folder named for their content,
    ``<directory>/paired-<hash>/`` for a complex with paired rows and
    ``<directory>/unpaired-<hash>/`` otherwise: Chai-1 finds a file only by
    the sequence, so another alignment of a chain, or another complex
    sharing it, must not rewrite a file this one reads. Identical files
    share a folder. Pass the folder of the returned paths to
    ``--msa-directory``."""
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
    columns: List[Dict[str, List[str]]] = []
    for c, m in enumerate(msas):
        keyed: Dict[int, str] = {}
        rows = [m.rows[0]]
        if pr is not None:
            # Pairs past Chai's cap stay in the file as unpaired rows.
            for k, r in enumerate(pr.rows[c][:CHAI_MAX_PAIRED - 1], 1):
                r = r or _gap_row(m, k)
                keyed[id(r)] = str(k)
                rows.append(r)
        used = {id(r) for r in rows}
        rows += [r for r in m.rows[1:] if id(r) not in used]
        columns.append({
            "sequence": [r.seq for r in rows],
            "source_database": ["query" if i == 0 else chai_source(r) for i, r in enumerate(rows)],
            "pairing_key": [keyed.get(id(r), "") for r in rows],
            "comment": [r.header for r in rows],
        })
    # Chai-1 finds a file only by its sequence, in one folder, so each set
    # of files gets a folder named for its content.
    kind = "paired" if pr is not None and pr.n_paired else "unpaired"
    where = d / f"{kind}-{_tag(json.dumps([[m.seq_sha256, col] for m, col in zip(msas, columns)]))}"
    out: Dict[str, Path] = {}
    for m, col in zip(msas, columns):
        name = chai_expected_basename(m.query)
        if name != m.seq_sha256 + ".aligned.pqt":
            raise AssertionError("Chai-1 file name does not match seq_sha256")
        table = pa.table({k: pa.array(v, pa.string()) for k, v in col.items()})
        p = where / name
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        # A unique temp name, so writers sharing one msas folder do not
        # write into each other's file.
        _atomic_write(p, buf.getvalue().to_pybytes())
        out[m.query] = p
    _sidecar(d, chains, notes, {m.seq_sha256: os.path.relpath(out[m.query], d) for m in msas})
    return [out[m.query] for m in chains]


# --------------------------------------------------------------------------- OpenFold3


def write_openfold3(chains: Sequence[Alignment], directory: "str | Path", *, pairing: Optional[Pairing] = None) -> List[dict]:
    """Write OpenFold3 MSA directories and return, per chain, the fields to
    put in that chain of the query JSON.

    ``<dir>/main/<seq_sha256>/<hash>/colabfold_main.a3m`` holds the unpaired
    rows, plus ``uniprot_hits.a3m`` when UniProt rows exist (OpenFold3 pairs
    those by species itself). ``<hash>`` is of the folder's content, so
    another alignment of the sequence gets a folder of its own. If some chain has no UniProt rows, rows paired by
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
            texts = [format_a3m([m.rows[0]] + [r or _gap_row(m, k) for k, r in enumerate(pr.rows[c], 1)])
                     for c, m in enumerate(msas)]
            # Named for the content: a complex with the same chains but
            # other rows must not rewrite one of these files and leave the
            # others, since every chain's file needs the same row count.
            cid = _tag("|".join(m.seq_sha256 + "\n" + t for m, t in zip(msas, texts)))
            for m, text in zip(msas, texts):
                pdir = d / "paired" / cid / m.seq_sha256
                _atomic_write(pdir / "colabfold_paired.a3m", text)
                paired_dirs[m.query] = pdir
    elif len(msas) > 1:
        notes.append("uniprot_hits.a3m written for every chain; OpenFold3 pairs them by species")
    for m in msas:
        main = {"colabfold_main.a3m": m.a3m()}
        if m.uniprot_rows and len(m.uniprot_rows) > 1 and (all_uniprot or len(msas) == 1):
            main["uniprot_hits.a3m"] = format_a3m(m.uniprot_rows or [])
        # OpenFold3 reads every file in the folder, and the file names are
        # fixed, so each set of files gets a folder named for its content.
        mdir = d / "main" / m.seq_sha256 / _tag(json.dumps(main, sort_keys=True))
        for fname, text in main.items():
            _atomic_write(mdir / fname, text)
        f = {"main_msa_file_paths": str(mdir)}
        if m.query in paired_dirs:
            f["paired_msa_file_paths"] = str(paired_dirs[m.query])
        fields[m.query] = f
        files[m.seq_sha256] = {k: os.path.relpath(v, d) for k, v in f.items()}
    _sidecar(d, chains, notes, files)
    return [fields[m.query] for m in chains]
