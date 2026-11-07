"""get_msa, get_msas and where.

Order of lookups for one sequence:

1. cache: alignments this machine computed or downloaded before.
2. exact: the public index of published MSAs. Only the first three hex
   digits of the hash reach the index host.
3. near: a near-identical indexed sequence (point mutant or small indel),
   searched on this machine. Rows are rebuilt for your query and labelled.
4. remote: a search server, only when you turned one on. The sequence is
   sent to that server.

``where`` walks the same steps without computing or sending anything.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .a3m import A3MError
from .assemble import EntryUnusable, load_entry, order_entries
from .cache import AlignmentCache
from .index import Index, IndexIntegrityError, IndexUnavailable
from .msa import MSA
from .near import find_near, remap_rows
from .pairing import Pairing, pair_by_species
from .remote import BACKENDS, announce, make_backend, remote_disabled
from .remote.colabfold import RemoteError
from .seqkey import canonical_sequence, seq_sha256

log = logging.getLogger("scigantic_msa")

__all__ = ["get_msa", "get_msas", "where", "Complex", "MSANotFound", "Step", "resolve_remote"]


class MSANotFound(LookupError):
    def __init__(self, message: str, steps: "List[Step]"):
        super().__init__(message)
        self.steps = steps


@dataclass
class Step:
    name: str
    outcome: str  # hit, miss, skipped, error, would-run
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.name:<8} {self.outcome:<10} {self.detail}"


RemoteArg = Union[None, bool, str, Any]


def resolve_remote(remote: RemoteArg) -> Optional[Any]:
    """The remote backend to use, or None. Explicit False always wins, and
    SCIGANTIC_MSA_NO_REMOTE=1 wins over everything."""
    if remote_disabled() or remote is False:
        return None
    if remote is None:
        name = os.environ.get("SCIGANTIC_MSA_REMOTE")
        if not name or str(name).lower() in ("off", "none", "false", "0"):
            return None
        return make_backend(str(name).lower())
    if remote is True:
        name = os.environ.get("SCIGANTIC_MSA_REMOTE") or "colabfold"
        if str(name).lower() in ("off", "none", "false", "0"):
            name = "colabfold"
        return make_backend(str(name).lower())
    if isinstance(remote, str):
        if remote.lower() not in BACKENDS:
            raise ValueError(f"unknown remote backend {remote!r}; choose one of {', '.join(BACKENDS)}")
        return make_backend(remote.lower())
    if hasattr(remote, "search") and hasattr(remote, "name"):
        return remote
    raise TypeError("remote must be None, True, False, a backend name, or a backend object")


def _print(msg: str, quiet: bool) -> None:
    log.info(msg)
    if not quiet:
        print(f"scigantic-msa: {msg}", file=sys.stderr, flush=True)


def _resolve(
    sequence: str,
    *,
    remote: RemoteArg,
    near: bool,
    index: Union[Index, None, bool],
    cache: Union[AlignmentCache, None, bool],
    purpose: str,
    want_uniprot: bool,
    min_identity: float,
    max_length_diff: int,
    near_method: str,
    dry_run: bool,
    quiet: bool,
) -> "tuple[Optional[MSA], List[Step]]":
    q = canonical_sequence(sequence)
    sha = seq_sha256(q)
    steps: List[Step] = []

    # 1. local cache
    ac: Optional[AlignmentCache] = None
    if cache is not False:
        ac = cache if isinstance(cache, AlignmentCache) else AlignmentCache()
        hit = ac.get(sha)
        if hit is not None and (not want_uniprot or hit.uniprot_rows or hit.provenance.get("backend")):
            steps.append(Step("cache", "hit", f"{hit.provenance.get('cache_file')} ({hit.provenance.get('backend') or hit.provenance.get('stored_from') or 'stored'})"))
            return hit, steps
        steps.append(Step("cache", "miss", str(ac.root)))
    else:
        steps.append(Step("cache", "skipped", "cache=False"))

    # 3. exact index match
    idx: Optional[Index] = None
    if index is not False:
        idx = index if isinstance(index, Index) else Index()
        try:
            entries = idx.lookup(sha)
        except IndexUnavailable as exc:
            steps.append(Step("exact", "error", f"index not reachable: {exc}"))
            idx = None
        else:
            entries = order_entries(entries, purpose)
            if entries and dry_run:
                steps.append(Step("exact", "hit", ", ".join(f"{x.source} {x.source_id} ({x.license})" for x in entries)))
                return None, steps
            for ent in entries:
                try:
                    msa = load_entry(idx, ent, q, purpose=purpose, want_uniprot=want_uniprot)
                except EntryUnusable as ex:
                    steps.append(Step("exact", "error", str(ex)))
                    _print(f"skipped an index entry that did not check out: {ex}", quiet)
                    continue
                steps.append(Step("exact", "hit", f"{ent.source} {ent.source_id}, {ent.license}"))
                return msa, steps
            if not entries:
                steps.append(Step("exact", "miss", f"index build {idx.build_id}"))
    else:
        steps.append(Step("exact", "skipped", "index=False"))

    # 4. near-identical
    if near and idx is not None:
        try:
            hits = find_near(idx, q, min_identity=min_identity, max_length_diff=max_length_diff,
                             method=near_method, exclude=[sha])
        except (IndexUnavailable, IndexIntegrityError) as exc:
            steps.append(Step("near", "error", str(exc)))
            hits = []
        if hits and dry_run:
            h = hits[0]
            steps.append(Step("near", "hit", f"parent {h.parent_seq_sha256[:12]} at {h.identity * 100:.1f}% identity "
                                             f"({len(h.mismatches)} substitutions, {h.insertions + h.deletions} indels)"))
            return None, steps
        for h in hits:
            try:
                parents = order_entries(idx.lookup(h.parent_seq_sha256), purpose)
            except IndexUnavailable as exc:
                steps.append(Step("near", "error", f"index not reachable: {exc}"))
                continue
            for ent in parents:
                try:
                    pm = load_entry(idx, ent, h.parent_seq, purpose=purpose, want_uniprot=want_uniprot)
                    rows = remap_rows(pm.rows, q, h.path)
                    urows = remap_rows(pm.uniprot_rows, q, h.path) if pm.uniprot_rows else None
                except (EntryUnusable, A3MError, ValueError) as ex:
                    steps.append(Step("near", "error", str(ex)))
                    continue
                derived = h.describe()
                derived["parent_source"] = ent.source
                derived["parent_source_id"] = ent.source_id
                msa = MSA(query=q, rows=rows, source="near", provenance=dict(pm.provenance), attribution=pm.attribution,
                          uniprot_rows=urows, derived=derived)
                steps.append(Step("near", "hit", f"parent {ent.source} {ent.source_id}, {h.identity * 100:.1f}% identity"))
                return msa, steps
        if not hits:
            steps.append(Step("near", "miss", f"nothing at or above {min_identity * 100:.0f}% identity"))
    elif not near:
        steps.append(Step("near", "skipped", "near=False"))
    else:
        steps.append(Step("near", "skipped", "no index"))

    # 5. remote
    backend = resolve_remote(remote)
    if backend is None:
        why = "SCIGANTIC_MSA_NO_REMOTE is set" if remote_disabled() else "no remote backend turned on"
        steps.append(Step("remote", "skipped", why))
        return None, steps
    if dry_run:
        steps.append(Step("remote", "would-run", f"would send the sequence to {backend.server_name} ({backend.name})"))
        return None, steps
    lock = ac.lock(sha, backend.name, backend.dbkey) if ac is not None else None
    if lock is not None:
        lock.acquire()
    try:
        if ac is not None:
            again = ac.get(sha, backend.name, backend.dbkey)
            if again is not None:
                steps.append(Step("cache", "hit", "computed by another process while waiting"))
                return again, steps
        announce(backend)
        try:
            res = backend.search(q)
        except RemoteError as e:
            steps.append(Step("remote", "error", str(e)))
            return None, steps
        msa = MSA(query=q, rows=res.rows, source="remote", provenance=dict(res.provenance), templates=res.templates,
                  attribution=[_remote_attribution(backend)])
        if ac is not None:
            msa.cache_path = str(ac.put(msa, backend.name, backend.dbkey))
        steps.append(Step("remote", "hit", f"{backend.server_name}, depth {msa.depth}"))
        return msa, steps
    finally:
        if lock is not None:
            lock.release()


def _remote_attribution(backend: Any) -> Dict[str, Any]:
    if backend.name == "colabfold":
        return {"id": "colabfold-api", "title": "ColabFold MMseqs2 server", "license": "see server terms",
                "citation": "Mirdita et al., ColabFold, Nature Methods 2022"}
    return {"id": "nvidia-msa-search", "title": "NVIDIA MSA Search", "license": "CC-BY-4.0 (NVIDIA API trial terms)"}


def get_msa(
    sequence: str,
    *,
    remote: RemoteArg = None,
    near: bool = True,
    index: Union[Index, None, bool] = None,
    cache: Union[AlignmentCache, None, bool] = None,
    purpose: str = "a3m",
    want_uniprot: bool = False,
    min_identity: float = 0.95,
    max_length_diff: int = 5,
    near_method: str = "auto",
    quiet: bool = False,
) -> MSA:
    """Find or compute the MSA for one protein sequence.

    remote: None uses SCIGANTIC_MSA_REMOTE or the config (off by default);
        True uses colabfold; "colabfold" or "nvidia" picks one; False never.
    purpose: "af3" picks the files AlphaFold 3 benefits from; anything else
        picks a single A3M.

    The result's ``.source`` says where it came from: exact, near, remote,
    or cache. A ``near`` result's ``.derived`` names the parent.
    """
    msa, steps = _resolve(
        sequence, remote=remote, near=near, index=index,
        cache=cache,
        purpose=purpose, want_uniprot=want_uniprot, min_identity=min_identity,
        max_length_diff=max_length_diff, near_method=near_method, dry_run=False, quiet=quiet,
    )
    if msa is None:
        tried = "; ".join(
            f"{s.name} {s.outcome}" + (f" ({s.detail})" if s.outcome == "error" or s.name == "remote" else "")
            for s in steps
        )
        hint = ""
        if any(s.name == "remote" and s.outcome == "skipped" for s in steps) and not remote_disabled():
            hint = (" To search a server, pass remote='colabfold' (sends the sequence to api.colabfold.com, "
                    "where results are readable by anyone who knows the sequence).")
        raise MSANotFound(f"no MSA for sequence {seq_sha256(sequence)[:12]}: {tried}.{hint}", steps)
    _print(msa.summary(), quiet)
    return msa


def where(sequence: str, **kw: Any) -> List[Step]:
    """Explain which source would answer and why, without computing or
    sending the sequence anywhere."""
    kw.setdefault("remote", None)
    for k, v in (("near", True), ("index", None), ("cache", None), ("purpose", "a3m"),
                 ("want_uniprot", False), ("min_identity", 0.95), ("max_length_diff", 5), ("near_method", "auto")):
        kw.setdefault(k, v)
    _, steps = _resolve(sequence, dry_run=True, quiet=True, **kw)
    return steps


@dataclass
class Complex:
    """MSAs for the chains of a complex, with pairing where the data allows."""

    chains: List[MSA]
    pairing: Pairing
    notes: List[str] = field(default_factory=list)

    def __iter__(self):
        return iter(self.chains)

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, i: int) -> MSA:
        return self.chains[i]

    def to_af3_json(self, path: Any = None, **kw: Any) -> dict:
        from .writers import write_af3_json

        return write_af3_json(self.chains, path, pairing=self._pairing(), **kw)

    def to_boltz(self, directory: Any) -> List[str]:
        from .writers import write_boltz

        return write_boltz(self.chains, directory, pairing=self._pairing())

    def to_chai(self, directory: Any) -> list:
        from .writers import write_chai

        return write_chai(self.chains, directory, pairing=self._pairing())

    def to_colabfold(self, path: Any) -> Any:
        from .writers import write_colabfold

        return write_colabfold(self.chains, path, pairing=self._pairing())

    def to_openfold3(self, directory: Any) -> List[dict]:
        from .writers import write_openfold3

        return write_openfold3(self.chains, directory, pairing=self._pairing())

    def _pairing(self) -> Optional[Pairing]:
        # The pairing was computed over distinct sequences in order.
        return self.pairing if len({m.query for m in self.chains}) > 1 else None


def get_msas(sequences: Sequence[str], **kw: Any) -> Complex:
    """MSAs for every chain of a complex. Identical chains share one lookup.
    Rows are paired across chains by species only where the rows carry it;
    ``.pairing.message`` says what was possible."""
    uniq: Dict[str, MSA] = {}
    order: List[str] = []
    kw.setdefault("want_uniprot", len({canonical_sequence(s) for s in sequences}) > 1)
    for s in sequences:
        q = canonical_sequence(s)
        if q not in uniq:
            uniq[q] = get_msa(q, **kw)
            order.append(q)
    chains = [uniq[canonical_sequence(s)] for s in sequences]
    pr = pair_by_species([uniq[q] for q in order])
    notes = [pr.message]
    if not kw.get("quiet"):
        print(f"scigantic-msa: {pr.message}", file=sys.stderr, flush=True)
    return Complex(chains, pr, notes)
