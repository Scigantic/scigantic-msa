"""Turn index entries into an MSA, fetching only the files needed.

Choice of files (by role, so new sources with the same roles just work):

- one A3M (ColabFold, Boltz, Chai, OpenFold3, plain A3M): a ``colabfold``
  file, else ``cfdb_uniref30``, else ``uniref90`` + ``bfd_uniclust`` +
  ``mgnify`` merged query first, else ``uniclust30``.
- AlphaFold 3: ``cfdb_uniref30`` + ``uniref90`` (up to 10,000 rows) +
  ``mgnify`` (up to 5,000), else the single-A3M choice.
- UniProt rows for pairing complexes come from the ``uniprot`` role, read up
  to a row cap. Big Stockholm files are never read whole.

Every file read is checked: the byte count must equal the index length, and
the first row must be the query. A mismatch is reported and the next entry
is tried; it never counts as a hit.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator, List, Optional, Sequence

from . import http
from .a3m import A3MError, Row, parse_a3m, parse_stockholm
from .index import FileRef, Index, IndexEntry, IndexIntegrityError, IndexUnavailable
from .msa import MSA, dedupe_rows

log = logging.getLogger("scigantic_msa")

__all__ = ["load_entry", "order_entries", "EntryUnusable", "A3M_ORDER", "AF3_ORDER"]

A3M_ORDER = ("litefold-human-colabfold", "openfold3-pdb", "openproteinset-pdb", "openproteinset-uniclust30")
AF3_ORDER = ("openfold3-pdb", "litefold-human-colabfold", "openproteinset-pdb", "openproteinset-uniclust30")
STREAM_THRESHOLD = 32 * 1024 * 1024
ROW_CAPS = {"uniref90": 10_000, "mgnify": 5_000, "uniprot": 10_000}


class EntryUnusable(Exception):
    """An index entry's files could not be read or did not match the query."""


def order_entries(entries: Sequence[IndexEntry], purpose: str) -> List[IndexEntry]:
    order = AF3_ORDER if purpose == "af3" else A3M_ORDER
    rank = {s: i for i, s in enumerate(order)}
    return sorted(entries, key=lambda e: rank.get(e.source, len(rank)))


def _rows_from(index: Index, f: FileRef, query: str, *, max_rows: Optional[int] = None) -> List[Row]:
    db = "uniprot" if f.role == "uniprot" else "auto"
    what = f"{f.role} ({f.url.rsplit('/', 2)[-2] if '/' in f.url else f.url})"
    if f.format == "a3m":
        data = index.fetch(f)
        rows = parse_a3m(data, query, db=db, skip_comments=True, what=what)
        return rows[: max_rows] if max_rows else rows
    if f.format == "sto":
        if f.length <= STREAM_THRESHOLD or f.compression != "none":
            data = index.fetch(f)
            return parse_stockholm(data.split(b"\n"), query, db=db, max_rows=max_rows, what=what)
        return parse_stockholm(_big_stockholm_lines(index, f, max_rows or 10_000), query, db=db,
                               max_rows=max_rows, what=what)
    raise A3MError(f"{what}: format {f.format} is not an alignment")


def _read(index: Index, f: FileRef, off: int, n: int) -> bytes:
    n = min(n, f.length - off)
    if n <= 0 or off < 0:
        return b""
    return http.get_bytes(index.resolve(f.url), offset=f.offset + off, length=n)


def _first_line_after(index: Index, f: FileRef, off: int, probe: int = 16384) -> Optional[bytes]:
    chunk = _read(index, f, off, probe)
    nl = chunk.find(b"\n")
    if nl < 0:
        return None
    rest = chunk[nl + 1 :]
    nl2 = rest.find(b"\n")
    return rest[: nl2 if nl2 >= 0 else len(rest)]


def _big_stockholm_lines(index: Index, f: FileRef, max_rows: int) -> Iterator[bytes]:
    """Lines of a big Stockholm file, read with a few Range requests.

    jackhmmer writes all #=GS lines, then the sequences in blocks of columns.
    Inside a block every line has the same length (names are padded), so the
    start of each block can be computed. For each block only the first
    ``max_rows`` lines are read. Anything that breaks these assumptions makes
    the width or query check fail later; it never yields a short alignment.
    """
    # 1. Skip the #=GS block with a binary search.
    lo, hi = 0, f.length
    while hi - lo > 65536:
        mid = (lo + hi) // 2
        line = _first_line_after(index, f, mid)
        if line is not None and line.startswith(b"#=GS"):
            lo = mid
        else:
            hi = mid
    # 2. Find the first sequence line and its length.
    chunk = _read(index, f, lo, 1 << 20)
    pos = 0 if lo == 0 else chunk.find(b"\n") + 1
    if lo and pos == 0:
        raise A3MError(f"{f.url}: no line break where the sequence rows should start")
    start = width = None
    while pos < len(chunk):
        nl = chunk.find(b"\n", pos)
        if nl < 0:
            break
        line = chunk[pos:nl]
        if line.strip() and not line.startswith(b"#"):
            start, width = lo + pos, nl - pos + 1
            break
        pos = nl + 1
    if start is None or width is None:
        raise A3MError(f"{f.url}: could not find the sequence rows")
    probe = chunk[start - lo : start - lo + width * 50]
    for line in probe.split(b"\n")[:-1]:
        if not line.strip() or line.startswith(b"#"):
            break  # end of a block shorter than 50 rows
        if len(line) + 1 != width:
            raise A3MError(f"{f.url}: sequence lines are not fixed width; cannot read it in parts")
    S, L = start, width

    # 3. Count the lines of the first block: line i starts at S + i*L.
    def is_line(i: int) -> bool:
        b = _read(index, f, S + i * L - 1, L + 1)
        return (len(b) == L + 1 and b[:1] == b"\n" and b[-1:] == b"\n" and b"\n" not in b[1:-1]
                and b[1:2] not in (b"#", b"/", b" "))

    hi_i = 1
    while S + hi_i * L < f.length and is_line(hi_i):
        hi_i *= 2
    lo_i = hi_i // 2  # is_line(lo_i) holds (line 0 is the first line)
    while hi_i - lo_i > 1:
        mid = (lo_i + hi_i) // 2
        if is_line(mid):
            lo_i = mid
        else:
            hi_i = mid
    n = lo_i + 1

    # 4. Header, descriptions of the first rows, then each block's first rows.
    yield b"# STOCKHOLM 1.0"
    m = min(max_rows, n)
    head = _read(index, f, 0, min(S, 64 + m * 400))
    k = 0
    for line in head.split(b"\n")[:-1]:
        if line.startswith(b"#=GS"):
            yield line
            k += 1
            if k >= m:
                break
    blocks: List[tuple] = []
    off: Optional[int] = S
    while off is not None and off < f.length:
        first = _first_line_after(index, f, off - 1)
        if not first or first.startswith((b"#", b"//")):
            break
        lk = len(first) + 1
        blocks.append((off, lk))
        off = _next_block(index, f, off + n * lk)
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(lambda b: _read(index, f, b[0], m * b[1]), blocks))
    for data in parts:
        yield b""
        yield from data.split(b"\n")[:m]
    yield b"//"


def _next_block(index: Index, f: FileRef, off: int) -> Optional[int]:
    """Offset of the next block's first sequence line, skipping the '#=GC'
    lines and the blank line that end a block. None at the end of the file."""
    tail = _read(index, f, off, 65536)
    p = 0
    while True:
        nl = tail.find(b"\n", p)
        line = tail[p:] if nl < 0 else tail[p:nl]
        if line.startswith(b"//"):
            return None
        if line.strip() and not line.startswith(b"#"):
            return off + p
        if nl < 0:
            return None
        p = nl + 1


def _merge(blocks: Iterable[List[Row]]) -> List[Row]:
    out: List[Row] = []
    for rows in blocks:
        if not out:
            out = list(rows)
        else:
            out.extend(rows[1:])
    return dedupe_rows(out)


def load_entry(
    index: Index,
    entry: IndexEntry,
    query: str,
    *,
    purpose: str = "a3m",
    want_uniprot: bool = False,
) -> MSA:
    roles = {f.role: f for f in entry.files}
    try:
        if purpose == "af3" and "cfdb_uniref30" in roles:
            blocks = [_rows_from(index, roles["cfdb_uniref30"], query)]
            for role in ("uniref90", "mgnify"):
                if role in roles:
                    blocks.append(_rows_from(index, roles[role], query, max_rows=ROW_CAPS[role]))
            rows = _merge(blocks)
            used = ["cfdb_uniref30", "uniref90", "mgnify"]
        elif "colabfold" in roles:
            rows = _rows_from(index, roles["colabfold"], query)
            used = ["colabfold"]
        elif "cfdb_uniref30" in roles:
            rows = _rows_from(index, roles["cfdb_uniref30"], query)
            used = ["cfdb_uniref30"]
        elif any(r in roles for r in ("uniref90", "bfd_uniclust", "mgnify")):
            used = [r for r in ("uniref90", "bfd_uniclust", "mgnify") if r in roles]
            rows = _merge(_rows_from(index, roles[r], query) for r in used)
        elif "uniclust30" in roles:
            rows = _rows_from(index, roles["uniclust30"], query)
            used = ["uniclust30"]
        else:
            raise EntryUnusable(f"{entry.source} {entry.source_id}: no alignment file")
        used = [u for u in used if u in roles]
    except A3MError as e:
        raise EntryUnusable(f"{entry.source} {entry.source_id}: {e}") from None
    except (IndexUnavailable, IndexIntegrityError, http.HTTPError) as e:
        raise EntryUnusable(f"{entry.source} {entry.source_id}: {e}") from None
    # UniProt rows only help pairing; a problem there must not lose the MSA.
    uniprot_rows = None
    uniprot_error = None
    if want_uniprot and "uniprot" in roles:
        try:
            uniprot_rows = _rows_from(index, roles["uniprot"], query, max_rows=ROW_CAPS["uniprot"])
            used.append("uniprot")
        except (A3MError, IndexUnavailable, IndexIntegrityError, http.HTTPError) as e:
            uniprot_error = str(e)
            log.warning("UniProt rows for %s %s could not be read: %s", entry.source, entry.source_id, e)
    attribution = [dict(index.attribution(entry.source), license=entry.license or index.source(entry.source).get("license", ""))]
    prov = {
        "from": "index",
        "index_url": index.base,
        "build_id": index.build_id,
        "index_source": entry.source,
        "source_id": entry.source_id,
        "roles": used,
        "files": [roles[r].url for r in used],
    }
    if uniprot_error:
        prov["uniprot_error"] = uniprot_error
    return MSA(query=query, rows=rows, source="exact", provenance=prov, attribution=attribution,
               uniprot_rows=uniprot_rows)
