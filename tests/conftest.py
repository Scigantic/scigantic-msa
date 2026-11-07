"""Shared test harness: an HTTP server with fault injection, an index builder
that writes the exact v1 format, and the real fixture files."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REAL = FIXTURES / "real"

Q_6T33 = "WGCVCSGSTAVANSHNAGPAYCVGYCGNNGVVTRNANANVAKTA"
Q_1HVW = "CIPSGQPCPYNENCCSQSCTGGRCD"
Q_1PXQ = "NKGCATCSIGAACLVDGPIPDFEIAGATGLFGLWG"
Q_LITEFOLD = ("MYRNEFTAWYRRMSVVYGIGTWSVLGSLLYYSRTMAKSSVDQKDGSASEVPSELSERPKGFYVETVVTYKEDFVPNTEKILNYWKSWTGGPGTEP")
Q_LIVE = "SGSMKTAISLPDETFDRVSRRASELGMSRSEFFTKAAQR"
Q_7M22A = ("EECCEFINVNHPPERCYDFKMCNRFTVALRCPDGEVCYSPEKTAEIRGIVTTMTHSLTRQVVHNKLTSCNYNPLYLEADGRIRCGKVNDKAQYLL"
           "GAAGSVPYRWINLEYDKITRIVGLDQYLESVKKHKRLDVCRAKMGYMLQ")
Q_7M22C = ("QCQRETAEKNDYYRVPHYWDACSRALPDQTRYKYVEQLVDLTLNYHYDASHGLDNFDVLKRINVTEVSLLISDFRRQNRRGGTNKRTTFNAAGSLA"
           "PHARSLEFSVRLFAN")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def key(seq: str) -> str:
    from scigantic_msa.seqkey import seq_sha256

    return seq_sha256(seq)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """No test touches the real cache, config, token or a real server."""
    for var in list(os.environ):
        if var.startswith(("SCIGANTIC_", "NVIDIA_", "HF_", "HUGGING_FACE")):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SCIGANTIC_MSA_CONFIG", str(tmp_path / "config" / "msa.json"))
    monkeypatch.setenv("SCIGANTIC_MSA_INDEX_URL", "http://127.0.0.1:9/unused/")
    monkeypatch.setenv("SCIGANTIC_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", "http://127.0.0.1:9")
    from scigantic_msa import http

    sleeps: List[float] = []
    monkeypatch.setattr(http, "_sleep", lambda s: sleeps.append(s))
    from scigantic_msa import remote

    remote._announced.clear()
    yield


# --------------------------------------------------------------------------- fake HTTP server


class FakeServer:
    """Serves static bytes and handler functions; records every request.

    faults[path] is a list of one-shot actions, consumed in order:
      ("status", code, headers_dict, body)   answer with this instead
      ("truncate", nbytes)                   send the real response cut short
      ("ignore_range",)                      answer 200 with the whole body
      ("sleep", seconds)                     wait before answering
    """

    def __init__(self) -> None:
        self.static: Dict[str, bytes] = {}
        self.handlers: Dict[str, Callable[["Req"], tuple]] = {}
        self.prefix_handlers: List[tuple] = []
        self.faults: Dict[str, List[tuple]] = defaultdict(list)
        self.requests: List["Req"] = []
        self.lock = threading.Lock()
        server = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def _go(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                req = Req(self.command, self.path, dict(self.headers.items()), body)
                with server.lock:
                    server.requests.append(req)
                    faults = server.faults.get(req.path_only)
                    fault = faults.pop(0) if faults else None
                if fault and fault[0] == "sleep":
                    time.sleep(fault[1])
                    fault = None
                if fault and fault[0] == "status":
                    _, code, hdrs, fbody = fault
                    return self._send(code, hdrs or {}, fbody or b"")
                status, hdrs, data = server.dispatch(req)
                if status == 200 and req.method in ("GET", "HEAD") and "Range" in req.headers and not (fault and fault[0] == "ignore_range"):
                    status, hdrs, data = _ranged(req.headers["Range"], data, hdrs)
                if fault and fault[0] == "truncate":
                    hdrs = dict(hdrs)
                    hdrs["Content-Length"] = str(len(data))
                    self.send_response(status)
                    for k, v in hdrs.items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(data[: fault[1]])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self._send(status, hdrs, data)

            def _send(self, status: int, hdrs: Dict[str, str], data: bytes) -> None:
                self.send_response(status)
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _go

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def dispatch(self, req: "Req") -> tuple:
        h = self.handlers.get(f"{req.method} {req.path_only}")
        if h:
            return h(req)
        for prefix, method, fn in self.prefix_handlers:
            if req.method == method and req.path_only.startswith(prefix):
                return fn(req)
        if req.method in ("GET", "HEAD") and req.path_only in self.static:
            return 200, {"Content-Type": "application/octet-stream"}, self.static[req.path_only]
        return 404, {"Content-Type": "text/plain"}, b"not found"

    def route(self, method: str, path: str, fn: Callable[["Req"], tuple]) -> None:
        self.handlers[f"{method} {path}"] = fn

    def route_prefix(self, method: str, prefix: str, fn: Callable[["Req"], tuple]) -> None:
        self.prefix_handlers.append((prefix, method, fn))

    def count(self, pred: Callable[["Req"], bool]) -> int:
        with self.lock:
            return sum(1 for r in self.requests if pred(r))

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class Req:
    def __init__(self, method: str, path: str, headers: Dict[str, str], body: bytes):
        self.method = method
        self.path = path
        self.path_only = path.split("?", 1)[0]
        self.query = path.split("?", 1)[1] if "?" in path else ""
        self.headers = headers
        self.body = body

    def json(self) -> Any:
        return json.loads(self.body or b"{}")


def _ranged(rng: str, data: bytes, hdrs: Dict[str, str]) -> tuple:
    spec = rng.split("=", 1)[1]
    a, b = spec.split("-", 1)
    start = int(a)
    end = int(b) if b else len(data) - 1
    end = min(end, len(data) - 1)
    if start >= len(data):
        return 416, {}, b""
    h = dict(hdrs)
    h["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
    return 206, h, data[start : end + 1]


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


# --------------------------------------------------------------------------- index builder


def real(rel: str) -> bytes:
    return (REAL / rel).read_bytes()


def serve_real_files(srv: FakeServer) -> Dict[str, str]:
    """Put the real fixture files on the server under source-like paths.
    Returns rel -> url."""
    urls = {}
    for p in sorted(REAL.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            rel = str(p.relative_to(REAL))
            path = "/src/" + rel
            srv.static[path] = p.read_bytes()
            urls[rel] = srv.url + path
    return urls


def fref(urls: Dict[str, str], rel: str, role: str, fmt: str) -> Dict[str, Any]:
    return {"role": role, "format": fmt, "url": urls[rel], "offset": 0, "length": len(real(rel)), "compression": "none"}


def standard_entries(urls: Dict[str, str]) -> List[Dict[str, Any]]:
    return [
        {"seq": Q_6T33, "source": "openfold3-pdb", "source_id": "6t33_A", "license": "CC-BY-4.0", "files": [
            fref(urls, "of3-6t33_A/cfdb_uniref30.a3m", "cfdb_uniref30", "a3m"),
            fref(urls, "of3-6t33_A/uniref90_hits.sto", "uniref90", "sto"),
            fref(urls, "of3-6t33_A/mgnify_hits.sto", "mgnify", "sto"),
            fref(urls, "of3-6t33_A/uniprot_hits.sto", "uniprot", "sto"),
        ]},
        {"seq": Q_6T33, "source": "openproteinset-pdb", "source_id": "6t33_A", "n_aliases": 1, "license": "CC-BY-4.0", "files": [
            fref(urls, "ops-6t33_A/uniref90_hits.a3m", "uniref90", "a3m"),
            fref(urls, "ops-6t33_A/bfd_uniclust_hits.a3m", "bfd_uniclust", "a3m"),
            fref(urls, "ops-6t33_A/mgnify_hits.a3m", "mgnify", "a3m"),
            fref(urls, "ops-6t33_A/pdb70_hits.hhr", "templates_pdb70", "hhr"),
        ]},
        {"seq": Q_1HVW, "source": "openfold3-pdb", "source_id": "1hvw_A", "license": "CC-BY-4.0", "files": [
            fref(urls, "of3-1hvw_A/cfdb_uniref30.a3m", "cfdb_uniref30", "a3m"),
            fref(urls, "of3-1hvw_A/uniprot_hits.sto", "uniprot", "sto"),
        ]},
        {"seq": Q_1PXQ, "source": "openfold3-pdb", "source_id": "1pxq_A", "license": "CC-BY-4.0", "files": [
            fref(urls, "of3-1pxq_A/cfdb_uniref30.a3m", "cfdb_uniref30", "a3m"),
            fref(urls, "of3-1pxq_A/uniprot_hits.sto", "uniprot", "sto"),
        ]},
        {"seq": Q_7M22A, "source": "openfold3-pdb", "source_id": "7m22_A", "license": "CC-BY-4.0", "files": [
            fref(urls, "of3-7m22_A/cfdb_uniref30.a3m", "cfdb_uniref30", "a3m"),
            fref(urls, "of3-7m22_A/uniprot_hits.sto", "uniprot", "sto"),
        ]},
        {"seq": Q_7M22C, "source": "openfold3-pdb", "source_id": "7m22_C", "license": "CC-BY-4.0", "files": [
            fref(urls, "of3-7m22_C/cfdb_uniref30.a3m", "cfdb_uniref30", "a3m"),
            fref(urls, "of3-7m22_C/uniprot_hits.sto", "uniprot", "sto"),
        ]},
        {"seq": Q_LITEFOLD, "source": "litefold-human-colabfold", "source_id": "A0A096LP01", "license": "CC-BY-4.0", "files": [
            fref(urls, "litefold-A0A096LP01/A0A096LP01.a3m", "colabfold", "a3m"),
            fref(urls, "litefold-A0A096LP01/A0A096LP01.json", "af3_json", "json"),
        ]},
        {"seq": uniclust_query(), "source": "openproteinset-uniclust30", "source_id": "A0A1A9ZI35", "license": "CC-BY-4.0", "files": [
            fref(urls, "uniclust30-A0A1A9ZI35/uniclust30.a3m", "uniclust30", "a3m"),
        ]},
    ]


def uniclust_query() -> str:
    from scigantic_msa.a3m import a3m_query

    q = a3m_query(real("uniclust30-A0A1A9ZI35/uniclust30.a3m").decode(), "A0A1A9ZI35")
    assert q
    return q


SOURCES = [
    {"id": "openfold3-pdb", "title": "OpenFold3 training MSAs (PDB chains)", "license": "CC-BY-4.0",
     "attribution": "OpenFold Consortium", "citation": "https://github.com/aqlaboratory/openfold-3",
     "url_prefixes": {"https://openfold3-data.s3.us-west-2.amazonaws.com/": "s3://openfold3-data/"}},
    {"id": "litefold-human-colabfold", "title": "LiteFold human proteome ColabFold MSAs", "license": "CC-BY-4.0",
     "attribution": "LiteFold", "citation": "https://huggingface.co/datasets/LiteFold/human-proteome-wide-msa"},
    {"id": "openproteinset-pdb", "title": "OpenProteinSet PDB chains", "license": "CC-BY-4.0",
     "attribution": "Ahdritz et al.", "citation": "https://arxiv.org/abs/2308.05326",
     "url_prefixes": {"https://openfold.s3.amazonaws.com/": "s3://openfold/"}},
    {"id": "openproteinset-uniclust30", "title": "OpenProteinSet UniClust30", "license": "CC-BY-4.0",
     "attribution": "Ahdritz et al.", "citation": "https://arxiv.org/abs/2308.05326"},
]
PRIORITY = ["openfold3-pdb", "litefold-human-colabfold", "openproteinset-pdb", "openproteinset-uniclust30"]


def build_index(root: Path, entries: List[Dict[str, Any]], build_id: str = "test-build-1",
                extra_sequences: Optional[Dict[str, str]] = None) -> Path:
    """Write a v1 index under root (root/v1/...). Entries carry 'seq', which
    is used for the by_length and sequences files and dropped from lines."""
    broot = root / "v1" / "builds" / build_id
    files: Dict[str, bytes] = {}
    shards: Dict[str, List[str]] = defaultdict(list)
    seqs: Dict[str, str] = dict(extra_sequences or {})
    prio = {s: i for i, s in enumerate(PRIORITY)}
    lines = []
    for e in entries:
        d = {k: v for k, v in e.items() if k != "seq"}
        sha = key(e["seq"])
        d["seq_sha256"] = sha
        d["length"] = len(e["seq"])
        seqs[sha] = e["seq"]
        lines.append((sha, prio.get(d["source"], 99), json.dumps(d, separators=(",", ":"))))
    lines.sort()
    for sha, _, line in lines:
        shards[sha[:3]].append(line)
    for i in range(4096):
        p = f"{i:03x}"
        body = [json.dumps({"_shard": p, "build_id": build_id, "entries": len(shards.get(p, []))})] + shards.get(p, [])
        files[f"shards/{p}.jsonl.gz"] = gzip.compress(("\n".join(body) + "\n").encode(), mtime=0)
    by_len: Dict[int, List[tuple]] = defaultdict(list)
    for sha, s in seqs.items():
        by_len[len(s)].append((sha, s))
    bl_index = {}
    for L, items in sorted(by_len.items()):
        data = gzip.compress("".join(f">{sha}\n{s}\n" for sha, s in sorted(items)).encode(), mtime=0)
        files[f"by_length/{L}.fa.gz"] = data
        bl_index[str(L)] = {"n": len(items), "sha256": _sha(data)}
    files["by_length/index.json"] = json.dumps(bl_index).encode()
    files["sequences.fasta.gz"] = gzip.compress("".join(f">{sha}\n{s}\n" for sha, s in sorted(seqs.items())).encode(), mtime=0)
    sums = "".join(f"{_sha(b)}  {rel}\n" for rel, b in sorted(files.items())).encode()
    for rel, b in files.items():
        (broot / rel).parent.mkdir(parents=True, exist_ok=True)
        (broot / rel).write_bytes(b)
    (broot / "SHA256SUMS").write_bytes(sums)
    manifest = {
        "format": "scigantic-msa-index", "format_version": 1, "build_id": build_id,
        "built_at": "2026-11-06T00:00:00Z", "build_root": f"v1/builds/{build_id}/",
        "shard_hex_len": 3, "shard_path": "shards/{prefix}.jsonl.gz",
        "sha256sums": "SHA256SUMS", "sha256sums_sha256": _sha(sums),
        "counts": {"entries": len(entries), "unique_sequences": len(seqs)},
        "source_priority": PRIORITY, "sources": SOURCES,
        "near": {"by_length": "by_length/{length}.fa.gz", "by_length_index": "by_length/index.json",
                 "sequences_fasta": "sequences.fasta.gz"},
    }
    mb = json.dumps(manifest, indent=1).encode()
    (broot / "manifest.json").write_bytes(mb)
    (root / "v1" / "manifest.json").write_bytes(mb)
    return root


def serve_dir(srv: FakeServer, root: Path, mount: str = "/idx") -> str:
    for p in root.rglob("*"):
        if p.is_file():
            srv.static[mount + "/" + str(p.relative_to(root))] = p.read_bytes()
    return srv.url + mount + "/"


@pytest.fixture
def index_setup(server, tmp_path):
    """A served index over the real fixture files. Returns (Index, base_url, root)."""
    from scigantic_msa.index import Index

    urls = serve_real_files(server)
    root = build_index(tmp_path / "index", standard_entries(urls))
    base = serve_dir(server, root)
    return Index(base, cache_root=tmp_path / "cache"), base, root
