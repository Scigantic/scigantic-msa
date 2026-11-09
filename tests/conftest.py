"""Shared test harness: an HTTP server with fault injection and the real
fixture files."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List

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
    """No test touches the real cache or a real server."""
    for var in list(os.environ):
        if var.startswith(("SCIGANTIC_", "NVIDIA_", "HF_", "HUGGING_FACE")):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SCIGANTIC_MSA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", "http://127.0.0.1:9")
    from scigantic_msa import http

    sleeps: List[float] = []
    monkeypatch.setattr(http, "_sleep", lambda s: sleeps.append(s))
    from scigantic_msa import remote

    remote._announced.clear()
    from scigantic_msa import tool

    monkeypatch.setattr(tool, "_cache_warned", False)
    # Without the cache, open data goes to one temporary folder per process;
    # each test gets its own.
    monkeypatch.setattr(tool, "_temp", {})
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


# --------------------------------------------------------------------------- real alignments


def real(rel: str) -> bytes:
    return (REAL / rel).read_bytes()


_DIRS = {
    Q_6T33: "of3-6t33_A", Q_1HVW: "of3-1hvw_A", Q_1PXQ: "of3-1pxq_A",
    Q_7M22A: "of3-7m22_A", Q_7M22C: "of3-7m22_C", Q_LITEFOLD: "litefold-A0A096LP01",
}


def real_alignment(seq: str, *, with_uniprot: bool = False):
    """An Alignment built from the real published files in fixtures/real:
    OpenFold3 training MSAs (cfdb_uniref30.a3m, plus uniprot_hits.sto for
    pairing) and a LiteFold ColabFold MSA."""
    from scigantic_msa.a3m import parse_a3m, parse_stockholm
    from scigantic_msa.alignment import Alignment

    d = REAL / _DIRS[seq]
    a3m = d / ("A0A096LP01.a3m" if seq == Q_LITEFOLD else "cfdb_uniref30.a3m")
    rows = parse_a3m(a3m.read_bytes(), seq, skip_comments=True)
    uni = None
    if with_uniprot and (d / "uniprot_hits.sto").is_file():
        uni = parse_stockholm((d / "uniprot_hits.sto").read_bytes().split(b"\n"), seq, db="uniprot", max_rows=10_000)
    return Alignment(query=seq, rows=rows, source="file", provenance={"from": "file", "file": a3m.name},
                     attribution=[{"id": _DIRS[seq], "license": "CC-BY-4.0"}], uniprot_rows=uni)


# --------------------------------------------------------------------------- a fake ColabFold server for any sequence


def make_result_tar(query: str, hits: List[tuple]) -> bytes:
    """A ColabFold-style result tar.gz. hits: (accession, row, taxid or None).
    Rows go to uniref.a3m; taxids to uniref_tax.tsv; the env file repeats the
    query only."""
    import io
    import tarfile

    uniref = f">101\n{query}\n" + "".join(f">{acc}\t50\t0.9\n{row}\n" for acc, row, _ in hits)
    env = f">101\n{query}\n"
    tax = "".join(f"{t}\t{acc}\n" for acc, _, t in hits if t)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in (("uniref.a3m", uniref), ("bfd.mgnify30.metaeuk30.smag30.a3m", env),
                           ("uniref_tax.tsv", tax), ("msa.sh", "#!/bin/sh\n")):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class ColabFoldStub:
    """Answers the ColabFold API for a set of sequences: a job is UNKNOWN
    until submitted, then COMPLETE. Records submissions."""

    def __init__(self, server: FakeServer, tars: Dict[str, bytes], mode: str = "env-taxonomy"):
        import urllib.parse

        from scigantic_msa.remote.colabfold import query_text, ticket_id

        self.by_id = {ticket_id(query_text(q), mode): t for q, t in tars.items()}
        self.submitted: List[str] = []
        self.submits = 0

        def submit(req):
            form = urllib.parse.parse_qs(req.body.decode())
            tid = ticket_id(form["q"][0], form["mode"][0])
            self.submits += 1
            self.submitted.append(tid)
            st = "COMPLETE" if tid in self.by_id else "ERROR"
            return 200, {}, json.dumps({"id": tid, "status": st}).encode()

        def status(req):
            tid = req.path_only.rsplit("/", 1)[1]
            st = "COMPLETE" if tid in self.submitted and tid in self.by_id else "UNKNOWN"
            return 200, {}, json.dumps({"id": tid, "status": st}).encode()

        def download(req):
            tid = req.path_only.rsplit("/", 1)[1]
            return (200, {}, self.by_id[tid]) if tid in self.by_id else (404, {}, b"")

        server.route("POST", "/ticket/msa", submit)
        server.route_prefix("GET", "/ticket/", status)
        server.route_prefix("GET", "/result/download/", download)


@pytest.fixture
def colabfold(server, monkeypatch):
    """Point the ColabFold client at the fake server and make waits instant."""
    monkeypatch.setenv("SCIGANTIC_MSA_COLABFOLD_URL", server.url)
    from scigantic_msa.remote import colabfold as cf

    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    return server


@pytest.fixture(autouse=True)
def no_open_hosts(monkeypatch):
    """Open data hosts point nowhere unless a test serves them."""
    from scigantic_msa.opendata import sources

    for k in list(sources.BASES):
        monkeypatch.setitem(sources.BASES, k, "http://127.0.0.1:9")
    from scigantic_msa.opendata import net

    monkeypatch.setattr(net, "UNIPROT_SPACING", 0.0)
    from scigantic_msa.opendata import tables

    # The fixture tables hold a few entries; the real ones tens of thousands.
    for k in list(tables.MIN_ENTRIES):
        monkeypatch.setitem(tables.MIN_ENTRIES, k, 1)


# --------------------------------------------------------------------------- open data hosts

import io  # noqa: E402
import tarfile  # noqa: E402
import urllib.parse  # noqa: E402

from scigantic_msa.opendata import sources  # noqa: E402
from scigantic_msa.remote.colabfold import ticket_id  # noqa: E402


def md5(q):
    return hashlib.md5(q.encode()).hexdigest().upper()


COMMIT = sources.LITEFOLD_COMMIT
LF_PATH = f"/datasets/{sources.LITEFOLD_REPO}/resolve/{COMMIT}"


class OpenHosts:
    """Every open data host on one fake server, with ETags, and UniParc
    answers per sequence MD5."""

    def __init__(self, server):
        self.s = server
        self.etag = {"fasta": "e1", "manifest": "m1"}
        self.uniparc = {}
        self.uniclust = set()
        reps = "".join(f">{n}\n{q}\n" for n, q in (("6t33_A", Q_6T33), ("7m22_A", Q_7M22A), ("7m22_C", Q_7M22C),
                                                     ("1abc_A", "ACGUACGUAC")))
        self.fasta = reps.encode()
        lf = REAL / "litefold-A0A096LP01" / "A0A096LP01.a3m"
        self.manifest = ("acc\tentry\tgene\tlength\tsequence\ta3m_path\taf3_json_path\ta3m_bytes\taf3_json_bytes\n"
                         f"A0A096LP01\tX_HUMAN\tX\t{len(Q_LITEFOLD)}\t{Q_LITEFOLD}\ta3m/A0A096LP01.a3m\t\t"
                         f"{lf.stat().st_size}\t0\n").encode()
        server.route("GET", "/msas/pdb_msas_completed_representives.fasta",
                     lambda r: (200, {"ETag": f'"{self.etag["fasta"]}"'}, self.fasta))
        server.route("HEAD", "/msas/pdb_msas_completed_representives.fasta",
                     lambda r: (200, {"ETag": f'"{self.etag["fasta"]}"'}, self.fasta))
        server.route("GET", f"{LF_PATH}/manifest.tsv", lambda r: (200, {"ETag": f'"{self.etag["manifest"]}"'}, self.manifest))
        server.route("HEAD", f"{LF_PATH}/manifest.tsv", lambda r: (200, {"ETag": f'"{self.etag["manifest"]}"'}, self.manifest))
        server.static[f"{LF_PATH}/a3m/A0A096LP01.a3m"] = lf.read_bytes()
        for d, name in (("of3-6t33_A", "6t33_A"), ("of3-7m22_A", "7m22_A"), ("of3-7m22_C", "7m22_C")):
            for f in (REAL / d).iterdir():
                server.static[f"/msas/pdb_msas_completed/{name}/{f.name}"] = f.read_bytes()
        server.route("GET", "/uniparc/search", self._uniparc)
        server.route_prefix("HEAD", "/uniclust30/", lambda r: (200, {}, b"x") if r.path_only.split("/")[2] in self.uniclust
                            else (404, {}, b""))
        for k in sources.BASES:
            sources.BASES[k] = server.url

    def _uniparc(self, req):
        qs = urllib.parse.parse_qs(req.query)
        m = qs["query"][0].split(":", 1)[1]
        res = self.uniparc.get(m)
        body = {"results": [] if res is None else [{"uniParcId": "UPI0000000001", "uniProtKBAccessions": res[0],
                                                     "organisms": [{"scientificName": n, "taxonId": t} for n, t in res[1]]}]}
        return 200, {"X-UniProt-Release": "2026_04"}, json.dumps(body).encode()

    def get_paths(self):
        return [r.path for r in self.s.requests if r.method == "GET"]


@pytest.fixture
def hosts(server, monkeypatch):
    for k in list(sources.BASES):
        monkeypatch.setitem(sources.BASES, k, server.url)
    return OpenHosts(server)


def _pair_tar(qs, rows):
    text = "\x00".join(f">{101 + c}\n{q}\n" + "".join(f">{h}\n{s}\n" for h, s in rows[c]) for c, q in enumerate(qs))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("pair.a3m")
        info.size = len(text.encode())
        tf.addfile(info, io.BytesIO(text.encode()))
    return buf.getvalue()


@pytest.fixture
def complex_server(colabfold, hosts):
    a_row, c_row = Q_7M22A[:-1] + ("A" if Q_7M22A[-1] != "A" else "G"), Q_7M22C[:-1] + ("A" if Q_7M22C[-1] != "A" else "G")
    stub = ColabFoldStub(colabfold, {Q_7M22A: make_result_tar(Q_7M22A, [("UniRef100_S1", a_row, "9606")]),
                                     Q_7M22C: make_result_tar(Q_7M22C, [("UniRef100_S2", c_row, "9606")])})
    q = "".join(f">{101 + i}\n{s}\n" for i, s in enumerate((Q_7M22A, Q_7M22C)))
    pid = ticket_id(q, "pairgreedy")
    stub.by_id[pid] = _pair_tar([Q_7M22A, Q_7M22C], [[("UniRef100_S1", a_row)], [("UniRef100_S2", c_row)]])

    def submit_pair(req):
        stub.submitted.append(pid)
        return 200, {}, json.dumps({"id": pid, "status": "COMPLETE"}).encode()

    colabfold.route("POST", "/ticket/pair", submit_pair)
    return stub


