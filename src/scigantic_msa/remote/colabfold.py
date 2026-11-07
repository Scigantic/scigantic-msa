"""Client for the ColabFold MMseqs2 API (https://api.colabfold.com).

Protocol, from the server source (soedinglab/MMseqs2-App, backend/) and a
live check:

- POST /ticket/msa with form fields ``q`` (FASTA) and ``mode``.
- GET /ticket/<id> returns {"id", "status"} with status PENDING, RUNNING,
  COMPLETE, ERROR or UNKNOWN. The rate limiter answers RATELIMIT, usually
  with HTTP 429. Some proxies answer MAINTENANCE.
- GET /result/download/<id> returns a tar.gz with uniref.a3m, the env a3m,
  uniref_tax.tsv (taxonomy mode), pdb70.m8 and msa.sh.
- The ticket id is base64url(sha224(q + mode + sorted databases)), so a job
  can be resumed or downloaded without submitting again.

Privacy: the public server needs no login to read a result. Anyone who knows
the sequence can compute the id and download it. Treat a submission as
publishing the sequence.

Unlike the reference clients, this one checks HTTP status codes, gives up on
rate limits and unknown tickets after a bounded number of tries, has an
overall deadline, and checks the download is a real result before using it.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import tarfile
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .. import http
from ..a3m import A3MError, Row, parse_a3m
from ..seqkey import canonical_sequence

__all__ = ["ColabFoldClient", "RemoteError", "RateLimited", "RemoteResult", "ticket_id", "DEFAULT_HOST"]

DEFAULT_HOST = "https://api.colabfold.com"
ENV_A3M = "bfd.mgnify30.metaeuk30.smag30.a3m"
TERMS = (
    "The public ColabFold MMseqs2 server is a shared resource run by the ColabFold team "
    "(a few thousand MSAs per day for everyone). Results are not private: anyone who "
    "knows the sequence can compute the job id and download the result."
)


class RemoteError(Exception):
    pass


class RateLimited(RemoteError):
    pass


@dataclass
class RemoteResult:
    rows: List[Row]
    provenance: Dict[str, object]
    templates: Optional[Dict[str, str]] = None
    members: Dict[str, int] = field(default_factory=dict)


def ticket_id(q: str, mode: str, databases: Optional[List[str]] = None) -> str:
    h = hashlib.sha224()
    h.update(q.encode())
    h.update(mode.encode())
    for d in sorted(databases or []):
        h.update(d.encode())
    return base64.urlsafe_b64encode(h.digest()).decode().rstrip("=")


def query_text(seq: str) -> str:
    return f">101\n{canonical_sequence(seq)}\n"


class ColabFoldClient:
    name = "colabfold"

    def __init__(
        self,
        host: Optional[str] = None,
        *,
        mode: str = "env-taxonomy",
        contact: Optional[str] = None,
        deadline: float = 70 * 60,
        poll_min: float = 5.0,
        poll_jitter: float = 5.0,
        ratelimit_initial: float = 60.0,
        ratelimit_retries: int = 6,
        unknown_polls: int = 3,
        timeout: float = 30.0,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
    ):
        self.host = (host or os.environ.get("SCIGANTIC_MSA_COLABFOLD_URL") or DEFAULT_HOST).rstrip("/")
        self.mode = mode
        contact = contact or os.environ.get("SCIGANTIC_MSA_CONTACT")
        self.user_agent = http.USER_AGENT + (f" {contact}" if contact else "")
        self.deadline = deadline
        self.poll_min = poll_min
        self.poll_jitter = poll_jitter
        self.ratelimit_initial = ratelimit_initial
        self.ratelimit_retries = ratelimit_retries
        self.unknown_polls = unknown_polls
        self.timeout = timeout
        self.sleep = sleep or (lambda s: time.sleep(s))
        self.clock = clock or (lambda: time.monotonic())
        self.on_status = on_status

    @property
    def server_name(self) -> str:
        return urllib.parse.urlsplit(self.host).netloc or self.host

    @property
    def dbkey(self) -> str:
        return hashlib.sha256(f"{self.host}|{self.mode}".encode()).hexdigest()[:16]

    # ------------------------------------------------------------ low level
    def _call(self, method: str, path: str, data: Optional[Dict[str, str]] = None) -> http.Response:
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        try:
            return http.request(
                method, self.host + path, data=body, headers=headers, timeout=self.timeout,
                retry=http.RetryPolicy(attempts=4, base_delay=5.0, max_delay=60.0),
                retry_statuses=frozenset({500, 502, 503, 504}),
                raise_for_status=False,
            )
        except http.HTTPError as e:
            # The server could not be reached at all (refused, DNS, timeouts).
            raise RemoteError(f"{self.server_name}: {e}") from None

    def _json(self, resp: http.Response, what: str) -> Dict[str, object]:
        if resp.status == 429:
            return {"status": "RATELIMIT", "_retry_after": resp.headers.get("Retry-After")}
        if resp.status >= 400:
            raise RemoteError(f"{what}: HTTP {resp.status} from {self.server_name}: {resp.body[:200]!r}")
        try:
            d = json.loads(resp.body or b"")
        except ValueError:
            raise RemoteError(f"{what}: {self.server_name} returned something that is not JSON: {resp.body[:200]!r}") from None
        if not isinstance(d, dict) or "status" not in d:
            raise RemoteError(f"{what}: unexpected answer from {self.server_name}: {d!r}")
        return d

    def status(self, tid: str) -> Dict[str, object]:
        return self._json(self._call("GET", f"/ticket/{urllib.parse.quote(tid)}"), "status")

    def submit(self, seq: str) -> Dict[str, object]:
        return self._json(self._call("POST", "/ticket/msa", {"q": query_text(seq), "mode": self.mode}), "submit")

    def download(self, tid: str) -> bytes:
        resp = self._call("GET", f"/result/download/{urllib.parse.quote(tid)}")
        if resp.status != 200:
            raise RemoteError(f"download: HTTP {resp.status} from {self.server_name}")
        if not resp.body:
            raise RemoteError("download: empty body")
        return resp.body

    # ------------------------------------------------------------ the job
    def _wait(self, seconds: float, end: float) -> None:
        if self.clock() + seconds > end:
            raise RemoteError(f"gave up after {self.deadline / 60:.0f} minutes waiting for {self.server_name}")
        self.sleep(seconds)

    def _ratelimit_delay(self, n: int, retry_after: object) -> float:
        if n >= self.ratelimit_retries:
            raise RateLimited(
                f"{self.server_name} rate limited this machine {n} times in a row. The public server "
                "allows about one new job per minute per IP address. Try again later, or run your own server."
            )
        try:
            ra = float(str(retry_after)) if retry_after is not None else None
        except ValueError:
            ra = None
        return max(ra or 0.0, self.ratelimit_initial * (2 ** n))

    def search(self, seq: str) -> RemoteResult:
        """Run (or resume) one unpaired MSA job and return its rows.

        The job id is computed locally, so the first call is a status check:
        a job that is already complete is downloaded without submitting.
        """
        seq = canonical_sequence(seq)
        q = query_text(seq)
        tid = ticket_id(q, self.mode)
        end = self.clock() + self.deadline
        started = time.time()
        rate_limited = 0
        maintenance = 0
        unknown = 0
        resubmits = 0
        submitted = False
        last = "status"
        st = self.status(tid)
        while True:
            s = str(st.get("status"))
            if self.on_status:
                self.on_status(s, tid)
            if s == "COMPLETE":
                break
            if s == "ERROR":
                raise RemoteError(f"{self.server_name} reported an error for job {tid}")
            if s == "RATELIMIT":
                self._wait(self._ratelimit_delay(rate_limited, st.get("_retry_after")), end)
                rate_limited += 1
                if last == "submit":
                    st = self.submit(seq)
                    if st.get("status") not in ("RATELIMIT", "MAINTENANCE"):
                        submitted = True
                else:
                    st = self.status(tid)
                continue
            rate_limited = 0
            if s == "MAINTENANCE":
                if maintenance >= self.ratelimit_retries:
                    raise RemoteError(f"{self.server_name} is in maintenance; try again later")
                self._wait(self.ratelimit_initial * (2 ** maintenance), end)
                maintenance += 1
                st = self.submit(seq) if last == "submit" else self.status(tid)
                continue
            maintenance = 0
            if s == "UNKNOWN":
                if not submitted:
                    last = "submit"
                    st = self.submit(seq)
                    if st.get("status") not in ("RATELIMIT", "MAINTENANCE"):
                        submitted = True
                        last = "status"
                        new_id = st.get("id")
                        if isinstance(new_id, str) and new_id and new_id != tid:
                            tid = new_id
                    continue
                unknown += 1
                if unknown >= self.unknown_polls:
                    if resubmits >= 1:
                        raise RemoteError(f"{self.server_name} does not know job {tid}, even after submitting it again")
                    resubmits += 1
                    unknown = 0
                    submitted = False
                    continue
            elif s in ("PENDING", "RUNNING"):
                unknown = 0
            else:
                raise RemoteError(f"{self.server_name} answered with an unknown status {s!r}")
            self._wait(self.poll_min + random.random() * self.poll_jitter, end)
            last = "status"
            st = self.status(tid)
        tar = self.download(tid)
        return self.parse_result(tar, seq, tid, started)

    def parse_result(self, tar_bytes: bytes, seq: str, tid: str, started: Optional[float] = None) -> RemoteResult:
        try:
            tf = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz")
            members = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
        except (tarfile.TarError, OSError, EOFError) as e:
            raise RemoteError(f"download for job {tid} is not a valid tar.gz: {e}") from None

        def read(name: str) -> Optional[bytes]:
            m = members.get(name)
            if m is None:
                return None
            f = tf.extractfile(m)
            return f.read() if f else None

        tokens = set(self.mode.split("-"))
        uniref = read("uniref.a3m")
        if not uniref:
            raise RemoteError(f"result for job {tid} has no uniref.a3m")
        n_env = 0
        try:
            rows = parse_a3m(uniref, seq, what="uniref.a3m")
            n_uniref = len(rows) - 1
            if "env" in tokens:
                env = read(ENV_A3M)
                if not env:
                    raise RemoteError(f"result for job {tid} has no {ENV_A3M}")
                env_rows = parse_a3m(env, seq, what=ENV_A3M)
                for r in env_rows[1:]:
                    r.db = "env" if not r.header.startswith("UniRef") else r.db
                rows = rows + env_rows[1:]
                n_env = len(env_rows) - 1
        except A3MError as e:
            raise RemoteError(f"result for job {tid} failed validation: {e}") from None
        taxmap: Dict[str, str] = {}
        tax = read("uniref_tax.tsv")
        if tax:
            for line in tax.decode("utf-8", "replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].isdigit():
                    taxmap[parts[1]] = parts[0]
        for r in rows[1:]:
            ident = r.header.split("\t", 1)[0].split()[0]
            if ident in taxmap:
                r.taxid = taxmap[ident]
        msa_sh = read("msa.sh")
        m8 = read("pdb70.m8")
        prov: Dict[str, object] = {
            "from": "remote",
            "backend": "colabfold",
            "server": self.host,
            "mode": self.mode,
            "ticket": tid,
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "uniref_rows": n_uniref,
            "env_rows": n_env,
            "rows_with_taxid": sum(1 for r in rows if r.taxid),
        }
        if msa_sh:
            prov["msa_sh_sha256"] = hashlib.sha256(msa_sh).hexdigest()
        if started is not None:
            prov["seconds"] = round(time.time() - started, 1)
        return RemoteResult(
            rows=rows,
            provenance=prov,
            templates={"pdb70_m8": m8.decode("utf-8", "replace")} if m8 else None,
            members={k: v.size for k, v in members.items()},
        )
