"""Experimental client for NVIDIA's hosted MSA Search (ColabFold MMseqs2 on
NVIDIA's cloud). Needs an API key in NVIDIA_API_KEY.

Request and response shapes are from NVIDIA's MSA Search NIM 3.0.0 API
reference. Rate limits, credit cost and whether long jobs answer 202 with an
NVCF-REQID are not documented for the hosted endpoint, so 202 is handled by
polling the NVCF status URL, and anything else unexpected is an error.
Results are covered by NVIDIA's API trial terms of service.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

from .. import http
from ..a3m import A3MError, parse_a3m
from ..seqkey import canonical_sequence
from .colabfold import JobFailed, RemoteError, RemoteResult

__all__ = ["NvidiaClient", "DEFAULT_URL"]

DEFAULT_URL = "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict"
STATUS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{reqid}"


class NvidiaClient:
    name = "nvidia"
    experimental = True

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        databases: Optional[List[str]] = None,
        e_value: float = 0.0001,
        deadline: float = 30 * 60,
        poll: float = 5.0,
        timeout: float = 120.0,
        sleep: Optional[Callable[[float], None]] = None,
        status_url: str = STATUS_URL,
    ):
        self.url = url or os.environ.get("SCIGANTIC_MSA_NVIDIA_URL") or DEFAULT_URL
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.databases = databases or ["all"]
        self.e_value = e_value
        self.deadline = deadline
        self.poll = poll
        self.timeout = timeout
        self.sleep = sleep or (lambda s: time.sleep(s))
        self.status_url = status_url

    @property
    def server_name(self) -> str:
        return http.split_url(self.url)[0]

    @property
    def dbkey(self) -> str:
        return "-".join(sorted(self.databases))

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RemoteError("the NVIDIA backend needs an API key: set NVIDIA_API_KEY")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, url: str, data: Optional[bytes] = None) -> http.Response:
        try:
            return http.request(method, url, data=data, headers=self._headers(), timeout=self.timeout,
                                raise_for_status=False)
        except http.HTTPError as e:
            # The server could not be reached at all (refused, DNS, timeouts).
            raise RemoteError(f"NVIDIA MSA search: {e}") from None

    def search(self, seq: str) -> RemoteResult:
        seq = canonical_sequence(seq)
        body = json.dumps({
            "sequence": seq,
            "databases": self.databases,
            "e_value": self.e_value,
            "output_alignment_formats": ["a3m"],
        }).encode()
        started = time.time()
        resp = self._request("POST", self.url, data=body)
        end = time.monotonic() + self.deadline
        while resp.status == 202:
            reqid = resp.headers.get("NVCF-REQID") or resp.headers.get("nvcf-reqid")
            if not reqid:
                raise RemoteError("NVIDIA answered 202 without an NVCF-REQID header")
            if time.monotonic() > end:
                raise RemoteError("gave up waiting for the NVIDIA MSA search")
            self.sleep(self.poll)
            resp = self._request("GET", self.status_url.format(reqid=reqid))
        if resp.status == 401 or resp.status == 403:
            raise RemoteError(f"NVIDIA rejected the API key (HTTP {resp.status})")
        if resp.status != 200:
            raise RemoteError(f"NVIDIA MSA search: HTTP {resp.status}: {resp.body[:200]!r}")
        try:
            d = json.loads(resp.body)
            al = d["alignments"]
            if "colabfold" in al:
                a3m = al["colabfold"]["a3m"]["alignment"]
                used = "colabfold"
            elif len(al) == 1:
                used = next(iter(al))
                a3m = al[used]["a3m"]["alignment"]
            else:
                raise KeyError("several databases and no merged 'colabfold' alignment")
        except (ValueError, KeyError, TypeError) as e:
            raise RemoteError(f"NVIDIA MSA search: unexpected response ({e})") from None
        try:
            rows = parse_a3m(a3m, seq, what="nvidia a3m")
        except A3MError as e:
            raise JobFailed(f"NVIDIA MSA search result failed validation: {e}") from None
        return RemoteResult(
            rows=rows,
            provenance={
                "from": "remote",
                "backend": "nvidia",
                "server": self.url,
                "databases": self.databases,
                "alignment_key": used,
                "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.time() - started, 1),
                "experimental": True,
            },
        )
