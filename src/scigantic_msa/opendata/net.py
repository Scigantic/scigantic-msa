"""Anonymous reads from the open data hosts.

Every request checks SCIGANTIC_MSA_NO_REMOTE first, like the search
clients, so the switch blocks open data lookups too. Requests are made one
at a time from a process (a lock), and UniProt requests are spaced out, to
be polite to hosts that serve everyone for free. No credentials are ever
sent.
"""
from __future__ import annotations

import http.client
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import IO, Dict, Iterator, Mapping, Optional, Tuple

from .. import http as _http

__all__ = ["OpenDataError", "NotFound", "get", "head", "stream", "Hit"]


class OpenDataError(RuntimeError):
    """An open data host could not be read."""


class NotFound(OpenDataError):
    pass


#: One request at a time per process.
_lock = threading.RLock()
#: Seconds between two requests to rest.uniprot.org.
UNIPROT_SPACING = 0.34
_last_uniprot = [0.0]
#: Attempts for one request (network errors, 429, 5xx).
ATTEMPTS = 5


def _headers(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    h = {"User-Agent": _http.USER_AGENT}
    if extra:
        h.update(extra)
    return h


def _space(url: str) -> None:
    if "rest.uniprot.org" in url:
        wait = _last_uniprot[0] + UNIPROT_SPACING - time.monotonic()
        if wait > 0:
            _http._sleep(wait)
        _last_uniprot[0] = time.monotonic()


@contextmanager
def stream(url: str, *, headers: Optional[Mapping[str, str]] = None,
           timeout: float = 60.0) -> Iterator[Tuple[IO[bytes], Dict[str, str]]]:
    """Open ``url`` for reading. Yields (response, headers). Retries the
    opening on network errors, 429 and 5xx. Raises NotFound for 403 and
    404 (S3 answers 403 for a missing key in a bucket that cannot be
    listed), OpenDataError otherwise."""
    with _lock:
        delay = 2.0
        resp = None
        for attempt in range(ATTEMPTS):
            _http.refuse_if_disabled()
            _space(url)
            req = urllib.request.Request(url, headers=_headers(headers))
            try:
                resp = _http._opener.open(req, timeout=timeout)
                break
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    raise NotFound(f"HTTP {e.code} from {_http._short(url)}") from None
                if e.code not in _http.RETRY_STATUSES or attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"HTTP {e.code} from {_http._short(url)}") from None
                ra = _http._retry_after_seconds(e.headers.get("Retry-After") if e.headers else None)
                wait = min(ra if ra is not None else delay, 120.0)
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError,
                    ConnectionError, OSError) as e:
                if attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"could not read {_http._short(url)}: {getattr(e, 'reason', e)}") from None
                wait = delay
            _http._sleep(wait)
            delay = min(delay * 2, 60.0)
        assert resp is not None
        try:
            yield resp, {k.lower(): v for k, v in resp.headers.items()}
        finally:
            resp.close()


def get(url: str, *, headers: Optional[Mapping[str, str]] = None, timeout: float = 60.0,
        max_bytes: Optional[int] = None) -> Tuple[bytes, Dict[str, str]]:
    """The whole body of ``url`` and its headers. A body cut short by the
    network is read again from the start, up to ATTEMPTS times."""
    for attempt in range(ATTEMPTS):
        try:
            with stream(url, headers=headers, timeout=timeout) as (r, h):
                body = r.read() if max_bytes is None else r.read(max_bytes + 1)
                if max_bytes is not None and len(body) > max_bytes:
                    raise OpenDataError(f"{_http._short(url)} is larger than {max_bytes} bytes")
                want = h.get("content-length")
                if want and want.isdigit() and int(want) != len(body) and max_bytes is None:
                    raise ConnectionError(f"got {len(body)} of {want} bytes")
                return body, h
        except (ConnectionError, http.client.IncompleteRead, socket.timeout, TimeoutError) as e:
            if attempt == ATTEMPTS - 1:
                raise OpenDataError(f"could not read {_http._short(url)}: {e}") from None
            _http._sleep(2.0 * (attempt + 1))
    raise OpenDataError(url)  # pragma: no cover


class Hit:
    """The answer to a HEAD request."""

    def __init__(self, status: int, headers: Dict[str, str]):
        self.status = status
        self.headers = headers

    @property
    def etag(self) -> Optional[str]:
        return etag_of(self.headers)


def etag_of(headers: Mapping[str, str]) -> Optional[str]:
    """The ETag of a response, without quotes. Hugging Face names the
    content's own ETag in X-Linked-Etag on a redirect."""
    for k in ("x-linked-etag", "etag"):
        v = headers.get(k)
        if v:
            return v.strip().strip('"').removeprefix("W/").strip('"')
    return None


def head(url: str, timeout: float = 30.0) -> Hit:
    """HEAD ``url``, following redirects. 403 and 404 come back as a Hit
    with that status, not an error."""
    with _lock:
        delay = 2.0
        for attempt in range(ATTEMPTS):
            _http.refuse_if_disabled()
            _space(url)
            req = urllib.request.Request(url, method="HEAD", headers=_headers())
            try:
                with _http._opener.open(req, timeout=timeout) as r:
                    return Hit(r.status, {k.lower(): v for k, v in r.headers.items()})
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    return Hit(e.code, {})
                if e.code not in _http.RETRY_STATUSES or attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"HTTP {e.code} from {_http._short(url)}") from None
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError,
                    ConnectionError, OSError) as e:
                if attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"could not reach {_http._short(url)}: {getattr(e, 'reason', e)}") from None
            _http._sleep(delay)
            delay = min(delay * 2, 60.0)
    raise OpenDataError(url)  # pragma: no cover
