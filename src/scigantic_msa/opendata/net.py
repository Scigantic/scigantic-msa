"""Anonymous reads from the open data hosts.

Every request checks SCIGANTIC_MSA_NO_REMOTE first, like the search
clients, so the switch blocks open data lookups too. Requests are made one
at a time from a process (a lock), and UniProt requests are spaced out, to
be polite to hosts that serve everyone for free. No credentials are ever
sent.

A host name that does not resolve, or a host that refuses the connection,
is not tried again: that is a machine offline or on a closed network, and
waiting will not help. Each request has a total deadline (``DEADLINE``
seconds to get an answer, over all its attempts), and a body must keep
coming at ``MIN_RATE`` bytes a second on average once ``GRACE`` seconds
have passed, so a server that sends a byte now and then cannot hold a run.
"""
from __future__ import annotations

import errno
import http.client
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from .. import http as _http

__all__ = ["OpenDataError", "NotFound", "Unreachable", "get", "head", "stream", "Hit"]


class OpenDataError(RuntimeError):
    """An open data host could not be read."""


class NotFound(OpenDataError):
    pass


class Unreachable(OpenDataError):
    """The host name does not resolve, or the host refused the connection."""


class TooSlow(OpenDataError):
    """A body came in slower than MIN_RATE. Not retried."""


#: One request at a time per process.
_lock = threading.RLock()
#: Seconds between two requests to rest.uniprot.org.
UNIPROT_SPACING = 0.34
_last_uniprot = [0.0]
#: Attempts for one request (network errors, 429, 5xx).
ATTEMPTS = 5
#: Most seconds to get an answer to one request, over all its attempts.
DEADLINE = 180.0
#: A body must arrive at this many bytes a second on average, once GRACE
#: seconds have passed since the request was answered.
MIN_RATE = 8 * 1024
GRACE = 60.0


def _unreachable(e: BaseException) -> bool:
    """True for errors that a retry will not fix: a name that does not
    resolve, a refused connection, no route to the network."""
    reason = getattr(e, "reason", e)
    if isinstance(reason, (socket.gaierror, ConnectionRefusedError)):
        return True
    return isinstance(reason, OSError) and reason.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH)


class _Paced:
    """A response read through ``read1``, which returns as soon as any bytes
    arrive, so the pace is checked as the body comes in: once GRACE seconds
    have passed, reads raise TooSlow when the body has come in slower than
    MIN_RATE on average. ``read`` and ``readline`` behave as on the
    response."""

    def __init__(self, r: Any, url: str):
        self._r = r
        self._url = url
        self._t0 = time.monotonic()
        self._n = 0
        self._buf = bytearray()
        self._pos = 0
        self._eof = False

    def _more(self) -> bool:
        if self._eof:
            return False
        reader = getattr(self._r, "read1", None) or self._r.read
        b = reader(1 << 16)
        self._n += len(b)
        spent = time.monotonic() - self._t0
        if spent > GRACE and spent > GRACE + self._n / MIN_RATE:
            raise TooSlow(f"{_http._short(self._url)} came in too slowly ({self._n} bytes in {spent:.0f} s)")
        if not b:
            self._eof = True
            return False
        if self._pos > (1 << 16):
            del self._buf[: self._pos]
            self._pos = 0
        self._buf += b
        return True

    def _take(self, k: int) -> bytes:
        out = bytes(self._buf[self._pos: self._pos + k])
        self._pos += len(out)
        return out

    def read(self, n: int = -1) -> bytes:
        while (n is None or n < 0 or len(self._buf) - self._pos < n) and self._more():
            pass
        left = len(self._buf) - self._pos
        return self._take(left if n is None or n < 0 else min(n, left))

    def readline(self, limit: int = -1) -> bytes:
        start = self._pos
        while True:
            i = self._buf.find(b"\n", start)
            if i >= 0 and (limit < 0 or i - self._pos < limit):
                return self._take(i + 1 - self._pos)
            if limit >= 0 and len(self._buf) - self._pos >= limit:
                return self._take(limit)
            start = len(self._buf)
            pos = self._pos
            if not self._more():
                return self._take(len(self._buf) - self._pos)
            start -= pos - self._pos  # the buffer may have been compacted

    def __getattr__(self, name: str) -> Any:
        return getattr(self._r, name)


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
           timeout: float = 60.0) -> Iterator[Tuple[Any, Dict[str, str]]]:
    """Open ``url`` for reading. Yields (response, headers). Retries the
    opening on network errors, 429 and 5xx. Raises NotFound for 403 and
    404 (S3 answers 403 for a missing key in a bucket that cannot be
    listed), Unreachable when the host cannot be reached at all,
    OpenDataError otherwise. The response's reads raise TooSlow when the
    body comes in too slowly."""
    with _lock:
        delay = 2.0
        resp = None
        end = time.monotonic() + DEADLINE
        for attempt in range(ATTEMPTS):
            _http.refuse_if_disabled()
            _space(url)
            req = urllib.request.Request(url, headers=_headers(headers))
            left = end - time.monotonic()
            try:
                resp = _http._opener.open(req, timeout=max(1.0, min(timeout, left)))
                break
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    raise NotFound(f"HTTP {e.code} from {_http._short(url)}") from None
                if e.code not in _http.RETRY_STATUSES or attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"HTTP {e.code} from {_http._short(url)}") from None
                ra = _http._retry_after_seconds(e.headers.get("Retry-After") if e.headers else None)
                wait = min(ra if ra is not None else delay, 120.0)
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError,
                    ConnectionError, OSError) as e:
                if _unreachable(e):
                    raise Unreachable(f"could not reach {_http._short(url)}: {getattr(e, 'reason', e)}") from None
                if attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"could not read {_http._short(url)}: {getattr(e, 'reason', e)}") from None
                wait = delay
                last = str(getattr(e, "reason", e))
            if time.monotonic() + wait > end:
                raise OpenDataError(f"no answer from {_http._short(url)} within {DEADLINE:.0f} s ({last})")
            _http._sleep(wait)
            delay = min(delay * 2, 60.0)
        assert resp is not None
        try:
            yield _Paced(resp, url), {k.lower(): v for k, v in resp.headers.items()}
        finally:
            resp.close()


def get(url: str, *, headers: Optional[Mapping[str, str]] = None, timeout: float = 60.0,
        max_bytes: Optional[int] = None) -> Tuple[bytes, Dict[str, str]]:
    """The whole body of ``url`` and its headers. A body cut short by the
    network is read again from the start, up to ATTEMPTS times and within
    DEADLINE seconds in all."""
    end = time.monotonic() + DEADLINE
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
            if attempt == ATTEMPTS - 1 or time.monotonic() > end:
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
    with that status, not an error. Unreachable hosts are not tried again,
    and the attempts end within DEADLINE seconds."""
    with _lock:
        delay = 2.0
        end = time.monotonic() + DEADLINE
        for attempt in range(ATTEMPTS):
            _http.refuse_if_disabled()
            _space(url)
            req = urllib.request.Request(url, method="HEAD", headers=_headers())
            left = end - time.monotonic()
            try:
                with _http._opener.open(req, timeout=max(1.0, min(timeout, left))) as r:
                    return Hit(r.status, {k.lower(): v for k, v in r.headers.items()})
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    return Hit(e.code, {})
                if e.code not in _http.RETRY_STATUSES or attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"HTTP {e.code} from {_http._short(url)}") from None
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError,
                    ConnectionError, OSError) as e:
                if _unreachable(e):
                    raise Unreachable(f"could not reach {_http._short(url)}: {getattr(e, 'reason', e)}") from None
                if attempt == ATTEMPTS - 1:
                    raise OpenDataError(f"could not reach {_http._short(url)}: {getattr(e, 'reason', e)}") from None
            if time.monotonic() + delay > end:
                raise OpenDataError(f"no answer from {_http._short(url)} within {DEADLINE:.0f} s")
            _http._sleep(delay)
            delay = min(delay * 2, 60.0)
    raise OpenDataError(url)  # pragma: no cover
