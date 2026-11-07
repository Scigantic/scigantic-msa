"""Small HTTP layer on the standard library.

- Retries 429, 5xx, timeouts and dropped connections with exponential
  backoff, and honours Retry-After (seconds or an HTTP date) up to a cap.
- Range reads check the status and Content-Range; a server that ignores
  Range and returns the whole object is sliced, and a short body is an error.
- Authorization headers are dropped when a redirect leaves the original host.
- Paths and file:// URLs are read from disk the same way.
"""
from __future__ import annotations

import email.utils
import http.client
import os
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Dict, Iterator, Mapping, Optional, Tuple

from . import __version__

PROJECT_URL = "https://github.com/Scigantic/scigantic-msa"
USER_AGENT = f"scigantic-msa/{__version__} (+{PROJECT_URL})"

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class HTTPError(Exception):
    """A request failed after retries, or returned something that is not data."""

    def __init__(self, message: str, status: int = 0, url: str = "", body: bytes = b""):
        super().__init__(message)
        self.status = status
        self.url = url
        self.body = body


class NotFound(HTTPError):
    pass


@dataclass
class Response:
    status: int
    headers: Mapping[str, str]
    body: bytes
    url: str


class _StripAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        old_host = urllib.parse.urlsplit(req.full_url).hostname
        new_host = urllib.parse.urlsplit(newurl).hostname
        if old_host != new_host:
            for h in list(new.headers):
                if h.lower() in ("authorization", "x-api-key"):
                    del new.headers[h]
            for h in list(new.unredirected_hdrs):
                if h.lower() in ("authorization", "x-api-key"):
                    del new.unredirected_hdrs[h]
        # urllib keeps most headers, but make sure Range survives.
        rng = req.get_header("Range")
        if rng and not new.get_header("Range"):
            new.add_header("Range", rng)
        return new


_opener = urllib.request.build_opener(_StripAuthOnHostChange())


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return max(0.0, dt.timestamp() - time.time())


@dataclass
class RetryPolicy:
    attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    # Longest Retry-After we are willing to sleep for before giving up.
    max_retry_after: float = 300.0


DEFAULT_RETRY = RetryPolicy()
_sleep: Callable[[float], None] = time.sleep


def _auth_headers_for(url: str) -> Dict[str, str]:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if tok:
            return {"Authorization": f"Bearer {tok}"}
    return {}


def request(
    method: str,
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 60.0,
    retry: RetryPolicy = DEFAULT_RETRY,
    retry_statuses: frozenset = RETRY_STATUSES,
    raise_for_status: bool = True,
) -> Response:
    """Send one request with retries. Returns the final response.

    Non-retryable HTTP errors are returned (``raise_for_status=False``) or
    raised as HTTPError (404 as NotFound).
    """
    hdrs = {"User-Agent": USER_AGENT}
    hdrs.update(_auth_headers_for(url))
    if headers:
        hdrs.update(headers)
    last: Optional[BaseException] = None
    last_resp: Optional[Response] = None
    for attempt in range(retry.attempts):
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        delay: Optional[float] = None
        try:
            with _opener.open(req, timeout=timeout) as r:
                body = r.read()
                return Response(r.status, dict(r.headers.items()), body, r.geturl())
        except urllib.error.HTTPError as e:
            try:
                body = e.read()
            except Exception:  # noqa: BLE001 - a broken error body is still an error
                body = b""
            resp = Response(e.code, dict(e.headers.items()) if e.headers else {}, body, url)
            if e.code not in retry_statuses:
                if raise_for_status:
                    cls = NotFound if e.code == 404 else HTTPError
                    raise cls(f"HTTP {e.code} from {_short(url)}", e.code, url, body) from None
                return resp
            last_resp = resp
            ra = _retry_after_seconds(resp.headers.get("Retry-After") or resp.headers.get("retry-after"))
            if ra is not None:
                if ra > retry.max_retry_after:
                    raise HTTPError(
                        f"HTTP {e.code} from {_short(url)}; server asks to wait {ra:.0f} s, more than "
                        f"the {retry.max_retry_after:.0f} s limit",
                        e.code, url, body,
                    ) from None
                delay = ra
            last = None
        except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, socket.gaierror):
                # The host name does not resolve; retrying will not help.
                raise HTTPError(f"cannot resolve the host of {_short(url)}: {reason}", 0, url) from None
            last = e
            last_resp = None
        if attempt == retry.attempts - 1:
            break
        if delay is None:
            delay = min(retry.max_delay, retry.base_delay * (2**attempt)) * (0.5 + random.random() / 2)
        _sleep(delay)
    if last_resp is not None:
        if raise_for_status:
            raise HTTPError(
                f"HTTP {last_resp.status} from {_short(url)} after {retry.attempts} attempts",
                last_resp.status, url, last_resp.body,
            )
        return last_resp
    raise HTTPError(f"could not reach {_short(url)} after {retry.attempts} attempts: {last}", 0, url)


def _short(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    return f"{p.scheme}://{p.netloc}{p.path}" if p.scheme else url


def is_local(url: str) -> bool:
    return url.startswith("file://") or not urllib.parse.urlsplit(url).scheme or (
        len(url) > 1 and url[1] == ":"  # Windows drive letter
    )


def local_path(url: str) -> Path:
    if url.startswith("file://"):
        return Path(urllib.request.url2pathname(urllib.parse.urlsplit(url).path))
    return Path(url)


def get_bytes(
    url: str,
    *,
    offset: int = 0,
    length: Optional[int] = None,
    timeout: float = 60.0,
    retry: RetryPolicy = DEFAULT_RETRY,
) -> bytes:
    """Read ``length`` bytes at ``offset`` (or the whole object). The result is
    exactly ``length`` bytes or an error is raised."""
    if is_local(url):
        p = local_path(url)
        try:
            with open(p, "rb") as fh:
                fh.seek(offset)
                body = fh.read() if length is None else fh.read(length)
        except FileNotFoundError:
            raise NotFound(f"{p} does not exist", 404, url) from None
        if length is not None and len(body) != length:
            raise HTTPError(f"{p}: wanted {length} bytes at {offset}, file has {len(body)}", 0, url)
        return body
    headers = {}
    ranged = offset > 0 or length is not None
    if ranged:
        end = "" if length is None else str(offset + length - 1)
        headers["Range"] = f"bytes={offset}-{end}"
    resp = request("GET", url, headers=headers, timeout=timeout, retry=retry)
    body = resp.body
    if resp.status == 206:
        cr = resp.headers.get("Content-Range") or resp.headers.get("content-range") or ""
        start = _content_range_start(cr)
        if start is not None and start != offset:
            raise HTTPError(f"{_short(url)}: asked for offset {offset}, got Content-Range {cr!r}", 206, url)
    elif resp.status == 200:
        if ranged:
            body = body[offset:] if length is None else body[offset : offset + length]
    else:
        raise HTTPError(f"{_short(url)}: unexpected HTTP {resp.status}", resp.status, url)
    if length is not None and len(body) != length:
        raise HTTPError(
            f"{_short(url)}: wanted {length} bytes, received {len(body)}", resp.status, url
        )
    return body


def _content_range_start(cr: str) -> Optional[int]:
    # "bytes 0-99/1234"
    try:
        unit, rest = cr.split(" ", 1)
        return int(rest.split("-", 1)[0])
    except (ValueError, IndexError):
        return None


def open_stream(url: str, *, offset: int = 0, timeout: float = 60.0) -> IO[bytes]:
    """Open a readable stream from ``offset`` to the end. The caller closes it.
    Used for big Stockholm files we stop reading early."""
    if is_local(url):
        fh = open(local_path(url), "rb")
        fh.seek(offset)
        return fh
    hdrs = {"User-Agent": USER_AGENT}
    hdrs.update(_auth_headers_for(url))
    if offset:
        hdrs["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=hdrs)
    last: Optional[BaseException] = None
    for attempt in range(DEFAULT_RETRY.attempts):
        try:
            r = _opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUSES or attempt == DEFAULT_RETRY.attempts - 1:
                raise HTTPError(f"HTTP {e.code} from {_short(url)}", e.code, url) from None
            last = e
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            last = e
            if attempt == DEFAULT_RETRY.attempts - 1:
                break
        else:
            if offset and r.status == 200:
                # Range ignored: skip to the offset ourselves.
                remaining = offset
                while remaining:
                    chunk = r.read(min(remaining, 1 << 20))
                    if not chunk:
                        raise HTTPError(f"{_short(url)}: shorter than offset {offset}", 200, url)
                    remaining -= len(chunk)
            return r
        _sleep(min(DEFAULT_RETRY.max_delay, DEFAULT_RETRY.base_delay * (2**attempt)))
    raise HTTPError(f"could not reach {_short(url)}: {last}", 0, url)


def iter_lines(stream: IO[bytes], chunk: int = 1 << 16) -> Iterator[bytes]:
    buf = b""
    while True:
        data = stream.read(chunk)
        if not data:
            break
        buf += data
        *lines, buf = buf.split(b"\n")
        for line in lines:
            yield line
    if buf:
        yield buf


def split_url(url: str) -> Tuple[str, str]:
    p = urllib.parse.urlsplit(url)
    return p.netloc, p.path
