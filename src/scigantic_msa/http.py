"""Small HTTP layer on the standard library, used only to talk to the MSA
server you chose.

- Retries 429, 5xx, timeouts and dropped connections with exponential
  backoff, and honours Retry-After (seconds or an HTTP date) up to a cap.
- Authorization headers are dropped when a redirect leaves the original host.
- Nothing is sent while SCIGANTIC_MSA_NO_REMOTE is set: every request to a
  search server goes through ``request`` here, which refuses first.
"""
from __future__ import annotations

import email.utils
import http.client
import os
import random
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

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


class RemoteNotAllowed(PermissionError):
    """A search was asked for without remote=True, or SCIGANTIC_MSA_NO_REMOTE is set."""


NO_REMOTE_MESSAGE = "SCIGANTIC_MSA_NO_REMOTE is set, so no sequence is sent to any server"


def remote_disabled() -> bool:
    """SCIGANTIC_MSA_NO_REMOTE turns every remote backend off, whatever the
    code or config asks for. Any value but 0, false, no or off counts, so a
    value we did not think of fails safe; that includes one of only spaces.
    An empty value is the shell's way to unset a variable, so it is off."""
    value = os.environ.get("SCIGANTIC_MSA_NO_REMOTE", "")
    if value == "":
        return False
    return value.strip().lower() not in ("0", "false", "no", "off")


def refuse_if_disabled() -> None:
    """Raise RemoteNotAllowed when SCIGANTIC_MSA_NO_REMOTE is set. Every
    client calls this before each request, and ``request`` calls it again."""
    if remote_disabled():
        raise RemoteNotAllowed(NO_REMOTE_MESSAGE)


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
    raised as HTTPError (404 as NotFound). Raises RemoteNotAllowed, and
    sends nothing, while SCIGANTIC_MSA_NO_REMOTE is set. The switch is
    checked again before every retry, so setting it during a backoff stops
    the request from being sent again. A URL with a user name or password is
    refused before any name lookup.
    """
    refuse_credentials(url, "the server URL")
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    for k, v in hdrs.items():
        check_header_value(v, k)
    last: Optional[BaseException] = None
    last_resp: Optional[Response] = None
    for attempt in range(retry.attempts):
        refuse_if_disabled()
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
    """The URL for a message: scheme, host and path, never a user name or
    password, a query or a fragment."""
    p = urllib.parse.urlsplit(url)
    if not p.scheme:
        return "the server" if "@" in url else url
    return f"{p.scheme}://{p.netloc.rpartition('@')[2]}{p.path}"


def split_url(url: str) -> Tuple[str, str]:
    p = urllib.parse.urlsplit(url)
    return p.netloc, p.path


def ascii_host(host: str) -> Optional[str]:
    """A host name in its ASCII form (IDNA, so a fullwidth or accented name
    becomes the one DNS answers for), or None when it has none."""
    try:
        return host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None


def refuse_credentials(url: str, what: str) -> None:
    """Refuse a URL with a user name or password in it (``user:pass@host``),
    or with a backslash, which no http URL needs. Every entry point calls
    this first, on the URL as given, before anything else can repeat it.

    It looks at the host part the way urllib.parse reads it, which is how
    the URL would be connected to, and also at the text after the scheme
    with any slashes removed, so that ``//user:pass@host``,
    ``http:/user:pass@host`` and the like are caught too. The error does not
    repeat the URL, so the password is not printed."""
    if "\\" in url:
        raise ValueError(f"{what} has a backslash in it; an http URL uses '/'. "
                         "Write it with forward slashes only")
    # NFKC, as urllib.parse checks a host with it: a fullwidth at sign is '@'.
    s = unicodedata.normalize("NFKC", url.strip())
    try:
        netloc = urllib.parse.urlsplit(s).netloc
    except ValueError:
        # urllib's own error repeats the host part, which may hold a password.
        raise ValueError(f"{what} is not a valid URL; give the server URL with http:// or https:// and a host name") from None
    rest = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*:", "", s).lstrip("/")
    if "@" in netloc or "@" in re.split(r"[/?#]", rest, maxsplit=1)[0]:
        raise ValueError(f"{what} has a user name or password in it (the part before '@'); "
                         "remove it. scigantic-msa does not send credentials in a URL")


def check_header_value(value: str, what: str) -> None:
    """Refuse a header value urllib cannot send: a line break or other
    control character, or a character outside Latin-1. Called before
    anything says it is sending, so the error is this one and not a codec
    error after the announcement. The error does not repeat the value."""
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{what} has a line break or another control character in it; "
                         "it goes in an HTTP header, so give it on one line")
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        raise ValueError(f"{what} has characters an HTTP header cannot carry (only Latin-1 is allowed); "
                         "write it in ASCII, such as an email address") from None


def require_ascii(url: str, what: str) -> None:
    """Refuse a URL with any non-ASCII character. urllib would send the
    host as typed in the Host header and fail with a latin-1 error after
    the tool had already said it was sending; and a fullwidth spelling of a
    host name is the same server as its ASCII one. The error gives the
    ASCII spelling when there is one."""
    if url.isascii():
        return
    msg = f"{what} {url!r} has characters that are not ASCII; write the URL in ASCII"
    parts = urllib.parse.urlsplit(url.strip())
    plain = ascii_host(parts.hostname or "") if parts.hostname else None
    if plain and parts.path.isascii():
        try:
            port = f":{parts.port}" if parts.port is not None else ""
        except ValueError:
            port = ""
        msg += f", as in {parts.scheme}://{plain}{port}{parts.path}"
    raise ValueError(msg)
