"""Store alignments in your Scigantic account and fetch them from any machine.

Stored alignments are private to you. Nothing is shared or published.

Signing in:

- Terminal: ``scigantic-msa login --email you@lab.edu`` emails a code. Paste
  it and you have a 30-day token. If the address has no Scigantic account yet,
  one is created when you confirm the code (you are asked to accept the terms
  first). The token is saved with mode 0600 in ``~/.config/scigantic/msa.json``.
- Scigantic notebooks: no step at all. The notebook's own credentials are
  exchanged for a 12-hour token kept in memory only.
- Elsewhere, ``SCIGANTIC_MSA_TOKEN`` can hold a token.

Uploads go straight to storage with a presigned URL; the API then checks the
file's first record hashes to the declared sequence before keeping it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from . import http
from .msa import MSA
from .seqkey import is_seq_sha256

__all__ = ["AccountClient", "AccountError", "NotLoggedIn", "config_path", "load_config", "save_config"]

DEFAULT_API_URL = "https://api.scigantic.com"
LOGIN_HINT = "Run: scigantic-msa login --email you@example.org"


def config_path() -> Path:
    env = os.environ.get("SCIGANTIC_MSA_CONFIG")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "scigantic" / "msa.json"


def load_config() -> Dict[str, Any]:
    try:
        d = json.loads(config_path().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    tmp = p.with_name("." + p.name + ".tmp")
    # Create with 0600 from the start so the token is never world readable.
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(cfg, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


class AccountError(Exception):
    def __init__(self, status: int, message: str, code: Optional[str] = None, details: Any = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.details = details


class NotLoggedIn(AccountError):
    def __init__(self, message: str = f"Not logged in. {LOGIN_HINT}"):
        super().__init__(401, message, "not_logged_in")


def in_notebook() -> bool:
    return bool(os.environ.get("SCIGANTIC_LIVENESS_TOKEN") and os.environ.get("SCIGANTIC_NOTEBOOK_ID"))


def _source_label(msa: MSA) -> str:
    backend = msa.provenance.get("backend")
    if backend == "colabfold":
        return "colabfold-api"
    if backend == "nvidia":
        return "nvidia-msa-search"
    if msa.provenance.get("from") == "upload":
        return "upload"
    if msa.provenance.get("from") == "index":
        return "index"
    return "other"


class AccountClient:
    """Client for the /api/msa routes."""

    def __init__(self, api_url: Optional[str] = None, token: Optional[str] = None, *, timeout: float = 60.0):
        cfg = load_config()
        self.base = (api_url or os.environ.get("SCIGANTIC_API_URL") or cfg.get("apiUrl") or DEFAULT_API_URL).rstrip("/")
        self.timeout = timeout
        self._explicit = token
        self._token: Optional[str] = token
        self._via: Optional[str] = "explicit" if token else None
        if self._token is None:
            env = os.environ.get("SCIGANTIC_MSA_TOKEN")
            if env:
                self._token, self._via = env, "env"
            elif cfg.get("token") and _not_expired(cfg.get("expiresAt")):
                self._token, self._via = cfg["token"], "config"

    # ------------------------------------------------------------ auth state
    @property
    def logged_in(self) -> bool:
        return self._token is not None or in_notebook()

    def auth_method(self) -> Optional[str]:
        if self._token is not None:
            return self._via
        if in_notebook():
            return "notebook"
        return None

    def _ensure_token(self) -> str:
        if self._token is not None:
            return self._token
        if in_notebook():
            self._mint_notebook()
            assert self._token is not None
            return self._token
        raise NotLoggedIn()

    def _mint_notebook(self) -> None:
        d = self.request(
            "POST", "/api/msa/token/notebook",
            {"notebookId": os.environ["SCIGANTIC_NOTEBOOK_ID"]},
            auth=False,
            headers={"X-Scigantic-Liveness": os.environ["SCIGANTIC_LIVENESS_TOKEN"]},
        )
        self._token = d["token"]
        self._via = "notebook"

    # ------------------------------------------------------------ transport
    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        auth: bool = True,
        headers: Optional[Dict[str, str]] = None,
        _retried: bool = False,
    ) -> Any:
        h = {"Accept": "application/json"}
        if headers:
            h.update(headers)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        if auth:
            h["Authorization"] = f"Bearer {self._ensure_token()}"
        resp = _send(
            method, self.base + path, data=data, headers=h, timeout=self.timeout,
            retry=http.RetryPolicy(attempts=3, base_delay=1.0, max_delay=10.0),
            retry_statuses=frozenset({502, 503, 504}),
        )
        try:
            payload = json.loads(resp.body) if resp.body else {}
        except ValueError:
            payload = {}
        if resp.status == 401 and auth and self._via == "notebook" and not _retried:
            self._token = None
            self._mint_notebook()
            return self.request(method, path, body, auth=auth, headers=headers, _retried=True)
        if resp.status >= 400:
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                msg = err.get("message") or f"HTTP {resp.status}"
                code, details = err.get("code"), err.get("details")
            else:
                msg = (err if isinstance(err, str) else None) or f"HTTP {resp.status} from {self.base}{path}"
                code = details = None
            if resp.status == 401 and auth and self._via in ("config", "env", "explicit"):
                msg = f"{msg}. {LOGIN_HINT}"
            raise AccountError(resp.status, str(msg), code, details)
        if not isinstance(payload, dict) or payload.get("success") is False:
            raise AccountError(resp.status, f"unexpected answer from {self.base}{path}")
        return payload.get("data", payload)

    # ------------------------------------------------------------ login
    def login_start(self, email: str) -> Dict[str, Any]:
        return self.request("POST", "/api/msa/login", {"email": email}, auth=False)

    def login_finish(self, email: str, code: str, *, accept_terms: bool = False, save: bool = True) -> Dict[str, Any]:
        d = self.request("POST", "/api/msa/token",
                         {"email": email, "code": code.strip(), "acceptTerms": bool(accept_terms)}, auth=False)
        self._token, self._via = d["token"], "config"
        if save:
            cfg = load_config()
            cfg.update({"token": d["token"], "email": d.get("email", email),
                        "expiresAt": d.get("expiresAt"), "apiUrl": self.base})
            save_config(cfg)
        return d

    def refresh(self) -> Dict[str, Any]:
        d = self.request("POST", "/api/msa/token/refresh", {})
        self._token = d["token"]
        if self._via == "config":
            cfg = load_config()
            cfg.update({"token": d["token"], "expiresAt": d.get("expiresAt")})
            save_config(cfg)
        return d

    def maybe_refresh(self, within_days: float = 7.0) -> None:
        """Slide a saved CLI token forward when it is close to expiry."""
        if self._via != "config":
            return
        exp = load_config().get("expiresAt")
        try:
            if exp and float(exp) - time.time() < within_days * 86400:
                self.refresh()
        except (AccountError, ValueError, TypeError):
            pass

    @staticmethod
    def logout() -> None:
        cfg = load_config()
        for k in ("token", "expiresAt"):
            cfg.pop(k, None)
        save_config(cfg)

    def account(self) -> Dict[str, Any]:
        return self.request("GET", "/api/msa/account")

    # ------------------------------------------------------------ alignments
    def save(self, msa: MSA, *, label: Optional[str] = None) -> str:
        if msa.source == "account" and msa.account_id:
            return msa.account_id
        data = msa.to_bundle()
        meta = {
            "seqSha256": msa.seq_sha256,
            "queryLength": len(msa.query),
            "format": "a3m",
            "kind": "combined" if msa.uniprot_rows else "unpaired",
            "compressed": True,
            "sizeBytes": len(data),
            "contentSha256": hashlib.sha256(data).hexdigest(),
            "source": _source_label(msa),
            "tool": "scigantic-msa",
            "toolVersion": _version(),
            "databases": _databases(msa),
            "params": _small_params(msa),
        }
        if label:
            meta["label"] = label[:120]
        start = self.request("POST", "/api/msa/alignments", meta)
        aid = start["alignmentId"]
        up = start["upload"]
        hdrs = {k: str(v) for k, v in (up.get("headers") or {}).items()}
        hdrs.setdefault("Content-Length", str(len(data)))
        if hdrs.get("Content-Length") != str(len(data)):
            raise AccountError(400, "the server signed a different size than this upload")
        resp = _send(up.get("method", "PUT"), up["url"], data=data, headers=hdrs,
                     timeout=max(self.timeout, 300), retry=http.RetryPolicy(attempts=3, base_delay=2.0))
        if resp.status not in (200, 201, 204):
            raise AccountError(resp.status, f"upload to storage failed: HTTP {resp.status}")
        rec = self.request("POST", f"/api/msa/alignments/{urllib.parse.quote(aid)}/complete")
        if rec.get("status") != "ready":
            raise AccountError(500, f"alignment {aid} is {rec.get('status')!r} after upload, not ready")
        return aid

    _refreshed = False

    def lookup(self, shas: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not AccountClient._refreshed:
            AccountClient._refreshed = True
            self.maybe_refresh()
        shas = list(dict.fromkeys(shas))
        for s in shas:
            if not is_seq_sha256(s):
                raise ValueError(f"not a seq_sha256: {s!r}")
        found: Dict[str, List[Dict[str, Any]]] = {}
        for i in range(0, len(shas), 100):
            d = self.request("POST", "/api/msa/lookup", {"seqSha256": shas[i : i + 100]})
            for k, v in (d.get("found") or {}).items():
                found.setdefault(k, []).extend(v)
        return found

    def list(self, seq_sha256: Optional[str] = None, *, limit: int = 100) -> Iterator[Dict[str, Any]]:
        cursor: Optional[str] = None
        while True:
            q: Dict[str, Any] = {"limit": min(max(limit, 1), 200)}
            if seq_sha256:
                q["seqSha256"] = seq_sha256
            if cursor:
                q["cursor"] = cursor
            d = self.request("GET", "/api/msa/alignments?" + urllib.parse.urlencode(q))
            for a in d.get("alignments") or []:
                yield a
            cursor = d.get("nextCursor")
            if not cursor:
                return

    def get(self, alignment_id: str) -> Dict[str, Any]:
        return self.request("GET", f"/api/msa/alignments/{urllib.parse.quote(alignment_id)}")

    def download_bytes(self, alignment_id: str) -> bytes:
        d = self.request("GET", f"/api/msa/alignments/{urllib.parse.quote(alignment_id)}/download")
        resp = _send("GET", d["url"], timeout=max(self.timeout, 300))
        if resp.status != 200:
            raise AccountError(resp.status, f"download from storage failed: HTTP {resp.status}")
        body = resp.body
        if d.get("sizeBytes") is not None and len(body) != int(d["sizeBytes"]):
            raise AccountError(500, f"downloaded {len(body)} bytes, expected {d['sizeBytes']}")
        if d.get("contentSha256") and hashlib.sha256(body).hexdigest() != d["contentSha256"]:
            raise AccountError(500, "downloaded alignment does not match its checksum")
        return body

    def download(self, alignment_id: str, *, query: Optional[str] = None) -> MSA:
        body = self.download_bytes(alignment_id)
        msa = MSA.from_bundle(body, source="account", query=query)
        msa.account_id = alignment_id
        msa.provenance = dict(msa.provenance, account_alignment_id=alignment_id, stored_from=msa.provenance.get("from"))
        return msa

    def delete(self, alignment_id: str) -> bool:
        d = self.request("DELETE", f"/api/msa/alignments/{urllib.parse.quote(alignment_id)}")
        return bool(d.get("deleted"))


def _send(method: str, url: str, **kw: Any) -> http.Response:
    """http.request that reports an unreachable server as an AccountError, so
    callers that fall through to other sources when the account fails also do
    so when the network is down."""
    try:
        return http.request(method, url, raise_for_status=False, **kw)
    except http.HTTPError as e:
        raise AccountError(e.status, str(e), "unreachable") from None


# What POST /api/msa/alignments accepts for databases and params values. The
# metadata is a label only, so anything outside it is dropped, not sent.
_DB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,63}$")
_MAX_DATABASES = 16
_MAX_PARAM_TEXT = 200


def _databases(msa: MSA) -> List[str]:
    names = [str(x) for x in (msa.provenance.get("databases") or [])]
    return [n for n in names if _DB_NAME.match(n)][:_MAX_DATABASES]


def _small_params(msa: MSA) -> Dict[str, Any]:
    p = msa.provenance
    out: Dict[str, Any] = {"result": msa.source, "depth": msa.depth}
    picked: Dict[str, Any] = {k: p.get(k) for k in ("backend", "mode", "index_source", "source_id", "build_id", "ticket")}
    if msa.derived:
        picked["derived_from"] = msa.derived.get("parent_seq_sha256")
        picked["identity"] = msa.derived.get("identity")
    for k, v in picked.items():
        if isinstance(v, bool) or (isinstance(v, (int, float)) and math.isfinite(v)):
            out[k] = v
        elif isinstance(v, str) and v.isprintable() and len(v) <= _MAX_PARAM_TEXT:
            out[k] = v
    while len(json.dumps(out)) > 1800 and len(out) > 2:
        out.popitem()
    return out


def _not_expired(exp: Any) -> bool:
    try:
        return exp is None or float(exp) > time.time() + 60
    except (TypeError, ValueError):
        return True


def _version() -> str:
    from . import __version__

    return __version__
