"""An in-memory stand-in for the /api/msa routes, following the backend
design contract (status codes, error shape, presigned PUT with a signed
Content-Length, server-side check of the first record's hash)."""
from __future__ import annotations

import base64
import gzip
import json
import urllib.parse
import uuid
from typing import Any, Dict, Optional

from scigantic_msa.a3m import a3m_query
from scigantic_msa.seqkey import seq_sha256

CODE = "7K3M-9QX2-4R8T"


def ok(data: Any, status: int = 200) -> tuple:
    return status, {"Content-Type": "application/json"}, json.dumps({"success": True, "data": data}).encode()


def err(status: int, message: str, code: Optional[str] = None, details: Any = None) -> tuple:
    e: Dict[str, Any] = {"status": status, "message": message}
    if code:
        e["code"] = code
    if details is not None:
        e["details"] = details
    return status, {"Content-Type": "application/json"}, json.dumps({"error": e}).encode()


class FakeBackend:
    def __init__(self, server, *, existing=("old@lab.edu",), notebooks=None):
        self.srv = server
        self.users: Dict[str, Dict[str, Any]] = {e: {"id": f"sub-{i}", "email": e} for i, e in enumerate(existing)}
        self.tokens: Dict[str, str] = {}
        self.items: Dict[str, Dict[str, Any]] = {}
        self.blobs: Dict[str, bytes] = {}
        self.codes_sent = []
        self.notebooks = notebooks or {}
        self.notebook_mints = 0
        self.expire_next = set()
        self.page_size = 2
        r = server.route
        r("POST", "/api/msa/login", self.login)
        r("POST", "/api/msa/token", self.token)
        r("POST", "/api/msa/token/notebook", self.nb_token)
        r("POST", "/api/msa/token/refresh", self.refresh)
        r("GET", "/api/msa/account", self.account)
        r("POST", "/api/msa/alignments", self.create)
        r("GET", "/api/msa/alignments", self.list)
        r("POST", "/api/msa/lookup", self.lookup)
        server.route_prefix("POST", "/api/msa/alignments/", self.complete)
        server.route_prefix("GET", "/api/msa/alignments/", self.get_or_download)
        server.route_prefix("DELETE", "/api/msa/alignments/", self.delete)
        server.route_prefix("PUT", "/s3/put/", self.s3_put)
        server.route_prefix("GET", "/s3/get/", self.s3_get)

    # -- auth
    def user_for(self, req) -> Optional[Dict[str, Any]]:
        h = req.headers.get("Authorization", "")
        if not h.startswith("Bearer "):
            return None
        tok = h[7:]
        if tok in self.expire_next:
            self.expire_next.discard(tok)
            self.tokens.pop(tok, None)
            return None
        email = self.tokens.get(tok)
        return self.users.get(email) if email else None

    def login(self, req):
        email = req.json().get("email", "")
        if "@" not in email:
            return err(400, "invalid email")
        self.codes_sent.append(email)
        return ok({"email": email, "expiresAt": "2026-11-06T15:10:00Z", "accountExists": email in self.users,
                   "termsUrl": "https://scigantic.com/terms"})

    def token(self, req):
        b = req.json()
        if b.get("code") != CODE:
            return err(401, "Invalid or expired code")
        created = False
        if b["email"] not in self.users:
            if b.get("acceptTerms") is not True:
                return err(400, "Accept the terms to create an account", "terms_required")
            self.users[b["email"]] = {"id": f"sub-{len(self.users)}", "email": b["email"]}
            created = True
        tok = "tok-" + uuid.uuid4().hex
        self.tokens[tok] = b["email"]
        return ok({"token": tok, "expiresAt": 1900000000, "userId": self.users[b["email"]]["id"],
                   "email": b["email"], "accountCreated": created})

    def nb_token(self, req):
        live = req.headers.get("X-Scigantic-Liveness")
        if not live:
            return err(401, "missing liveness header")
        nb = self.notebooks.get(req.json().get("notebookId"))
        if not nb or nb["live"] != live:
            return err(403, "bad notebook credential")
        if nb.get("trial"):
            return err(403, "Sign in to keep alignments", "trial_notebook")
        if nb.get("status", "running") != "running":
            return err(409, "notebook is not running", "notebook_not_running")
        self.notebook_mints += 1
        tok = "nbtok-" + uuid.uuid4().hex
        self.tokens[tok] = nb["email"]
        return ok({"token": tok, "expiresAt": 1900000000, "email": nb["email"]})

    def refresh(self, req):
        u = self.user_for(req)
        if not u:
            return err(401, "expired")
        tok = "tok-" + uuid.uuid4().hex
        self.tokens[tok] = u["email"]
        return ok({"token": tok, "expiresAt": 1900000000})

    def account(self, req):
        u = self.user_for(req)
        if not u:
            return err(401, "Run: scigantic-msa login --email you@example.org")
        mine = [i for i in self.items.values() if i["_owner"] == u["id"] and i["status"] == "ready"]
        return ok({"email": u["email"], "plan": "free",
                   "limits": {"maxBytes": 2 << 30, "maxCount": 10000, "maxObjectBytes": 100 << 20, "maxPerSequence": 20},
                   "usage": {"bytes": sum(i["sizeBytes"] for i in mine), "count": len(mine), "pendingBytes": 0, "pendingCount": 0}})

    # -- alignments
    def public(self, item):
        return {k: v for k, v in item.items() if not k.startswith("_")}

    def create(self, req):
        u = self.user_for(req)
        if not u:
            return err(401, "unauthorized")
        b = req.json()
        if len(b.get("seqSha256", "")) != 64 or b.get("format") not in ("a3m", "sto", "hhr", "chai-parquet"):
            return err(400, "validation failed")
        if not 0 < int(b.get("sizeBytes", 0)) <= 100 << 20:
            return err(400, "bad size")
        if len(json.dumps(b.get("params") or {})) > 2048:
            return err(400, "params too large")
        aid = str(uuid.uuid4())
        self.items[aid] = dict(b, alignmentId=aid, status="pending", visibility="private", via="cli",
                               createdAt="2026-11-06T00:00:00Z", _owner=u["id"])
        ctype = "application/gzip" if b.get("compressed") else "text/plain"
        return ok({"alignmentId": aid, "status": "pending",
                   "upload": {"method": "PUT", "url": f"{self.srv.url}/s3/put/{aid}?X-Amz-Signature=x",
                              "headers": {"Content-Length": str(b["sizeBytes"]), "Content-Type": ctype},
                              "expiresAt": "2026-11-06T00:15:00Z"}}, 201)

    def s3_put(self, req):
        aid = req.path_only.rsplit("/", 1)[1]
        it = self.items.get(aid)
        if not it:
            return 403, {}, b"AccessDenied"
        if int(req.headers.get("Content-Length", -1)) != it["sizeBytes"] or len(req.body) != it["sizeBytes"]:
            return 403, {}, b"SignatureDoesNotMatch"
        self.blobs[aid] = req.body
        return 200, {}, b""

    def complete(self, req):
        parts = req.path_only.split("/")
        if len(parts) != 6 or parts[5] != "complete":
            return err(404, "not found")
        u = self.user_for(req)
        if not u:
            return err(401, "unauthorized")
        aid = parts[4]
        it = self.items.get(aid)
        if not it or it["_owner"] != u["id"]:
            return err(404, "not found")
        if it["status"] == "ready":
            return ok(self.public(it))
        blob = self.blobs.get(aid)
        if blob is None:
            return err(422, "upload missing", "upload_missing")
        if len(blob) != it["sizeBytes"]:
            return err(422, "size mismatch", "size_mismatch")
        head = gzip.decompress(blob)[:65536].decode() if it.get("compressed") else blob[:65536].decode()
        q = a3m_query(head)
        if q is None or seq_sha256(q) != it["seqSha256"]:
            del self.blobs[aid]
            del self.items[aid]
            return err(422, "first record does not match seqSha256", "sequence_mismatch")
        it["status"] = "ready"
        it["completedAt"] = "2026-11-06T00:01:00Z"
        return ok(self.public(it))

    def _mine(self, req):
        u = self.user_for(req)
        return u, ([] if not u else [i for i in self.items.values() if i["_owner"] == u["id"] and i["status"] == "ready"])

    def lookup(self, req):
        u, mine = self._mine(req)
        if not u:
            return err(401, "unauthorized")
        want = req.json()["seqSha256"]
        found = {h: [self.public(i) for i in mine if i["seqSha256"] == h] for h in want}
        return ok({"found": {k: v for k, v in found.items() if v}, "missing": [k for k, v in found.items() if not v]})

    def list(self, req):
        u, mine = self._mine(req)
        if not u:
            return err(401, "unauthorized")
        q = urllib.parse.parse_qs(req.query)
        if "seqSha256" in q:
            mine = [i for i in mine if i["seqSha256"] == q["seqSha256"][0]]
        start = int(base64.b64decode(q["cursor"][0]).decode()) if "cursor" in q else 0
        page = mine[start : start + self.page_size]
        nxt = base64.b64encode(str(start + self.page_size).encode()).decode() if start + self.page_size < len(mine) else None
        return ok({"alignments": [self.public(i) for i in page], "nextCursor": nxt})

    def get_or_download(self, req):
        u = self.user_for(req)
        if not u:
            return err(401, "unauthorized")
        parts = req.path_only.split("/")
        aid = parts[4]
        it = self.items.get(aid)
        if not it or it["_owner"] != u["id"] or it["status"] != "ready":
            return err(404, "not found")
        if len(parts) == 6 and parts[5] == "download":
            return ok({"url": f"{self.srv.url}/s3/get/{aid}", "expiresAt": "x", "sizeBytes": it["sizeBytes"],
                       "contentSha256": it["contentSha256"], "compressed": it["compressed"]})
        return ok(self.public(it))

    def s3_get(self, req):
        aid = req.path_only.rsplit("/", 1)[1]
        return (200, {}, self.blobs[aid]) if aid in self.blobs else (403, {}, b"")

    def delete(self, req):
        u = self.user_for(req)
        aid = req.path_only.split("/")[4]
        it = self.items.get(aid)
        if not u or not it or it["_owner"] != u["id"]:
            return err(404, "not found")
        del self.items[aid]
        self.blobs.pop(aid, None)
        return ok({"deleted": True})
