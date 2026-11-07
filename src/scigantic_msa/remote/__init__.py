"""Remote MSA search backends. A sequence goes to one only when you ask with remote=True."""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Optional, Set

from .colabfold import DEFAULT_HOST, ColabFoldClient, JobFailed, RateLimited, RemoteError, RemoteResult
from .nvidia import NvidiaClient

__all__ = [
    "ColabFoldClient", "NvidiaClient", "RemoteError", "RateLimited", "JobFailed", "RemoteResult",
    "make_backend", "announce", "remote_disabled", "BACKENDS",
]

BACKENDS = ("colabfold", "nvidia")

_announced: Set[str] = set()
_lock = threading.Lock()


def remote_disabled() -> bool:
    """SCIGANTIC_MSA_NO_REMOTE=1 turns every remote backend off, whatever the
    code or config asks for."""
    return os.environ.get("SCIGANTIC_MSA_NO_REMOTE", "").strip().lower() in ("1", "true", "yes")


def make_backend(name: str, **kw: Any):
    if name == "colabfold":
        return ColabFoldClient(**kw)
    if name == "nvidia":
        return NvidiaClient(**kw)
    raise ValueError(f"unknown remote backend {name!r}; choose one of {', '.join(BACKENDS)}")


def announce(backend: Any, stream: Optional[Any] = None) -> None:
    """Say once per process which server the sequences are going to."""
    key = f"{backend.name}:{backend.server_name}"
    with _lock:
        if key in _announced:
            return
        _announced.add(key)
    msg = f"scigantic-msa: sending the sequence to {backend.server_name} ({backend.name} backend)."
    if backend.name == "colabfold" and str(getattr(backend, "host", DEFAULT_HOST)).rstrip("/") == DEFAULT_HOST:
        msg += " Results on that public server can be read by anyone who knows the sequence."
    if getattr(backend, "experimental", False):
        msg += " This backend is experimental."
    print(msg, file=stream or sys.stderr, flush=True)
