"""Remote MSA search backends.

``MSA().get()`` uses one only when you pass remote=True. Calling a client's
``search()`` yourself is that choice made in code, so it needs no
remote=True. SCIGANTIC_MSA_NO_REMOTE wins over both: every client checks it
before each request, and so does the HTTP layer underneath.
"""
from __future__ import annotations

import sys
import threading
from typing import Any, Optional, Set

from ..http import remote_disabled
from .colabfold import (DEFAULT_HOST, ColabFoldClient, JobFailed, PairResult, RateLimited, RemoteError, RemoteResult,
                        public_host, replay_dir)
from .nvidia import NvidiaClient

__all__ = [
    "ColabFoldClient", "NvidiaClient", "RemoteError", "RateLimited", "JobFailed", "RemoteResult",
    "make_backend", "announce", "remote_disabled", "BACKENDS", "PairResult",
]

BACKENDS = ("colabfold", "nvidia")

_announced: Set[str] = set()
_lock = threading.Lock()


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
    if backend.name == "colabfold" and replay_dir():
        print(f"scigantic-msa: reading saved ColabFold results from {replay_dir()} (test only); nothing is sent.",
              file=stream or sys.stderr, flush=True)
        return
    msg = f"scigantic-msa: sending the sequence to {backend.server_name} ({backend.name} backend)."
    if backend.name == "colabfold" and public_host(str(getattr(backend, "host", DEFAULT_HOST))):
        msg += " Results on that public server can be read by anyone who knows the sequence."
    if getattr(backend, "experimental", False):
        msg += " This backend is experimental."
    print(msg, file=stream or sys.stderr, flush=True)
