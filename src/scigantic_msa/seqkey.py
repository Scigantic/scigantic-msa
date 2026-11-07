"""The canonical form of a sequence, used for cache keys and file names.

Canonical form: remove every character for which ``str.isspace()`` is true,
uppercase, then strip one trailing ``*``. Everything else is kept as is.
``seq_sha256`` is the lowercase hex sha256 of the UTF-8 bytes of that form.

A sequence that is empty, or has non-ASCII characters once whitespace is
removed, is rejected, so that every language uppercases it the same way.
"""
from __future__ import annotations

import hashlib

__all__ = ["canonical_sequence", "seq_sha256", "InvalidSequence"]


class InvalidSequence(ValueError):
    """The input cannot be turned into a canonical sequence."""


def canonical_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        raise InvalidSequence(f"sequence must be a str, got {type(seq).__name__}")
    s = "".join(ch for ch in seq if not ch.isspace())
    if not s.isascii():
        bad = sorted({ch for ch in s if not ch.isascii()})
        raise InvalidSequence(f"sequence has non-ASCII characters: {''.join(bad)!r}")
    s = s.upper()
    if s.endswith("*"):
        s = s[:-1]
    if not s:
        raise InvalidSequence("sequence is empty")
    return s


def seq_sha256(seq: str) -> str:
    return hashlib.sha256(canonical_sequence(seq).encode("utf-8")).hexdigest()
