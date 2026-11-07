import json
from pathlib import Path

import pytest

from scigantic_msa.seqkey import InvalidSequence, canonical_sequence, seq_sha256

VECTORS = json.loads((Path(__file__).parent / "vectors" / "seq_sha256.json").read_text())


@pytest.mark.parametrize("v", VECTORS["accept"], ids=lambda v: repr(v["input"]))
def test_shared_vectors(v):
    assert canonical_sequence(v["input"]) == v["canonical"]
    assert seq_sha256(v["input"]) == v["seq_sha256"]


@pytest.mark.parametrize("raw", VECTORS["reject"])
def test_rejects(raw):
    with pytest.raises(InvalidSequence):
        seq_sha256(raw)


def test_fixed_vector_shared_with_backend():
    # The backend design note uses this exact input as its cross-language check.
    assert seq_sha256("mkt aYIAKQR*\n") == "0b2b6b8411d220f8648b67b5f839a4336470f4fc93d9604009a1cb3a093ae29e"


def test_only_one_trailing_star_is_stripped():
    assert canonical_sequence("MKV**") == "MKV*"
    assert canonical_sequence("MK*V") == "MK*V"


def test_unicode_whitespace_is_removed_before_the_ascii_check():
    assert canonical_sequence("M K V　") == "MKV"


def test_gaps_and_dots_are_kept():
    # The key keeps every non-whitespace character; it is not a residue filter.
    assert canonical_sequence("m-k.v") == "M-K.V"


def test_matches_chai_filename_hash():
    import hashlib

    s = "WGCVCSGSTAVANSHNAGPAYCVGYCGNNGVVTRNANANVAKTA"
    assert seq_sha256(s) == hashlib.sha256(s.upper().encode()).hexdigest()
