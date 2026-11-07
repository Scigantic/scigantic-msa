"""scigantic-msa: protein MSAs from published sets or a search
server, written in the input format each structure predictor reads.

    from scigantic_msa import get_msa
    msa = get_msa("MKTAYIAKQRQISFVKSHFSRQ")
    print(msa.source, msa.depth)
    msa.to_af3_json("input.json")
"""

__version__ = "0.1.0"

from .seqkey import InvalidSequence, canonical_sequence, seq_sha256  # noqa: E402
from .a3m import A3MError, Row, parse_a3m, parse_stockholm  # noqa: E402
from .msa import MSA, SOURCES  # noqa: E402
from .index import Index, IndexIntegrityError, IndexUnavailable, DEFAULT_INDEX_URL  # noqa: E402
from .core import Complex, MSANotFound, get_msa, get_msas, where  # noqa: E402
from .remote import ColabFoldClient, NvidiaClient, RemoteError  # noqa: E402

__all__ = [
    "__version__",
    "get_msa", "get_msas", "where", "MSA", "Complex", "SOURCES", "MSANotFound",
    "seq_sha256", "canonical_sequence", "InvalidSequence",
    "parse_a3m", "parse_stockholm", "Row", "A3MError",
    "Index", "IndexIntegrityError", "IndexUnavailable", "DEFAULT_INDEX_URL",
    "ColabFoldClient", "NvidiaClient", "RemoteError",
]
