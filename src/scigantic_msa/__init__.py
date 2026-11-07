"""scigantic-msa: one command from a protein sequence to the MSA input files
a structure predictor reads.

    from scigantic_msa import MSA
    from scigantic_msa.adapters import AlphaFold3, Boltz

    result = MSA().get(sequence, remote=True)   # sends the sequence to api.colabfold.com
    af3_input = AlphaFold3.prepare(result)      # dict ready for json.dump
"""

__version__ = "0.2.2"

from .seqkey import InvalidSequence, canonical_sequence, seq_sha256  # noqa: E402
from .a3m import A3MError, Row, parse_a3m, parse_stockholm  # noqa: E402
from .alignment import Alignment, SOURCES  # noqa: E402
from .remote import ColabFoldClient, NvidiaClient, RemoteError  # noqa: E402
from .tool import MSA, RemoteNotAllowed  # noqa: E402
from .write import MODELS, write_model

__all__ = [
    "write_model",
    "MODELS",
    "__version__",
    "MSA", "Alignment", "SOURCES", "RemoteNotAllowed", "RemoteError",
    "seq_sha256", "canonical_sequence", "InvalidSequence",
    "parse_a3m", "parse_stockholm", "Row", "A3MError",
    "ColabFoldClient", "NvidiaClient",
]
