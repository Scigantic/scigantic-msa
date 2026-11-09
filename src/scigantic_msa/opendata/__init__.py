"""Open data: published alignments read where their publishers put them.

See :mod:`.lookup` for what is read and what is sent, :mod:`.merge` for how
rows are merged and complexes paired, and :mod:`.sources` for the sets and
their terms.
"""
from .lookup import OpenData, OpenResult, Prov
from .sources import SOURCES

__all__ = ["OpenData", "OpenResult", "Prov", "SOURCES"]
