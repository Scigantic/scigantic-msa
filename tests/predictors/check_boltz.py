"""Load scigantic-msa output with Boltz's own MSA parsers.

Run with an interpreter that has boltz installed:
    python check_boltz.py FILE [FILE ...]
Prints one JSON object per file: number of sequences, residues per row,
and the taxonomy keys Boltz read (for pairing)."""
import json
import sys
from pathlib import Path

from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv

for arg in sys.argv[1:]:
    p = Path(arg)
    if p.suffix not in (".a3m", ".csv"):
        # Boltz main.py refuses anything else.
        raise SystemExit(f"boltz would refuse suffix {p.suffix}")
    msa = parse_a3m(p, taxonomy=None) if p.suffix == ".a3m" else parse_csv(p)
    seqs = msa.sequences
    widths = sorted({int(s["res_end"] - s["res_start"]) for s in seqs})
    taxa = [int(s["taxonomy"]) for s in seqs]
    print(json.dumps({"file": p.name, "n": int(len(seqs)), "widths": widths, "taxonomy": taxa}))
