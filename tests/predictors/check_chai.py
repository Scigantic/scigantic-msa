"""Load scigantic-msa .aligned.pqt files with Chai-1's own loader and pair
them with Chai's own pairing code.

    python check_chai.py DIR SEQ [SEQ ...]
"""
import json
import sys
from pathlib import Path

from chai_lab.data.dataset.msas import load
from chai_lab.data.parsing.msas.aligned_pqt import expected_basename, parse_aligned_pqt_to_msa_context

d = Path(sys.argv[1])
ctxs = []
out = {"files": []}
for seq in sys.argv[2:]:
    path = d / expected_basename(seq)
    if not path.is_file():
        raise SystemExit(f"chai would not find {path.name} for this sequence")
    ctx = parse_aligned_pqt_to_msa_context(path)
    ctxs.append(ctx)
    out["files"].append({"file": path.name, "depth": int(ctx.depth), "length": int(ctx.tokens.shape[1])})
if len(ctxs) > 1:
    merged = load.pair_and_merge_msas(ctxs)
    out["merged_depth"] = int(merged.depth)
    out["paired_rows"] = int(merged.is_paired_mask.any(dim=-1).sum()) if hasattr(merged, "is_paired_mask") else None
print(json.dumps(out))
