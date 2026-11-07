"""Fetch the small real alignment files used as test fixtures.

Run once by hand; the files are committed under tests/fixtures/real/ so the
test suite never touches the network. Every file is public under CC BY 4.0
(OpenProteinSet, OpenFold3 data, LiteFold human proteome MSAs). Sizes and
sha256 go into tests/fixtures/real/MANIFEST.json.

tests/fixtures/colabfold-live1.tar.gz is one real result from the public
ColabFold server for the public 39-residue example sequence
SGSMKTAISLPDETFDRVSRRASELGMSRSEFFTKAAQR (mode env-taxonomy).

    python tests/fixtures/fetch_real.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "real"
OPS = "https://openfold.s3.amazonaws.com/"
OF3 = "https://openfold3-data.s3.us-west-2.amazonaws.com/msas/pdb_msas_completed/"
HF = ("https://huggingface.co/datasets/LiteFold/human-proteome-wide-msa/resolve/"
      "632cfc87f567e429570897c3addd80665a697f3e/")

FILES = {
    "ops-6t33_A/uniref90_hits.a3m": OPS + "pdb/6t33_A/a3m/uniref90_hits.a3m",
    "ops-6t33_A/mgnify_hits.a3m": OPS + "pdb/6t33_A/a3m/mgnify_hits.a3m",
    "ops-6t33_A/bfd_uniclust_hits.a3m": OPS + "pdb/6t33_A/a3m/bfd_uniclust_hits.a3m",
    "ops-6t33_A/pdb70_hits.hhr": OPS + "pdb/6t33_A/hhr/pdb70_hits.hhr",
    "of3-6t33_A/cfdb_uniref30.a3m": OF3 + "6t33_A/cfdb_uniref30.a3m",
    "of3-6t33_A/uniref90_hits.sto": OF3 + "6t33_A/uniref90_hits.sto",
    "of3-6t33_A/mgnify_hits.sto": OF3 + "6t33_A/mgnify_hits.sto",
    "of3-6t33_A/uniprot_hits.sto": OF3 + "6t33_A/uniprot_hits.sto",
    "of3-1hvw_A/cfdb_uniref30.a3m": OF3 + "1hvw_A/cfdb_uniref30.a3m",
    "of3-1hvw_A/uniprot_hits.sto": OF3 + "1hvw_A/uniprot_hits.sto",
    "of3-1pxq_A/cfdb_uniref30.a3m": OF3 + "1pxq_A/cfdb_uniref30.a3m",
    "of3-1pxq_A/uniprot_hits.sto": OF3 + "1pxq_A/uniprot_hits.sto",
    # 7m22 chains A and C: a small real heterodimer whose UniProt hits share species.
    "of3-7m22_A/cfdb_uniref30.a3m": OF3 + "7m22_A/cfdb_uniref30.a3m",
    "of3-7m22_A/uniprot_hits.sto": OF3 + "7m22_A/uniprot_hits.sto",
    "of3-7m22_C/cfdb_uniref30.a3m": OF3 + "7m22_C/cfdb_uniref30.a3m",
    "of3-7m22_C/uniprot_hits.sto": OF3 + "7m22_C/uniprot_hits.sto",
    "litefold-A0A096LP01/A0A096LP01.a3m": HF + "a3m/shard-000/A0A096LP01.a3m",
    "litefold-A0A096LP01/A0A096LP01.json": HF + "af3_json/shard-000/A0A096LP01.json",
    "uniclust30-A0A1A9ZI35/uniclust30.a3m": OPS + "uniclust30/A0A1A9ZI35/a3m/uniclust30.a3m",
    "uniclust30-M4VQ73/uniclust30.a3m": OPS + "uniclust30/M4VQ73/a3m/uniclust30.a3m",
}


def main() -> int:
    manifest = {}
    for rel, url in FILES.items():
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "scigantic-msa-tests"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        if not body:
            print(f"empty body for {url}", file=sys.stderr)
            return 1
        dest.write_bytes(body)
        manifest[rel] = {"url": url, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        print(f"{len(body):>8}  {rel}")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"total {sum(v['bytes'] for v in manifest.values())} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
