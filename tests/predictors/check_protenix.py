"""Load a scigantic-msa Protenix input JSON with Protenix's own code and run
its data pipeline up to the MSA features (no weights, no inference).

    python check_protenix.py JOB.json [JOB.json ...]

Protenix needs the PDB Chemical Component Dictionary. When PROTENIX_ROOT_DIR
(or a new temporary directory, when it is not set) has no
common/components.cif, a small one with the standard amino acids is built
there from the copy biotite ships, so nothing is downloaded.

Prints one JSON object per job: the files Protenix read with the rows it
kept from each, its depths, the species of every paired row in each chain,
and whether the first MSA row is each chain's sequence.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

AMINO = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
         "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", "MSE", "SEC", "PYL"]


def mini_ccd(root: Path) -> None:
    import numpy as np
    from biotite.structure.info.ccd import get_ccd
    from biotite.structure.io import pdbx

    common = root / "common"
    common.mkdir(parents=True, exist_ok=True)
    ccd, blocks = get_ccd(), {}
    for code in AMINO:
        cats = {}
        for cname, key in (("chem_comp", "id"), ("chem_comp_atom", "comp_id"), ("chem_comp_bond", "comp_id")):
            cat = ccd[cname]
            sel = np.flatnonzero(cat[key].as_array(str) == code)
            cols = {}
            for k in cat.keys():
                vals = cat[k].as_array(str)[sel]
                if cat[k].mask is not None:
                    m = cat[k].mask.array[sel]
                    vals = np.where(m == 1, ".", np.where(m == 2, "?", vals))
                cols[k] = vals.tolist()
            cats[cname] = pdbx.CIFCategory(cols)
        blocks[code] = pdbx.CIFBlock(cats)
    pdbx.CIFFile(blocks).write(str(common / "components.cif"))
    from scripts.gen_ccd_cache import precompute_ccd_mol

    precompute_ccd_mol(common / "components.cif", common / "components.cif.rdkit_mol.pkl")


def main(paths) -> None:
    root = os.environ.get("PROTENIX_ROOT_DIR") or tempfile.mkdtemp(prefix="protenix-ccd-")
    if not (Path(root) / "common" / "components.cif").is_file():
        mini_ccd(Path(root))
    os.environ["PROTENIX_ROOT_DIR"] = root

    import numpy as np
    from protenix.data.constants import PROTEIN_CHAIN
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.msa import msa_featurizer
    from protenix.data.msa.msa_utils import MSACore, MSAPairingEngine, RawMsa
    from runner import msa_search

    seen = {}
    original = MSAPairingEngine.pair_chains_by_species

    def recording(chains, *args, **kwargs):
        out = original(chains, *args, **kwargs)
        species = []
        for c, o in zip(chains, out):
            # Map each paired row back to the file row it came from; None is
            # Protenix's gap row for a chain with no row of that species.
            # A row and its deletions; the residues alone can repeat the query.
            def keys(f):
                return [m.tobytes() + d.tobytes() for m, d in zip(f["msa_all_seq"], f["deletion_matrix_all_seq"])]

            rows = {}
            for i, k in enumerate(keys(c)):
                rows.setdefault(k, i)
            ids = list(c["msa_species_identifiers_all_seq"]) or [""]
            species.append([ids[rows[k]] if k in rows else None for k in keys(o)])
        seen["paired_species"] = species
        return out

    MSAPairingEngine.pair_chains_by_species = staticmethod(recording)

    def rows_read(path, seq, dedup):
        text = Path(path).read_text()
        return [len(MSACore.parse_fasta(text)[0]), RawMsa.from_a3m(seq, PROTEIN_CHAIN, text, dedup=dedup).depth]

    jobs = [job for path in paths for job in json.loads(Path(path).read_text())]
    for job in jobs:
        seen.clear()
        if msa_search.need_msa_search(job):
            raise SystemExit("Protenix would start an MSA search for this job")
        feats, atom_array, _ = SampleDictToFeatures(job).get_feature_dict()
        msa = msa_featurizer.InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=job["sequences"], atom_array=atom_array, msa_pair_as_unpair=True)
        chains = []
        col = 0
        for ent in job["sequences"]:
            pc = ent["proteinChain"]
            seq = pc["sequence"]
            want, _ = MSACore.sequences_to_array([seq], PROTEIN_CHAIN)
            for _ in range(pc["count"]):
                got = msa["msa"][0, col:col + len(seq)]
                chains.append({
                    "length": len(seq),
                    "query_row_matches": bool(np.array_equal(got, want[0].astype(got.dtype))),
                    # Rows in each file, and the rows Protenix keeps from it.
                    "files": {Path(pc[k]).name: rows_read(pc[k], seq, dedup) for k, dedup in
                              (("pairedMsaPath", False), ("unpairedMsaPath", True)) if k in pc},
                })
                col += len(seq)
        print(json.dumps({
            "name": job["name"],
            "tokens": int(feats["residue_index"].shape[0]),
            "msa_shape": list(msa["msa"].shape),
            "paired_depth": int(msa["prot_paired_num_alignments"]),
            "unpaired_depth": int(msa["prot_unpaired_num_alignments"]),
            "chains": chains,
            "paired_species": seen.get("paired_species"),
        }))


if __name__ == "__main__":
    main(sys.argv[1:])
