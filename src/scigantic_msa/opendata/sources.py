"""The open alignment sets this package reads, where they live, and the terms
that come with them.

Every set is read where its publisher put it, with anonymous HTTPS: two
public S3 buckets of the OpenFold Consortium and one Hugging Face dataset.
Nothing is mirrored and no index is hosted by anyone else. To find the
alignment of a sequence, each set's own published table of sequences is
downloaded once and kept in the local cache (see :mod:`.tables`).

``BASES`` holds the host of each set. Tests point it at a local server.
"""
from __future__ import annotations

from typing import Dict, List

#: Base URL of each host, without a trailing slash.
BASES: Dict[str, str] = {
    "of3": "https://openfold3-data.s3.us-west-2.amazonaws.com",
    "ops": "https://openfold.s3.amazonaws.com",
    "hf": "https://huggingface.co",
    "uniprot": "https://rest.uniprot.org",
}

#: The LiteFold dataset is read at one commit, so its table and its files
#: always belong together.
LITEFOLD_REPO = "LiteFold/human-proteome-wide-msa"
LITEFOLD_COMMIT = "632cfc87f567e429570897c3addd80665a697f3e"

CC_BY = "CC-BY-4.0"

#: One entry per alignment set. ``roles`` are the files read for a matched
#: entry, in the order they are read; ``upstream`` the sequence databases
#: the alignments were searched against, whose terms a derived MSA also
#: carries.
SOURCES: Dict[str, Dict] = {
    "openfold3": {
        "title": "OpenFold3 training MSAs for PDB chains",
        "publisher": "OpenFold Consortium",
        "where": "s3://openfold3-data/msas/pdb_msas_completed/ (us-west-2)",
        "license": CC_BY,
        "license_url": "https://openfold3-data.s3.us-west-2.amazonaws.com/LICENSE",
        "citation": "OpenFold3, OpenFold Consortium, https://github.com/aqlaboratory/openfold-3",
        "attribution": "OpenFold3 training MSAs by the OpenFold Consortium, CC BY 4.0",
        "databases": "UniProtKB, UniRef90, MGnify (jackhmmer); UniRef30 + ColabFold envdb (HHblits); 2025 releases",
        "upstream": ["UniProt", "ColabFold databases", "MGnify"],
        "default": True,
    },
    "litefold": {
        "title": "LiteFold human proteome ColabFold MSAs",
        "publisher": "LiteFold",
        "where": f"huggingface.co/datasets/{LITEFOLD_REPO} at commit {LITEFOLD_COMMIT}",
        "license": CC_BY,
        "license_url": f"https://huggingface.co/datasets/{LITEFOLD_REPO}",
        "citation": f"LiteFold, human proteome-wide MSAs, https://huggingface.co/datasets/{LITEFOLD_REPO}",
        "attribution": "Human proteome-wide MSAs by LiteFold, CC BY 4.0 per the dataset card",
        "databases": "UniRef30 2302 + ColabFold envdb 202108 (MMseqs2)",
        "upstream": ["ColabFold databases"],
        "default": True,
    },
    "openproteinset": {
        "title": "OpenProteinSet PDB chain MSAs",
        "publisher": "OpenFold Consortium",
        "where": "s3://openfold/alignment_db/ (us-east-1)",
        "license": CC_BY,
        "license_url": "https://openfold.s3.amazonaws.com/LICENSE",
        "citation": "Ahdritz et al., OpenProteinSet: training data for structural biology at scale, "
                    "NeurIPS 2023, arXiv:2308.05326",
        "attribution": "OpenProteinSet by Ahdritz et al. (OpenFold Consortium), CC BY 4.0",
        "databases": "UniRef90, MGnify (jackhmmer); BFD + UniClust30 (HHblits); 2020 to 2021 releases",
        "upstream": ["UniProt", "BFD", "UniClust30", "MGnify"],
        "default": False,
    },
    "uniclust30": {
        "title": "OpenProteinSet UniClust30 representative MSAs",
        "publisher": "OpenFold Consortium",
        "where": "s3://openfold/uniclust30/ (us-east-1)",
        "license": CC_BY,
        "license_url": "https://openfold.s3.amazonaws.com/LICENSE",
        "citation": "Ahdritz et al., OpenProteinSet, arXiv:2308.05326",
        "attribution": "OpenProteinSet UniClust30 MSAs by Ahdritz et al. (OpenFold Consortium), CC BY 4.0",
        "databases": "UniClust30 (HHblits)",
        "upstream": ["UniClust30"],
        "default": True,
    },
}

#: What a file of each source holds, for the Chai-1 source column and the
#: provenance table.
ROLE_DB: Dict[str, str] = {
    "uniprot": "uniprot",
    "uniref90": "uniref90",
    "mgnify": "mgnify",
    "cfdb_uniref30": "bfd_uniclust",
    "colabfold": "uniref",
    "bfd_uniclust": "bfd_uniclust",
    "uniclust30": "uniclust30",
}

#: Terms of the databases that were searched. Checked on the databases'
#: own pages on 2026-11-08, except where ``checked`` is False.
UPSTREAM: Dict[str, Dict] = {
    "UniProt": {"terms": "CC BY 4.0", "url": "https://www.uniprot.org/help/license", "checked": True},
    "ColabFold databases": {"terms": "CC BY 4.0", "url": "https://colabfold.mmseqs.com/", "checked": True},
    "BFD": {"terms": "CC BY 4.0", "url": "https://bfd.mmseqs.com/", "checked": True},
    "MGnify": {"terms": "CC0 1.0", "url": "https://www.ebi.ac.uk/metagenomics/", "checked": True},
    "UniClust30": {"terms": "not confirmed; the Uniclust authors are reported to publish it as CC BY-SA 4.0, "
                            "which would make derived alignments share-alike",
                   "url": "https://uniclust.mmseqs.com/", "checked": False},
}

#: The ColabFold server, which is not an open data set but whose rows the
#: default run merges.
COLABFOLD = {
    "id": "colabfold-api",
    "title": "ColabFold MMseqs2 server",
    "license": "see the server's terms; its databases (UniRef30, ColabFold envdb) are CC BY 4.0",
    "citation": "Mirdita et al., ColabFold: making protein folding accessible to all, Nature Methods 19, "
                "679-682 (2022)",
}


def licence_flags(source_ids: List[str]) -> List[str]:
    """Notes for someone who redistributes the output: upstream databases
    whose terms were not confirmed or are share-alike."""
    out = []
    for sid in sorted(set(source_ids)):
        for up in SOURCES.get(sid, {}).get("upstream", []):
            t = UPSTREAM[up]
            if not t["checked"] or "SA" in t["terms"]:
                out.append(f"{sid}: rows were searched against {up}, whose terms are {t['terms']} ({t['url']})")
    return out
