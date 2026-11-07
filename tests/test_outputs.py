"""Model outputs for complexes and the printed next-step commands."""
from pathlib import Path

from scigantic_msa.a3m import Row
from scigantic_msa.adapters import Protenix, protenix_species
from scigantic_msa.alignment import Alignment


def synthetic(query, hits, taxids=None):
    rows = [Row("101", query, "query")]
    for i, s in enumerate(hits):
        rows.append(Row(f"UniRef100_H{i}", s, "uniref", (taxids or {}).get(i)))
    return Alignment(query=query, rows=rows, source="remote", provenance={"backend": "colabfold"})


def _paired_species(pc):
    p = pc.get("pairedMsaPath")
    if not p:
        return None
    text = Path(p).read_text()
    return {protenix_species(h[1:]) for h in text.splitlines()[2::2]}


def test_protenix_pairs_two_chains_when_a_third_has_no_species(tmp_path):
    # Protenix pairs any species found in two or more chains. A chain with no
    # species rows (a designed binder, say) must not cost the other two
    # their pairing.
    a = synthetic("MKVLAGWERT", ["MRVLAGWERT", "MKILAGWERT", "MKVAAGWERT"], {0: "9606", 1: "10090", 2: "7227"})
    b = synthetic("GHSEQKLLPA", ["GHTEQKLLPA", "GASEQKLLPA"], {0: "10090", 1: "9606"})
    c = synthetic("PPWWGGYYST", ["PPWWGGYYSA"])
    jobs = Protenix.prepare([a, b, c], out_dir=tmp_path, quiet=True)
    pcs = [s["proteinChain"] for s in jobs[0]["sequences"]]
    assert _paired_species(pcs[0]) >= {"9606", "10090"}
    assert _paired_species(pcs[1]) == {"9606", "10090"}
    assert "pairedMsaPath" not in pcs[2]


def test_protenix_pairs_species_shared_by_some_chains_only(tmp_path):
    # A and B share 9606, B and C share 10090; no species is in all three.
    a = synthetic("MKVLAGWERT", ["MRVLAGWERT"], {0: "9606"})
    b = synthetic("GHSEQKLLPA", ["GHTEQKLLPA", "GASEQKLLPA"], {0: "9606", 1: "10090"})
    c = synthetic("PPWWGGYYST", ["PPWWGGYYSA"], {0: "10090"})
    jobs = Protenix.prepare([a, b, c], out_dir=tmp_path, quiet=True)
    pcs = [s["proteinChain"] for s in jobs[0]["sequences"]]
    assert [_paired_species(pc) for pc in pcs] == [{"9606"}, {"9606", "10090"}, {"10090"}]


def test_protenix_dimer_with_no_shared_species_gets_no_pairing_file(tmp_path):
    a = synthetic("MKVLAGWERT", ["MRVLAGWERT"], {0: "9606"})
    b = synthetic("GHSEQKLLPA", ["GHTEQKLLPA"], {0: "10090"})
    jobs = Protenix.prepare([a, b], out_dir=tmp_path, quiet=True)
    assert all("pairedMsaPath" not in s["proteinChain"] for s in jobs[0]["sequences"])


def test_next_command_survives_a_shell_with_spaces_and_quotes_in_out(tmp_path):
    import shlex

    from scigantic_msa import write_model

    a = synthetic("MKVLAGWERT", ["MRVLAGWERT"], {0: "9606"})
    b = synthetic("GHSEQKLLPA", ["GHTEQKLLPA"], {0: "9606"})
    out = tmp_path / "my run (v2)" / "o'k"
    for model in ("af3", "boltz", "colabfold", "openfold3", "protenix"):
        files, nxt = write_model(model, [a, b], out, name="job", quiet=True)
        words = shlex.split(nxt)
        # The input file the command names is one word that exists.
        assert any(Path(w.split("=", 1)[-1]) == Path(files[0]) for w in words), (model, nxt)
        assert Path(files[0]).is_file()
