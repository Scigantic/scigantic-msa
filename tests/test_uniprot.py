"""A UniProt alignment per chain (convert --uniprot, from_a3m(uniprot=...))
reaches the models that pair on UniProt headers."""
import json

import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, REAL
from scigantic_msa import MSA
from scigantic_msa.a3m import A3MError, parse_a3m
from scigantic_msa.adapters import AlphaFold3, protenix_species
from scigantic_msa.cli import main

A = REAL / "of3-7m22_A"
C = REAL / "of3-7m22_C"


def _convert(model, out, *extra):
    return main(["convert", str(A / "cfdb_uniref30.a3m"), str(C / "cfdb_uniref30.a3m"),
                 "--uniprot", str(A / "uniprot_hits.sto"), "--uniprot", str(C / "uniprot_hits.sto"),
                 "--model", model, "--out", str(out), "--quiet", *extra])


def test_from_a3m_reads_a_uniprot_alignment():
    m = MSA().from_a3m(A / "cfdb_uniref30.a3m", uniprot=A / "uniprot_hits.sto")
    assert m.uniprot_rows and len(m.uniprot_rows) > 1 and m.uniprot_rows[0].seq == Q_7M22A
    assert all(r.db == "uniprot" for r in m.uniprot_rows[1:])
    assert m.provenance["uniprot_file"] == "uniprot_hits.sto"
    assert MSA().from_a3m(A / "cfdb_uniref30.a3m").uniprot_rows is None


def test_convert_af3_puts_uniprot_rows_in_paired_msa(tmp_path):
    assert _convert("af3", tmp_path) == 0
    doc = json.loads(next(tmp_path.glob("af3/*.json")).read_text())
    prots = [s["protein"] for s in doc["sequences"]]
    assert [p["sequence"] for p in prots] == [Q_7M22A, Q_7M22C]
    for p in prots:
        rows = parse_a3m(p["pairedMsa"], p["sequence"])
        assert len(rows) > 1 and all(r.header.startswith(("tr|", "sp|")) for r in rows[1:])


def test_convert_openfold3_writes_uniprot_hits(tmp_path):
    assert _convert("openfold3", tmp_path) == 0
    hits = sorted(tmp_path.glob("**/uniprot_hits.a3m"))
    assert len(hits) == 2
    for h in hits:
        assert parse_a3m(h.read_text())[1].header.startswith(("tr|", "sp|"))


def test_convert_protenix_pairs_from_uniprot_rows(tmp_path):
    assert _convert("protenix", tmp_path) == 0
    paired = sorted(tmp_path.glob("**/pairing.a3m"))
    assert len(paired) == 2
    species = [{protenix_species(r.header) for r in parse_a3m(p.read_text())[1:]} for p in paired]
    assert all(species) and species[0] & species[1]


def test_convert_without_uniprot_has_no_paired_msa(tmp_path):
    assert main(["convert", str(A / "cfdb_uniref30.a3m"), str(C / "cfdb_uniref30.a3m"),
                 "--model", "af3", "--out", str(tmp_path), "--quiet"]) == 0
    doc = json.loads(next(tmp_path.glob("af3/*.json")).read_text())
    assert all(s["protein"]["pairedMsa"] == "" for s in doc["sequences"])


def test_uniprot_count_must_match_the_chains(tmp_path, capsys):
    rc = main(["convert", str(A / "cfdb_uniref30.a3m"), str(C / "cfdb_uniref30.a3m"),
               "--uniprot", str(A / "uniprot_hits.sto"), "--model", "af3", "--out", str(tmp_path)])
    assert rc == 1 and "one per chain" in capsys.readouterr().err


def test_uniprot_file_for_the_wrong_chain_is_refused(tmp_path, capsys):
    rc = main(["convert", str(A / "cfdb_uniref30.a3m"), "--uniprot", str(C / "uniprot_hits.sto"),
               "--model", "af3", "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1 and "UniProt alignment for cfdb_uniref30.a3m" in err


def test_uniprot_file_without_uniprot_headers_is_refused(tmp_path):
    f = tmp_path / "u.a3m"
    f.write_text(f">q\n{Q_6T33}\n>UniRef100_A0A123 TaxID=9606\n{Q_6T33}\n")
    with pytest.raises(A3MError, match="no row with a UniProt header"):
        MSA().from_a3m(REAL / "of3-6t33_A" / "cfdb_uniref30.a3m", uniprot=f)


def test_af3_pairs_on_uniprot_rows_from_files():
    a = MSA().from_a3m(A / "cfdb_uniref30.a3m", uniprot=A / "uniprot_hits.sto")
    c = MSA().from_a3m(C / "cfdb_uniref30.a3m", uniprot=C / "uniprot_hits.sto")
    doc = AlphaFold3.prepare([a, c], quiet=True)
    assert all(s["protein"]["pairedMsa"] for s in doc["sequences"])


def test_every_row_of_the_uniprot_file_is_marked_uniprot(tmp_path):
    # Chai-1 reads source_database from this; a row without a tr| header is still a UniProt hit.
    q = Q_6T33
    f = tmp_path / "u.a3m"
    f.write_text(f">q\n{q}\n>tr|A0A123B4C5|A0A123B4C5_HUMAN\n{q[:-1]}W\n>hit2 OX=10090\n{q[:-1]}Y\n")
    m = MSA().from_a3m(REAL / "of3-6t33_A" / "cfdb_uniref30.a3m", uniprot=f)
    assert [r.db for r in m.uniprot_rows[1:]] == ["uniprot", "uniprot"]
