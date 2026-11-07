import pytest

from conftest import Q_1HVW, Q_6T33, Q_LITEFOLD, REAL, real
from scigantic_msa.a3m import (
    A3MError, Row, a3m_query, format_a3m, match_length, parse_a3m, parse_colabfold_complex, parse_stockholm,
)


def test_real_openproteinset_files_parse():
    for name, n in (("uniref90_hits.a3m", 22), ("bfd_uniclust_hits.a3m", 21), ("mgnify_hits.a3m", 43)):
        rows = parse_a3m(real(f"ops-6t33_A/{name}"), Q_6T33)
        assert len(rows) == n
        assert rows[0].seq == Q_6T33 and rows[0].db == "query"
        assert all(match_length(r.seq) == len(Q_6T33) for r in rows)


def test_real_litefold_file_labels_uniref_and_env_rows():
    rows = parse_a3m(real("litefold-A0A096LP01/A0A096LP01.a3m"), Q_LITEFOLD)
    assert rows[0].header == "101"
    dbs = {r.db for r in rows[1:]}
    assert dbs <= {"uniref", "env", "mgnify"} and "uniref" in dbs and "env" in dbs
    # The env block repeats the >101 query record; it must be dropped.
    assert sum(1 for r in rows if r.header == "101") == 1


def test_uniclust_consensus_first_file_has_no_query():
    text = real("uniclust30-M4VQ73/uniclust30.a3m").decode()
    assert a3m_query(text, "M4VQ73") is None
    with pytest.raises(A3MError, match="not a usable query row"):
        parse_a3m(text)


def test_uniclust_query_first_file():
    text = real("uniclust30-A0A1A9ZI35/uniclust30.a3m").decode()
    q = a3m_query(text, "A0A1A9ZI35")
    rows = parse_a3m(text, q)
    assert rows[0].seq == q and len(rows) == 200
    assert sum(1 for r in rows if r.taxid) >= 190


def test_query_mismatch_is_an_error():
    with pytest.raises(A3MError, match="does not match the query"):
        parse_a3m(real("ops-6t33_A/uniref90_hits.a3m"), Q_1HVW)


def test_one_residue_query_mismatch_is_caught():
    mutant = Q_6T33[:10] + ("A" if Q_6T33[10] != "A" else "G") + Q_6T33[11:]
    with pytest.raises(A3MError, match="position 11"):
        parse_a3m(real("ops-6t33_A/uniref90_hits.a3m"), mutant)


@pytest.mark.parametrize("text,msg", [
    (">q\nMKV\n>h\nMK.V\n", "outside A-Z"),
    (">q\nMKV\n>h\nMK*\n", "outside A-Z"),
    (">q\nMKV\n>h\nMK1\n", "outside A-Z"),
    (">q\nMKV\n>h\nMKVV\n", "match columns"),
    (">q\nMKV\n>h\nMK\n", "match columns"),
    (">q\nMkV\n", "not a usable query"),
    (">q\nM-V\n", "not a usable query"),
    ("MKV\n", "before any"),
    ("", "no records"),
    (">q\n\n>h\nMKV\n", "has no sequence"),
    ("#4,3\t1,1\n>101\t102\nMKTAGLV\n", "complex header"),
    ("#comment\n>q\nMKV\n", "not a ColabFold header"),
    ("#4\t1\n>q\nMKV\n", "header says length 4"),
])
def test_bad_alignments(text, msg):
    with pytest.raises(A3MError, match=msg):
        parse_a3m(text)


def test_insertions_do_not_count_as_columns():
    rows = parse_a3m(">q\nMKV\n>h\nMkkK-\n")
    assert rows[1].seq == "MkkK-"


def test_wrapped_lines_nul_and_crlf():
    text = ">q\r\nMK\r\nV\r\n>h\nM\n-V\n\x00"
    rows = parse_a3m(text, "MKV")
    assert [r.seq for r in rows] == ["MKV", "M-V"]


def test_all_gap_rows_are_dropped():
    rows = parse_a3m(">q\nMKV\n>gaps\n---\n>ins\n-a--\n>ok\nM--\n")
    assert [r.header for r in rows] == ["q", "ok"]


def test_single_chain_colabfold_header_is_accepted():
    rows = parse_a3m("#3\t1\n>101\nMKV\n>h\nMRV\n", "MKV")
    assert len(rows) == 2


def test_skip_comments_mode():
    rows = parse_a3m("#A0A acc line\n>A_consensus\nMKV\n>q\nMKV\n>h\nMRV\n", skip_comments=True)
    assert rows[0].header == "q" and len(rows) == 2


def test_taxid_from_headers():
    rows = parse_a3m(">q\nMKV\n>a TaxID=9606 x\nMKV\n>b OS=Homo OX=10090\nMRV\n>c\nMRA\n")
    assert [r.taxid for r in rows] == [None, "9606", "10090", None]


def test_stockholm_real_file_converts_to_a3m():
    rows = parse_stockholm(real("of3-6t33_A/uniref90_hits.sto").split(b"\n"), Q_6T33)
    assert rows[0].seq == Q_6T33
    assert len(rows) == 20
    for r in rows:
        assert match_length(r.seq) == len(Q_6T33)
    assert rows[1].header.startswith("UniRef90_")
    # DE text from the #=GS lines is appended to the header, so taxids are found.
    assert rows[1].taxid is not None


def test_stockholm_insert_columns_become_lowercase():
    sto = "# STOCKHOLM 1.0\n\nq   MK-V\nh1  MKAV\nh2  M--V\n//\n"
    rows = parse_stockholm(sto.splitlines(), "MKV")
    assert [r.seq for r in rows] == ["MKV", "MKaV", "M-V"]


def test_stockholm_interleaved_blocks():
    sto = "# STOCKHOLM 1.0\n\nq  MK\nh  MR\n\nq  V-\nh  Va\n//\n"
    rows = parse_stockholm(sto.splitlines(), "MKV")
    assert [r.seq for r in rows] == ["MKV", "MRVa"]


def test_stockholm_cut_inside_an_interleaved_block_fails_the_query_check():
    sto = "# STOCKHOLM 1.0\n\nq  MK\nh  MR\n"
    with pytest.raises(A3MError):
        parse_stockholm(sto.splitlines(), "MKV")


def test_stockholm_max_rows():
    rows = parse_stockholm(real("of3-6t33_A/uniprot_hits.sto").split(b"\n"), Q_6T33, max_rows=5)
    assert len(rows) == 5


def test_stockholm_not_stockholm():
    with pytest.raises(A3MError, match="STOCKHOLM"):
        parse_stockholm([">q", "MKV"])


def test_colabfold_complex_round_trip():
    text = "#4,3\t1,1\n>101\t102\nMKTAGLV\n>pA\tpB\nMK-AG-V\n>101\nMKTA---\n>hitA\nMRTA---\n>102\n----GLV\n"
    lens, cards, queries, records = parse_colabfold_complex(text)
    assert lens == [4, 3] and cards == [1, 1] and queries == ["MKTA", "GLV"]
    assert len(records) == 5


def test_format_round_trip_keeps_headers_verbatim():
    rows = parse_a3m(real("litefold-A0A096LP01/A0A096LP01.a3m"))
    again = parse_a3m(format_a3m(rows))
    assert [(r.header, r.seq) for r in rows] == [(r.header, r.seq) for r in again]
    assert "\t" in rows[1].header
