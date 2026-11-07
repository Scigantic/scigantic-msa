"""The only network traffic is to the MSA server the user chose.

Every outgoing connection is recorded at the socket layer, under urllib and
anything else, and any address other than the fake server fails the test.
"""
import re
import socket
from pathlib import Path

import pytest

from conftest import Q_6T33, Q_7M22A, Q_7M22C, REAL, ColabFoldStub, make_result_tar
from scigantic_msa import MSA
from scigantic_msa.adapters import AlphaFold3, Boltz, Chai, ColabFold, OpenFold3, Protenix
from scigantic_msa.cli import main

SRC = Path(__file__).resolve().parent.parent / "src" / "scigantic_msa"


@pytest.fixture
def connections(monkeypatch):
    """Record every connect() and refuse any that is not to the allowed port."""
    seen = []
    allowed = {}
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        seen.append(address)
        host, port = address[0], address[1]
        if (host, port) != (allowed.get("host"), allowed.get("port")):
            raise AssertionError(f"unexpected connection to {address}")
        return real_connect(self, address)

    def getaddrinfo(host, *a, **kw):
        seen.append(("dns", host))
        if host not in ("127.0.0.1",):
            raise AssertionError(f"unexpected name lookup for {host}")
        return real_getaddrinfo(host, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return seen, allowed


def test_a_full_run_talks_only_to_the_chosen_server(colabfold, connections, tmp_path, monkeypatch):
    seen, allowed = connections
    allowed.update(host="127.0.0.1", port=colabfold.port)
    ColabFoldStub(colabfold, {q: make_result_tar(q, [("UniRef100_X", q[:-1] + "A", "9606")])
                              for q in (Q_7M22A, Q_7M22C)})
    results = MSA(quiet=True).get([Q_7M22A, Q_7M22C], remote=True)
    AlphaFold3.prepare(results, quiet=True)
    Boltz.prepare(results, out_dir=tmp_path / "b", quiet=True)
    ColabFold.prepare(results, out_dir=tmp_path / "c", quiet=True)
    OpenFold3.prepare(results, out_dir=tmp_path / "o", quiet=True)
    Protenix.prepare(results, out_dir=tmp_path / "p", quiet=True)
    try:
        import pyarrow  # noqa: F401

        Chai.prepare(results, out_dir=tmp_path / "ch", quiet=True)
    except ImportError:
        pass
    # The CLI path as well.
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    assert main(["run", f"{Q_7M22A}:{Q_7M22C}", "-m", "protenix", "-o", str(tmp_path / "cli"), "--yes", "-q"]) == 0
    conns = [a for a in seen if a[0] != "dns"]
    assert conns, "the fake server was never contacted"
    assert {(a[0], a[1]) for a in conns} == {("127.0.0.1", colabfold.port)}
    # Every request was a ColabFold API call.
    assert all(r.path.startswith(("/ticket/", "/result/download/")) for r in colabfold.requests)


def test_convert_makes_no_connection_at_all(connections, tmp_path):
    seen, _ = connections
    a3m = REAL / "of3-6t33_A" / "cfdb_uniref30.a3m"
    for model in ("af3", "boltz", "colabfold", "openfold3", "protenix"):
        assert main(["convert", str(a3m), "-m", model, "-o", str(tmp_path / model), "--sequence", Q_6T33, "-q"]) == 0
    assert seen == []


def test_the_guard_catches_another_host(connections):
    """The fixture itself must fail on a stray connection."""
    with pytest.raises(AssertionError, match="unexpected"):
        socket.create_connection(("192.0.2.1", 443), timeout=0.1)


def test_no_scigantic_host_in_the_source():
    """No code path names a Scigantic server. The project URL in the
    User-Agent is a GitHub page and is never requested."""
    hosts = set()
    for p in SRC.rglob("*.py"):
        for m in re.finditer(r"https?://([A-Za-z0-9.-]+)", p.read_text()):
            hosts.add(m.group(1))
    assert not any("scigantic" in h for h in hosts), hosts
    assert hosts <= {"api.colabfold.com", "health.api.nvidia.com", "api.nvcf.nvidia.com", "github.com"}, hosts
