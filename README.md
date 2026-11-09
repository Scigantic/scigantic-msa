# scigantic-msa

Build a multiple sequence alignment (MSA) for a protein sequence, and write
the input files for the structure predictor you are about to run:
AlphaFold 3, Boltz, Chai-1, ColabFold, OpenFold3 or Protenix.

Each MSA is the ColabFold MMseqs2 server's search merged with the
alignments that others have already computed and published as open data:
the OpenFold3 training MSAs, LiteFold's human proteome MSAs and, if you
ask, OpenProteinSet. These are read where their publishers put them; there
is no index or mirror of ours in between. Every row keeps a record of
where it came from. For a complex, the server's own paired search is kept
and pairs by species from open data are added when the chains share an
organism.

With `--open-only` the sequence is not sent to any search server, which is
the mode for unpublished sequences. There is no account and no login, and
nothing is stored on our side.

The open data lookup is on by default, so `run` contacts the open data
hosts (two public S3 buckets and Hugging Face) even when the ColabFold
server is your own. On a machine with no internet access, pass
`--no-open` or set `SCIGANTIC_MSA_NO_REMOTE=1`.

## Install

```
pip install scigantic-msa            # or: uv tool install scigantic-msa
pip install 'scigantic-msa[chai]'    # adds pyarrow, needed for Chai-1 files
uv tool install 'scigantic-msa[chai]'  # the same, for the uv tool
```

Python 3.10 or newer. The core has no dependencies. If your shell cannot
find `scigantic-msa` after installing, `python -m scigantic_msa` runs the same
command.

## Privacy

Your sequence may be unpublished, so read this before you run a search.

- `run` and `MSA().get()` send the sequence to a third-party server:
  api.colabfold.com by default. The public ColabFold server needs no login
  to read a result, and the job id is a hash of the sequence, so anyone who
  knows the sequence can download the result. Treat a submission there as
  publishing the sequence. For a complex, the chains are also sent together
  once more, for the server's paired search.
- The open data lookup never sends the sequence. Each set's table of
  sequences is downloaded once and the sequence is looked up on your
  machine. If a table holds the sequence, the public files of that entry
  are then read from their publisher. So the publisher's host learns which
  public entry you asked for, and so that you hold that sequence or one
  identical to it.
- With a search server, the lookup also sends the MD5 of the sequence
  (UniProt's own sequence checksum) to rest.uniprot.org, to find which
  UniProt entries, and so which organisms, have exactly this sequence, and
  tries up to 25 of their accessions as OpenProteinSet UniClust30 entries
  on openfold.s3.amazonaws.com. The MD5 of a sequence close to a public
  one can be reversed: a point mutant of a known protein can be found from
  its MD5 by trying the variants. So with `--open-only` this UniParc
  lookup is off unless you pass `--uniparc` (`MSA(server=None, uniparc=True)`).
  Without it, the organisms of a chain come only from the open files
  themselves (OpenFold3's UniProt hits, LiteFold for human proteins), which
  matters for pairing a complex, and UniClust30 entries are not used.
  `--no-uniparc` turns it off with a server too.
- `--open-only` (`MSA(server=None)` in Python) does the open data lookup and
  nothing else: no search server receives the sequence, and there is
  nothing to confirm. Before a run the command line says which requests
  the lookup makes, and after it, what was sent where, including requests
  that failed.
- The attribution file written next to the model's input holds the
  sequence's sha256 (`seq_sha256`) and, for a `--near` match, the public
  entry and its identity to the query. Like the MD5, these can identify a
  close variant of a public protein, so treat the file as sensitive when the
  sequence is.
- The Python API refuses to send anything to a search server unless you
  pass `remote=True`. Constructing a client yourself (`ColabFoldClient(...).search(seq)`, or
  `NvidiaClient`) is that same choice made in code, so it sends without
  `remote=True`; it does not use the cache or say where it sends. The
  command line says which server will receive the sequence and asks once; `--yes` skips the question for scripts. Without a terminal and
  without `--yes`, it stops and sends nothing.
- `convert` and `MSA().from_a3m()` make no network connection at all.
- The tool never contacts a Scigantic server.
- The sequence is never written to a log. Results are kept in a local cache
  so the same sequence is not submitted twice: `SCIGANTIC_MSA_CACHE` if set,
  else `$XDG_CACHE_HOME/scigantic-msa` if `XDG_CACHE_HOME` is an absolute
  path (a relative one is ignored, as the XDG spec says), else
  `~/.cache/scigantic-msa` (on Windows, `%LOCALAPPDATA%\scigantic-msa\cache`,
  when `LOCALAPPDATA` is absolute). `scigantic-msa cache` prints the one in
  use. Delete it to forget them, or pass `--no-cache` (open data then goes
  to a temporary folder: the tables are kept there for the life of the
  process, the files read for each `MSA` object are removed by `close()`,
  and the folder is removed at exit or on SIGTERM). Only you can read its
  files or list its folders, which are named by a hash of the sequence; a
  cache made by 0.2.7 or older is made so the first time it is used, even
  only to read it. The tool only ever removes group and other permissions
  there; it never gives you back write access you took away. If the cache
  directory cannot be written (read-only home, say), the tool warns once
  and runs without it.
- `SCIGANTIC_MSA_NO_REMOTE=1` blocks all network traffic, whatever the code
  asks, including a client used directly and the open data lookups: every
  request checks it, and so does every retry. Any value but `0`, `false`, `no` or `off`
  counts, including one of only spaces; an empty value
  (`SCIGANTIC_MSA_NO_REMOTE=`) is the same as not setting it. Results,
  tables and open files already in the cache are still used, since they
  send nothing.

The public ColabFold server is a shared resource with a rate limit of about
one new job per minute per IP address. This client backs off, honours
Retry-After, resumes a job from its id instead of submitting it again, and
gives up after a bounded wait. One `run` sends at most 100 new jobs
(sequences and paired searches) to it; split larger inputs into several runs. To use your own server, pass
`--host URL` (or `MSA(host=URL)`).

## Command line

```
scigantic-msa run SEQ_OR_FASTA --model boltz --out job/
scigantic-msa run SEQ_OR_FASTA --model boltz --out job/ --open-only
scigantic-msa run chains.fasta --complex --model af3 --out job/ --yes
scigantic-msa run many.fasta --batch --model boltz --out jobs/
scigantic-msa convert my_alignment.a3m --model protenix --out job/
scigantic-msa sources      # the cache, the open data sets, their terms, and the tables in the cache
scigantic-msa sources --fetch   # download the tables now (67 MB to download, about 96 MB on disk)
scigantic-msa sources --fetch --openproteinset   # also the OpenProteinSet tables (about 610 MB)
scigantic-msa cache        # prints the cache directory
scigantic-msa version      # or --version
```

`--model` is one of `af3`, `boltz`, `chai`, `colabfold`, `openfold3`,
`protenix`. `convert` takes one A3M (or `.a3m.gz`, or Stockholm `.sto`) per
chain, in chain order. For a complex, `--uniprot FILE` (once per chain, in
the same order) adds an alignment of that chain against UniProt, such as
AlphaFold 3's or OpenFold3's `uniprot_hits`; see Complexes.

The argument to `run` is a FASTA file (UTF-8, UTF-16 as Windows Notepad
saves it, or gzipped), `-` for stdin, or the sequence itself. So that a
mistyped file name is never sent as a protein, a typed argument is refused when it names a directory, when a file with that name
plus an extension exists (`query` next to `query.fasta`), or when it is
shorter than 10 residues. To send a short peptide on purpose, start it
with `=` and quote it, as in `'=MKVLAAG'` (zsh reads an unquoted word that
starts with `=` as a command name; the quoted form works in bash, zsh and
fish). A sequence that is only `A`, `C`, `G`, `T`,
`U` and `N` gets a warning that it looks like DNA or RNA.

`:` inside a sequence separates the chains of one complex. A FASTA file
with several records needs one of:

- `--complex`: the records are one complex, one chain per record (at most
  20 chains).
- `--batch`: each record is its own job, written to its own folder under
  `--out` and named after the record, such as `jobs/rec1/boltz/` (at most
  100 jobs). With `--name run1`, every job's name starts with it:
  `jobs/run1_rec1/boltz/`. A job the server cannot do does not stop the
  others; the command ends with an error naming the failed jobs.

Options for the open data part of `run`:

- `--open-only`: open data alone; the sequence goes to no search server.
- `--no-open`: the server's result alone, as in 0.2.
- `--openproteinset`: also OpenProteinSet's PDB alignments (2021 searches
  of BFD, UniClust30, UniRef90 and MGnify). Its tables are about 610 MB the
  first time, so it is off by default, and `run` asks before that download
  (`--yes` skips the question; without a terminal, `--yes` is needed).
- `--uniparc` / `--no-uniparc`: send the sequence's MD5 to UniParc, or not
  (see Privacy). On by default, off with `--open-only`.
- `--near`: also entries at 95% identity or more over 90% of the query,
  such as another construct of the same protein. Needs an `mmseqs` binary
  on PATH; the search runs on your machine.
- `--open-pairs auto|always|never`: see Complexes.

Before sending, `run` says how many sequences go to which server, and for
a complex or batch, how many chains or jobs, and what the open data lookup
sends. After writing, `run` prints each chain's depth. With open data it
gives rows, an estimated Neff at 80% identity and species, for the merged
MSA and for the server's result alone, and how many rows each open set
added; `convert` and `run --no-open` give the number of sequences. Both
warn when the alignment is shallow or holds only the query, which makes
the model run in single-sequence mode. If you stop `run` with Ctrl-C while it waits, run the
same command again: it picks up the same job without sending it again.

With `--open-only`, a run where the open data could not be searched (no
network, a host that fails, a table that cannot be downloaded) ends with
an error that names what failed, and writes nothing, rather than a query
only MSA. A sequence that no table holds is not an error: it gets the
query alone and a warning. An entry a table lists but whose files are not
published says so.

`--host` takes an `http://` or `https://` URL in ASCII with no spaces,
backslash, user name or password, query or fragment, and so do
`SCIGANTIC_MSA_COLABFOLD_URL` and `SCIGANTIC_MSA_NVIDIA_URL`. Any spelling of the public server's URL
(`HTTPS://API.colabfold.com:443/` or `https://api.colabfold.com/./`, say)
is the public server, reached over https. `--server nvidia` uses NVIDIA's hosted MSA Search and needs
`NVIDIA_API_KEY`; without it, `run` stops before asking to send.

Each model's files go in their own folder under `--out` (`job/boltz/`,
`job/af3/` and so on), so several models can share one `--out` without
overwriting each other. After writing the files, the command prints the
command to run the model on them.

## Python

### One call, any model

```python
from scigantic_msa import MSA, write_model

chains = [MSA().from_a3m("my_alignment.a3m")]
files, next_command = write_model("boltz", chains, out_dir, name="job")
```

`write_model` takes the same model names as `--model`
(`af3`, `boltz`, `chai`, `colabfold`, `openfold3`, `protenix`), writes the
files into `out_dir/<model>/`, and returns the paths it wrote plus the
command to run next. Use it unless you need an adapter's return value
directly: the adapters do not share a signature.

| Adapter | `out_dir` | `name` | returns |
|---|---|---|---|
| `AlphaFold3` | optional | yes | `dict` |
| `Boltz` | **required** | no | `str` (YAML) |
| `Chai` | **required** | no | `Path` |
| `ColabFold` | **required** | yes | `Path` |
| `OpenFold3` | **required** | yes | `dict` |
| `Protenix` | **required** | yes | `list[dict]` |

### Open data from Python

```python
from scigantic_msa import MSA

merged = MSA().get(sequence, remote=True)          # ColabFold + open data
private = MSA(server=None).get(sequence)           # open data only, nothing sent to a search server
a, b = MSA().get([seq_a, seq_b], remote=True)      # a complex: both chains carry its pairs
print(merged.provenance["depth"], merged.provenance["merge"])
```

`MSA(open_data=False)` gives the server's result alone. `openproteinset`,
`near`, `open_pairs` and `uniparc` match the command line options
(`uniparc` defaults to off with `server=None`). Each row has
`row.extra["prov"]`, a `Prov(source, role, entry, url, record, anchor)`.
`merged.provenance["open"]` records what the lookup sent (`sent`), what
it could not search or read (`failed`) and the entries listed but not
published (`unpublished`). One `MSA` object can be used from several
threads. With `cache=False`, use it in a `with` block, or call `close()`,
to remove the files it read.

### Adapters directly

Every adapter takes one result, or a list of results for the chains of a
complex. The result carries its sequence; if you also pass the sequence,
it must match.

AlphaFold 3:

```python
import json
from scigantic_msa import MSA
from scigantic_msa.adapters import AlphaFold3
result = MSA().get(sequence, remote=True)
json.dump(AlphaFold3.prepare(result, name="job"), open("job.json", "w"))
```

`AlphaFold3.prepare(result, name="job", out_dir="af3/", inline=False)`
writes the MSAs as files next to `af3/job.json` instead of inside it.

Boltz:

```python
from scigantic_msa import MSA
from scigantic_msa.adapters import Boltz
result = MSA().get(sequence, remote=True)
yaml_text = Boltz.prepare(result, out_dir="boltz/")
open("job.yaml", "w").write(yaml_text)        # then: boltz predict job.yaml
```

Chai-1:

```python
from scigantic_msa import MSA
from scigantic_msa.adapters import Chai
result = MSA().get(sequence, remote=True)
msa_dir = Chai.prepare(result, out_dir="chai_msas/")
open("job.fasta", "w").write(Chai.fasta(result))   # chai-lab fold --msa-directory <msa_dir> job.fasta out/
```

Protenix:

```python
import json
from scigantic_msa import MSA
from scigantic_msa.adapters import Protenix
result = MSA().get(sequence, remote=True)
json.dump(Protenix.prepare(result, out_dir="protenix_msas/"), open("job.json", "w"))
```

ColabFold and OpenFold3 work the same way: `ColabFold.prepare(result,
out_dir=...)` returns the path of the A3M for `colabfold_batch`, and
`OpenFold3.prepare(result, out_dir=...)` returns the query JSON as a dict.

With an alignment you already have, and no network:

```python
result = MSA().from_a3m("my_alignment.a3m", sequence=sequence)
result = MSA().from_a3m("chainA.a3m", uniprot="chainA_uniprot_hits.sto")  # for pairing
```

## Open data

| set | where it is read | licence | searched |
| --- | --- | --- | --- |
| OpenFold3 training MSAs (`openfold3`) | `s3://openfold3-data/msas/pdb_msas_completed/` | CC BY 4.0 | UniProtKB, UniRef90, MGnify (jackhmmer); UniRef30 + ColabFold envdb; 2025 releases |
| LiteFold human proteome (`litefold`) | huggingface.co/datasets/LiteFold/human-proteome-wide-msa, one pinned commit | CC BY 4.0 (dataset card) | UniRef30 2302 + ColabFold envdb (MMseqs2) |
| OpenProteinSet PDB (`openproteinset`, opt in) | `s3://openfold/alignment_db/` | CC BY 4.0 | UniRef90, MGnify, BFD + UniClust30; 2021 |
| OpenProteinSet UniClust30 (`uniclust30`) | `s3://openfold/uniclust30/` | CC BY 4.0 | UniClust30 |

How an entry is found, with no index of ours:

- OpenFold3: `msas/pdb_msas_completed_representives.fasta` (55 MB) gives the
  sequence of every MSA directory. Only representative chains have a
  directory, so the lookup is by sequence, not by PDB chain id.
- LiteFold: `manifest.tsv` (12 MB) gives the sequence of every protein.
- OpenProteinSet: `data_caches/chain_data_cache.json` (256 MB) and
  `alignment_db/alignment_db.index` (352 MB); files are read by byte range.
- UniClust30: UniParc is asked by MD5 for the sequence's UniProtKB
  accessions, and up to 25 of them are tried as entries. This needs the
  UniParc lookup, which is off with `--open-only`. Some UniClust30 files
  start with a `_consensus` row and leave the cluster's representative
  out; the consensus is in the representative's columns, so such a file is
  used when the consensus is as long as the query and has the query's
  residue at half its positions or more.

Each table is downloaded once into the cache, with its URL, ETag, size and
sha256 recorded, and kept as a small SQLite table keyed by the sha256 of
the sequence. A download is kept only when it is whole and is the table it
should be: a web page sent in its place (a captive portal, say) or a cut
download is refused, and a table whose file is damaged is built again.
Its ETag is checked at most once a day
(`SCIGANTIC_MSA_OPEN_REFRESH_HOURS`) and a changed table is downloaded
again. The table versions used are named in the attribution file. Files of
matched entries are kept in the cache as parsed rows. A file is read until
its 50,000th row or 256 MB, whichever comes first (OpenFold3's
`uniprot_hits.sto` files reach 2.9 GB); their rows are ordered best hit
first, so a cut drops the weakest hits. A file whose first row is not the
sequence is refused; a single broken row is dropped and counted. A file
that ends early (fewer bytes than the server said, or a Stockholm file
without its closing `//`) is read again and never kept; only the cap above
cuts a file, and the attribution file then says `truncated: true`. A host
that does not resolve or refuses the connection is not tried again, each
request has an overall deadline, and a table that failed is tried again
after five minutes.

Merging, for each chain:

1. A row with the query's residues is dropped.
2. Rows with the same residues (gaps removed) are kept once, the
   ColabFold copy first.
3. Rows of the same entry (accession) over the same part of it are kept
   once: the one with the most aligned residues. "The same part" means the
   entry spans the rows state (`/12-200` in Stockholm names, the target
   start and end in ColabFold headers) overlap by at least half of the
   shorter, or, when a row states no span, its span on the query. Rows of
   one entry over different parts of it, such as repeats, all stay.
   Deduplicating by exact strings alone kept such copies and raised Neff
   by up to 38% on the targets this was measured on.
4. The ColabFold rows come first, in the server's order, then the open
   rows, one from each open file in turn. A model that reads only the
   first N rows (Boltz reads 8,192 by default, AlphaFold 3 16,384) sees the
   whole server result first, so open rows count most where the server
   found few.

Measured on 50 PDB monomers released from 2023 on, each of which has an
open entry, with ColabFold results fetched once and replayed: the merged
MSA's Neff at 80% identity (after the dedup above) was a median 2.0 times
ColabFold's with the default sets, and 2.5 times with `--openproteinset
--near`. Of the 17 targets where ColabFold gave fewer than 5 Neff per
residue, 5 went from under 1.4 to over 5 with the default sets (6 with
both options), and 5 stayed under 1 because no open set has more for them.
Depth is not accuracy. AlphaFold2 (ColabFold 1.5.5, 5 models, no
templates) on those 17 targets gave no better structures with the merged
MSA: the top model's TM-score changed by a median -0.004 on the 13 targets
with most of the chain modelled, and dropped by more than 0.02 on 3. With
open data only (`--open-only`) the median change was -0.006, so for these
proteins open data alone gave structures as good as the server's MSA.

Open data only has what was published: PDB chains (OpenFold3 to 2025,
OpenProteinSet to 2021), the human proteome and UniClust30
representatives. A sequence in none of them gets nothing from open data,
and `--open-only` then gives the query alone.

## Checks

- An A3M is checked strictly: the first record must be the query (no gaps,
  no lowercase), only `A-Z`, `a-z` and `-` are allowed, and every row must
  have as many match columns as the query has residues.
- Each adapter reads back what it wrote: every file it names exists, is not
  empty, starts with the chain's sequence, and has the name the model looks
  for. Chai-1 skips a chain's MSA without an error when the file name is
  wrong, and Protenix ignores a file in `precomputed_msa_dir` that is not
  called `pairing.a3m` or `non_pairing.a3m`; here both are errors.
- `Protenix.check(path_or_jobs)` runs the same checks on any Protenix input
  JSON, including a missing `*MsaPath`, which would make Protenix start its
  own MSA search and send the sequence to a server.
- Protenix reads only the 20 standard amino acids and `X` in a protein
  sequence and stops on any other letter, such as `U` (selenocysteine), so
  the Protenix adapter refuses them before writing anything.

## Complexes

With the ColabFold server, a complex gets the server's own paired search
(`/ticket/pair` with greedy pairing, as `colabfold_batch` sends for a
heteromer), and those pairs come first. Every server pair is kept as the
server made it, except a pair that repeats an earlier pair in every chain.
Boltz reads a row that repeats an earlier row of the same chain only once,
so such a pair is not paired in Boltz; the attribution file says how many.
Then pairs by species from the open rows the merge kept are added, for species and entries the server's pairs do not
already have, but only when the chains come from the same organism: every
chain's organism must be known (from the UniProt entries that have exactly
its sequence, OpenFold3 UniProt hits that are copies of it, or LiteFold for
human proteins) and some species must have all of them. For chains from
different organisms, such as a virus protein and its human receptor, pairs
by species are not coevolved partners and are left out; for chains of
unknown organism they are left out too. `--open-pairs always` adds them
anyway and `--open-pairs never` never does. The tool says which applied.
Every writer then uses exactly these pairs. On 15 benchmark heterodimers
the server's paired search found more pairs than the open rows on large
families from one organism, and the open rows added species the server did
not pair; for one human and snake toxin complex the open rows alone would
have made 112 pairs where the server made 6.

Chai-1 keeps at most 9,999 uniref90, 49,999 uniprot and 4,999 mgnify rows
of a chain (its per-source quotas). Paired rows come first and the rest are
in merge order, best first, so the weakest go; `prepare` says how many.

Without a paired search (`--server nvidia`, or when it fails), the rows
are paired by species as follows.

Each chain gets its own unpaired MSA. Rows are paired across chains only
where the rows say the species: an NCBI taxid (from ColabFold's taxonomy
output, or `TaxID=` / `OX=` in headers) or a UniProt species code, and only
when it names one species. A UniRef row names its cluster's common taxon,
so it counts only for a cluster of one member (`n=1`). Taxa above species
(root, Bacteria, Archaea, Eukaryota, Viruses, "unclassified ..." and
metagenome names) and UniProt's virtual species codes, which start with 9
(`9GAMM`, `9BACT`) and stand for many organisms, are not used.
Environmental and metagenomic hits have none and stay unpaired: a row never
takes the species of an identical row from another source. When
pairing is not possible the tool says why, for example "no species is
shared by any two chains; the chains get unpaired MSAs".

Every species found in two or more chains is paired, so a chain from
another organism, or one with no species information at all (only
environmental or clustered hits, or a designed binder), does not stop the
others pairing. Rows shared by more chains come first. The tool says what
paired, for example "paired 8 rows by species across chains A, B; chain C
shares no species with another chain, so it is not paired", or "paired 5
rows by species across chains A, B, C (3 with chains A, B, C, 2 with
chains A, B)". In a paired row, a chain without that species is filled in
as each model expects: gaps for ColabFold and OpenFold3, gap rows that
carry the pairing key for Chai-1, no key for Boltz, and one of its own
unpaired rows for AlphaFold 3. AlphaFold 3 joins unpaired rows by index,
so each paired row sits at the same row in every chain that has it. A
chain with too few rows of its own to stand in gets a gap row there, as
Boltz does itself; such rows count in its profile, so the pairs are put in
the order that needs the fewest, and the tool says how many each chain
got.

Several jobs can be written into one `--out` (`--name ab`, then `--name
ac`, or two alignments of one sequence from `convert`, `--host` and the
public server). Every MSA file is named, or put in a folder named, for its
content, so one job never rewrites a file another reads, and identical
files are shared: Boltz files are `<sha256>.<hash>.csv` or `.a3m`, Chai-1
files go in `msas/paired-<hash>/` or `msas/unpaired-<hash>/` (the printed
command passes that folder to `--msa-directory`), OpenFold3 folders are
`main/<sha256>/<hash>/` and `paired/<hash>/<sha256>/`, Protenix folders
are `<sha256>/<hash>/`, and AlphaFold 3 files with `inline=False` are
`<sha256>.<hash>.unpaired.a3m`.

| model | how pairing is passed on |
| --- | --- |
| AlphaFold 3 | paired rows first in `unpairedMsa`; run with `--resolve_msa_overlaps=false` |
| Boltz | one CSV per chain; paired rows share a key |
| Chai-1 | `pairing_key` column |
| ColabFold | the complex A3M layout, paired rows first |
| OpenFold3 | `colabfold_paired.a3m` per chain |
| Protenix | `pairing.a3m` per chain, with the taxid (or UniProt species code) in each header so Protenix pairs on the same species; for the pairs of `run`, each pair's number stands in for the species so Protenix pairs exactly those rows |

ColabFold server results carry taxids but no UniProt rows. If you have a
UniProt alignment for every chain (`convert --uniprot`, or
`from_a3m(..., uniprot=...)`), its rows are used for pairing instead:
AlphaFold 3 gets them as `pairedMsa` and pairs them itself, OpenFold3 gets
`uniprot_hits.a3m` in each chain's directory, Protenix's `pairing.a3m` is
built from them, and the other models pair them by species as above. The
file must have rows with UniProt headers (`tr|ACC|NAME_SPECIES`).

## Formats

The file layouts follow the models' own code: AlphaFold 3
`folding_input.py`, Boltz `main.py` and `parse/csv.py`, Chai-1
`aligned_pqt.py`, ColabFold `input.py`, OpenFold3 0.5.1
`inference_query_format.py`, and Protenix 2.0.0 `runner/msa_search.py`,
`protenix/data/msa/msa_featurizer.py` and `msa_utils.py`. The opt-in tests
in `tests/test_predictors.py` load the files with Boltz, Chai-1 and
Protenix's own code; for Protenix that is its input JSON parser and MSA
featurisation, up to the features and without model weights.

## Data sources and attribution

The alignments come from:

- ColabFold MMseqs2 server, run by the ColabFold team, under its terms.
  Cite Mirdita et al., ColabFold: making protein folding accessible to
  all, Nature Methods 19, 679-682 (2022), and the databases it searches
  (UniRef30 and the ColabFold environmental database, CC BY 4.0).
- NVIDIA MSA Search (experimental here), under NVIDIA's API trial terms.
- The open data sets above, each CC BY 4.0: credit the publisher as the
  attribution file says. The rows were searched against UniProt (CC BY
  4.0), the ColabFold databases and BFD (CC BY 4.0) and MGnify (CC0).
  UniClust30's terms were not confirmed; the Uniclust authors are reported
  to publish it as CC BY-SA 4.0, which would make alignments with its rows
  share-alike. Rows from OpenProteinSet's `bfd_uniclust` files and from
  UniClust30 entries are flagged in the attribution file for that reason.
- UniProt UniParc (CC BY 4.0), for the organisms of a sequence.

Every adapter that writes files also writes `scigantic-msa-attribution.json`
next to them. `sources` lists every distinct credit: each server and open
set used, with its licence, citation and, for an open set, the version of
its table (URL, ETag, Last-Modified, size and sha256 of each file). For
each chain it holds its length, depth, `seq_sha256` (the files next to it
are named with it too), where the alignment came from, the open entries
and files read (with their ETags and row counts, and any error), how many
rows each source kept after the merge and why others were dropped, the
organisms found, the depth summary, and for a complex the number of server
and open pairs and the rule that applied. It holds no local path and no
server job id. Next to it, `scigantic-msa-rows/<seq_sha256>.<hash>.tsv`
gives one line per row of each chain: its id, source, file kind, entry,
the URL of the file, the byte range read from it when only part was read
(OpenProteinSet's shared `alignment_db` files), the row's position in that
file or range, its licence and which
other sources had the same row. When several jobs, models or runs write
into one folder, the attribution file keeps one entry per write under
`outputs`. A write replaces an older entry only when it rewrote all of
that entry's files.

## Settings

| variable | meaning |
| --- | --- |
| `SCIGANTIC_MSA_CACHE` | cache directory (default `$XDG_CACHE_HOME/scigantic-msa` when that is absolute, else `~/.cache/scigantic-msa`; `scigantic-msa cache` prints it) |
| `SCIGANTIC_MSA_NO_REMOTE=1` | never send a sequence to any server (any value but empty, `0`, `false`, `no`, `off`) |
| `SCIGANTIC_MSA_COLABFOLD_URL` | another ColabFold MMseqs2 server |
| `SCIGANTIC_MSA_NVIDIA_URL` | another NVIDIA MSA Search endpoint |
| `SCIGANTIC_MSA_CONTACT` | text added to the User-Agent sent to the server, such as an email address (one line, Latin-1; `run` stops before sending otherwise) |
| `NVIDIA_API_KEY` | key for `--server nvidia` |
| `SCIGANTIC_MSA_OPEN_REFRESH_HOURS` | how often a table's ETag is checked (default 24) |
| `SCIGANTIC_MSA_TEST_COLABFOLD_REPLAY` | test only: a folder of saved ColabFold results named `<job id>.tar.gz`, read instead of the server, which is then never contacted |

## Development

```
pip install -e '.[dev]'
pytest
```

The tests run offline against small real alignment files in
`tests/fixtures/real` and a local HTTP server standing in for the ColabFold
and NVIDIA APIs and the open data hosts. `tests/test_predictors.py` loads the output with Boltz's
and Chai-1's own parsers when interpreters with those packages are given;
see the file for how.

## License

MIT. The alignments are under the terms of the server and databases they
came from.
