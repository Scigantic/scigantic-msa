# scigantic-msa

Get a multiple sequence alignment (MSA) for a protein sequence once, and
write the input files for the structure predictor you are about to run:
AlphaFold 3, Boltz, Chai-1, ColabFold, OpenFold3 or Protenix.

The alignment comes from the ColabFold MMseqs2 server (or NVIDIA's MSA
Search, if you have a key), or from an A3M file you already have. There is
no account and no login, and nothing is stored on our side.

## Install

```
pip install scigantic-msa            # or: uv tool install scigantic-msa
pip install 'scigantic-msa[chai]'    # adds pyarrow, needed for Chai-1 files
```

Python 3.10 or newer. The core has no dependencies.

## Privacy

Your sequence may be unpublished, so read this before you run a search.

- `run` and `MSA().get()` send the sequence to a third-party server:
  api.colabfold.com by default. The public ColabFold server needs no login
  to read a result, and the job id is a hash of the sequence, so anyone who
  knows the sequence can download the result. Treat a submission there as
  publishing the sequence.
- The Python API refuses to send anything unless you pass `remote=True`.
  The command line says which server will receive the sequence and asks
  once; `--yes` skips the question for scripts. Without a terminal and
  without `--yes`, it stops and sends nothing.
- `convert` and `MSA().from_a3m()` make no network connection at all.
- The only network traffic is to the server you chose. The tool never
  contacts a Scigantic server.
- The sequence is never written to a log. Results are kept in a local cache
  (`~/.cache/scigantic-msa`, or `SCIGANTIC_MSA_CACHE`) so the same sequence
  is not submitted twice; delete the directory to forget them, or pass
  `--no-cache`. If the cache directory cannot be written (read-only home,
  say), the tool warns once and runs without it.
- `SCIGANTIC_MSA_NO_REMOTE=1` blocks every search, whatever the code asks.
  Results already in the cache are still used, since they send nothing.

The public ColabFold server is a shared resource with a rate limit of about
one new job per minute per IP address. This client backs off, honours
Retry-After, resumes a job from its id instead of submitting it again, and
gives up after a bounded wait. To use your own server, pass `--host URL`
(or `MSA(host=URL)`).

## Command line

```
scigantic-msa run SEQ_OR_FASTA --model boltz --out job/
scigantic-msa run chains.fasta --complex --model af3 --out job/ --yes
scigantic-msa run many.fasta --batch --model boltz --out jobs/
scigantic-msa convert my_alignment.a3m --model protenix --out job/
scigantic-msa version
```

`--model` is one of `af3`, `boltz`, `chai`, `colabfold`, `openfold3`,
`protenix`. `convert` takes one A3M (or `.a3m.gz`, or Stockholm `.sto`) per
chain, in chain order. For a complex, `--uniprot FILE` (once per chain, in
the same order) adds an alignment of that chain against UniProt, such as
AlphaFold 3's or OpenFold3's `uniprot_hits`; see Complexes.

The argument to `run` is a FASTA file (UTF-8, UTF-16 as Windows Notepad
saves it, or gzipped), `-` for stdin, or the sequence itself. So that a mistyped file name is never sent as a protein, a typed
argument is refused when it names a directory, when a file with that name
plus an extension exists (`query` next to `query.fasta`), or when it is
shorter than 10 residues. To send a short peptide on purpose, start it
with `=`, as in `=MKVLAAG`. A sequence that is only `A`, `C`, `G`, `T`,
`U` and `N` gets a warning that it looks like DNA or RNA.

`:` inside a sequence separates the chains of one complex. A FASTA file
with several records needs one of:

- `--complex`: the records are one complex, one chain per record (at most
  20 chains).
- `--batch`: each record is its own job, written to its own folder under
  `--out` and named after the record, such as `jobs/rec1/boltz/` (at most
  100 jobs). A job the server cannot do does not stop the others; the
  command ends with an error naming the failed jobs.

Before sending, `run` says how many sequences go to which server, and for
a complex or batch, how many chains or jobs. After writing, `run` and
`convert` print the depth of each chain's alignment, and warn when it is
shallow or holds only the query, which makes the model run in
single-sequence mode. If you stop `run` with Ctrl-C while it waits, run the
same command again: it picks up the same job without sending it again.

`--host` takes an `http://` or `https://` URL, and so do
`SCIGANTIC_MSA_COLABFOLD_URL` and `SCIGANTIC_MSA_NVIDIA_URL`.
`--server nvidia` uses NVIDIA's hosted MSA Search and needs
`NVIDIA_API_KEY`.

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
open("job.fasta", "w").write(Chai.fasta(result))   # chai-lab fold --msa-directory chai_msas/ job.fasta out/
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

## Complexes

Each chain gets its own unpaired MSA. Rows are paired across chains only
where the rows say the species: an NCBI taxid (from ColabFold's taxonomy
output, or `TaxID=` / `OX=` in headers) or a UniProt species code.
Environmental and metagenomic hits have none and stay unpaired. When
pairing is not possible the tool says why, for example "no species is
shared by all chains; the chains get unpaired MSAs".

| model | how pairing is passed on |
| --- | --- |
| AlphaFold 3 | paired rows first in `unpairedMsa`; run with `--resolve_msa_overlaps=false` |
| Boltz | one CSV per chain; paired rows share a key |
| Chai-1 | `pairing_key` column |
| ColabFold | the complex A3M layout, paired rows first |
| OpenFold3 | `colabfold_paired.a3m` per chain |
| Protenix | `pairing.a3m` per chain, with the taxid (or UniProt species code) in each header so Protenix pairs on the same species |

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
`protenix/data/msa/msa_featurizer.py` and `msa_utils.py`.

## Data sources and attribution

The alignments come from the server you chose, under its terms:

- ColabFold MMseqs2 server, run by the ColabFold team. Cite Mirdita et al.,
  ColabFold: making protein folding accessible to all, Nature Methods 19,
  679-682 (2022), and the databases it searches (UniRef30 and the ColabFold
  environmental database).
- NVIDIA MSA Search (experimental here), under NVIDIA's API trial terms.

Every adapter that writes files also writes `scigantic-msa-attribution.json`
next to them, with the server, mode, date and the terms to credit.

## Settings

| variable | meaning |
| --- | --- |
| `SCIGANTIC_MSA_CACHE` | cache directory (default `~/.cache/scigantic-msa`) |
| `SCIGANTIC_MSA_NO_REMOTE=1` | never send a sequence to any server |
| `SCIGANTIC_MSA_COLABFOLD_URL` | another ColabFold MMseqs2 server |
| `SCIGANTIC_MSA_NVIDIA_URL` | another NVIDIA MSA Search endpoint |
| `SCIGANTIC_MSA_CONTACT` | text added to the User-Agent sent to the server, such as an email address |
| `NVIDIA_API_KEY` | key for `--server nvidia` |

## Development

```
pip install -e '.[dev]'
pytest
```

The tests run offline against small real alignment files in
`tests/fixtures/real` and a local HTTP server standing in for the ColabFold
and NVIDIA APIs. `tests/test_predictors.py` loads the output with Boltz's
and Chai-1's own parsers when interpreters with those packages are given;
see the file for how.

## License

MIT. The alignments are under the terms of the server and databases they
came from.
