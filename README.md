# scigantic-msa

Protein multiple sequence alignments (MSAs) for structure predictors, without
running a fresh MSA search when one already exists.

For a sequence, scigantic-msa looks for an alignment in this order:

1. your local cache,
2. your Scigantic account, if you are signed in,
3. a public index of already published MSA sets (exact match),
4. the same index for a near-identical sequence, such as a point mutant,
5. a search server, only if you turned one on.

It then writes the input each predictor reads: ColabFold A3M, AlphaFold 3
JSON, Boltz A3M or CSV, Chai-1 aligned parquet, or OpenFold3 MSA directories.

## Install

```
pip install scigantic-msa            # or: uv tool install scigantic-msa
pip install 'scigantic-msa[chai]'    # adds pyarrow, for Chai-1 files
```

The core has no dependencies and needs Python 3.10 or newer. Near-identical
search uses a local `mmseqs` binary when it is installed, and plain Python
otherwise.

## Example

```python
from scigantic_msa import get_msa

msa = get_msa("WGCVCSGSTAVANSHNAGPAYCVGYCGNNGVVTRNANANVAKTA")
print(msa.source, msa.depth)        # exact 66
msa.to_af3_json("input.json")       # or to_boltz(dir), to_chai(dir), to_colabfold(path), to_openfold3(dir)
```

From a terminal:

```
scigantic-msa get chain.fasta --format af3 --out input.json
scigantic-msa get "SEQA:SEQB" --format boltz --out msas/     # ':' separates chains of a complex
scigantic-msa where MKTAYIAKQRQISFVKSHFSRQ                   # which source would answer, and why
```

`msa.source` is always one of `exact`, `near`, `remote`, `cache` or
`account`. A `near` result says which sequence it came from in
`msa.derived` (parent hash, identity, every substitution and indel). Its rows
were rebuilt for your sequence and row 1 is your query, which AlphaFold 3,
Boltz and Chai require.

## Where the alignments come from

The index does not copy alignments. It records, for each sequence, where its
alignment already lives (a URL, a byte offset and a length) and reads it from
there. Every file read is checked: the byte count must match, and the first
row must be the query. A file that fails is skipped and reported, never used.

| source | what | license |
| --- | --- | --- |
| OpenFold3 training MSAs, `s3://openfold3-data` | about 155,000 PDB protein chains, with UniProt hits for pairing | CC BY 4.0, OpenFold Consortium |
| LiteFold human proteome MSAs, Hugging Face `LiteFold/human-proteome-wide-msa` | 18,500 human proteins, ColabFold MSAs | CC BY 4.0, LiteFold. Built from the ColabFold databases, whose own terms also apply |
| OpenProteinSet, `s3://openfold` | about 131,000 PDB chains and the UniClust30 set | CC BY 4.0. Ahdritz et al., OpenProteinSet, arXiv:2308.05326 |

Each result carries `msa.attribution`, and every writer puts a
`scigantic-msa-attribution.json` file next to its output with the source,
license and any note about reuse or pairing.

The index lives at `DEFAULT_INDEX_URL` in `scigantic_msa/index.py`. Set
`SCIGANTIC_MSA_INDEX_URL` to use another copy, including a local directory.
`scigantic-msa index download` fetches the whole index for offline use.

## Privacy

Your sequence may be an unpublished target, so here is what leaves your
machine at each step:

- cache: nothing.
- account: the sequence hash, sent to the Scigantic API you are signed in to.
- exact index match: the first three hex digits of the sequence hash.
- near-identical search: the sequence length, or nothing if you have
  downloaded the index (`scigantic-msa index download`) and have `mmseqs`.
- remote search: the sequence itself, to the server you chose.

Remote search is off until you turn it on, per call (`remote="colabfold"`,
`--remote colabfold`) or by default (`scigantic-msa config set remote
colabfold`). `SCIGANTIC_MSA_NO_REMOTE=1` turns it off whatever the code or
config says. The first remote call in a process prints the server's name.

The public ColabFold server (api.colabfold.com) needs no login to read a
result. Its job id is a hash of the sequence, so anyone who knows the
sequence can download your result. Treat a submission there as publishing
the sequence. The server is a shared resource with a rate limit of about one
new job per minute per IP address; this client backs off, resumes a job from
its id instead of submitting again, and gives up after a bounded wait.

The NVIDIA hosted MSA Search backend (`remote="nvidia"`, needs
`NVIDIA_API_KEY`) is experimental.

## Your alignments in your Scigantic account

Alignments you store are private to you. Nothing is stored unless you ask:

```python
msa = get_msa(seq, remote="colabfold", save=True)   # or msa.save(label="WT")
```

```
scigantic-msa login --email you@lab.edu   # emails a code; creates an account if the address has none
scigantic-msa list
scigantic-msa pull <id> --format chai --out msas/
scigantic-msa push my.a3m                 # store an alignment you already have
scigantic-msa rm <id>
```

Login takes one email round trip, also for a new account; you accept the
terms before the code is used. The token lasts 30 days, is refreshed while
you use it, and is saved with mode 0600 in `~/.config/scigantic/msa.json`.
To use the website later, sign in with Google using the same address, or
choose Forgot password.

In a Scigantic notebook there is no login step: the notebook's own
credentials are exchanged for a 12-hour token that is kept in memory only.
Once signed in, `get_msa` checks your stored alignments before anything else
that leaves the machine.

## Complexes

```python
from scigantic_msa import get_msas

cx = get_msas([seq_a, seq_b])
print(cx.pairing.message)       # for example: paired 8 rows by species across 2 chains
cx.to_af3_json("input.json")
```

Each chain gets its own unpaired MSA. Rows are paired across chains by
species only where the rows say the species: an NCBI taxid (ColabFold's
taxonomy output, `TaxID=` or `OX=` in headers) or a UniProt species code.
Environmental and metagenomic hits have none and stay unpaired; UniRef90
clusters with more than one member are not used either. When pairing is not
possible, `cx.pairing.message` says why.

How pairing reaches each predictor:

| predictor | how |
| --- | --- |
| AlphaFold 3 | UniProt rows in `pairedMsa`, which AF3 pairs itself; otherwise taxid-paired rows first in `unpairedMsa` (run with `--resolve_msa_overlaps=false`) |
| Boltz | one CSV per chain, paired rows share a key |
| Chai-1 | `pairing_key` column |
| ColabFold | the complex A3M layout, paired rows first |
| OpenFold3 | `uniprot_hits.a3m` per chain, or `colabfold_paired.a3m` |

## Settings

| variable | meaning |
| --- | --- |
| `SCIGANTIC_MSA_INDEX_URL` | index location (URL or directory) |
| `SCIGANTIC_MSA_REMOTE` | default remote backend: `colabfold`, `nvidia` or `off` |
| `SCIGANTIC_MSA_NO_REMOTE=1` | never send a sequence to a server |
| `SCIGANTIC_MSA_OFFLINE=1` | use only what is cached |
| `SCIGANTIC_MSA_CACHE` | cache directory (default `~/.cache/scigantic-msa`) |
| `SCIGANTIC_MSA_MOUNTS` | read source buckets from mounts, e.g. `s3://openfold/=/mnt/openfold` |
| `SCIGANTIC_MSA_CONTACT` | text added to the User-Agent sent to search servers |
| `SCIGANTIC_MSA_TOKEN` | an account token, instead of the saved one |
| `SCIGANTIC_API_URL` | Scigantic API (default https://api.scigantic.com) |
| `HF_TOKEN` | optional, sent only to huggingface.co |

## Development

```
pip install -e '.[dev]'
pytest
```

Tests run offline against small real files in `tests/fixtures/real`, and a
local HTTP server that stands in for the index, the ColabFold server, the
NVIDIA endpoint and the Scigantic API. `tests/test_predictors.py` loads the
outputs with Boltz's and Chai-1's own parsers when those packages are
available; see the file for how to point it at them.

## License

MIT. The alignments themselves are under their sources' licenses, listed
above.
