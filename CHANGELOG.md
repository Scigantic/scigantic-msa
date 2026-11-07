# Changelog

## 0.2.4 (unreleased)

Changes you may notice:

- `scigantic-msa-attribution.json` keeps one entry per write under
  `outputs`, and `sources` lists every distinct credit. Before, the last
  write into a folder replaced the file, so two models or jobs written
  there from Python, or a second `convert` or `run` with another `--name`
  into the same `--out`, lost the earlier credit. A write replaces an older
  entry only when it rewrote all of that entry's files. A file from 0.2.3
  is kept as the first entry. The old top-level `chains`, `files` and
  `notes` keys are now inside each entry.
- `--name` with `--batch` is a prefix for every job: `--name run1` writes
  `<out>/run1_<record>/<model>/` and names the job `run1_<record>` in every
  model's files. Before, it was ignored.

Fixes:

- `scigantic-msa --version` works, as well as `scigantic-msa version`.

## 0.2.3

Changes you may notice:

- A FASTA file with several records needs `--complex` (one complex, at most
  20 chains) or `--batch` (one job per record, at most 100). Before, it
  silently became one complex.
- `--batch` writes each record to `<out>/<record>/<model>/`. A job the
  server cannot do no longer stops the others; the command ends with exit 1
  and names the failed jobs.
- Each model's files go in their own folder: `<out>/boltz/`, `<out>/af3/`
  and so on, from both the command line and `write_model`. The printed next
  command points there, with paths quoted for the shell.
- `run` refuses a typed argument that is a directory, that matches a file
  with an extension (`query` next to `query.fasta`), or that is under 10
  residues. Start a short peptide with `=`, as in `=MKVLAAG`.
- `--host`, `SCIGANTIC_MSA_COLABFOLD_URL` and `SCIGANTIC_MSA_NVIDIA_URL`
  must be `http://` or `https://` URLs. `ftp://`, `file://` and a missing
  scheme are refused before anything is sent.
- `SCIGANTIC_MSA_NO_REMOTE=1` still uses results already in the cache, and
  says nothing was sent when it blocks a search.
- `run` and `convert` print each chain's depth, and warn when it is below
  20 or only the query. A sequence of only `ACGTUN` gets a DNA/RNA warning.
- Written files follow the umask (usually 0644) instead of 0600.
- Job names in AlphaFold 3, OpenFold3 and Protenix JSON are file-safe:
  `my job` becomes `my_job`, and a name like `../x` cannot leave `--out`.
- New: `convert --uniprot FILE` (once per chain) and
  `MSA().from_a3m(..., uniprot=...)` add a UniProt alignment used for
  pairing.
- New: `python -m scigantic_msa` runs the same command.

Fixes:

- An unusable cache (read-only, or a path that is a file) gives one warning
  and the run goes on without it, instead of failing.
- Cache `.lock` files are removed after use, and a lock left by a crashed
  run no longer blocks.
- Protenix `pairing.a3m` headers carry the taxid for every row that has one,
  so Protenix pairs on the same species as the other models. Chains that
  share a species with only some of the others now get `pairing.a3m` too.
- AlphaFold 3 merge into an existing input writes the sequence in the
  canonical form, so it matches the MSA's first row.
- `Protenix.check` refuses a `sequences` entry or `proteinChain` that is not
  an object.
- A3M, Stockholm and FASTA with a UTF-8 BOM, CR or CRLF line endings, or a
  leading `#A3M#` line are read. FASTA as UTF-16, Latin-1 or gzip is read.
  Blank lines and indented headers in FASTA no longer crash.
- Input errors name the record and chain at fault, and never echo a long
  sequence back to the terminal.
- `run` checks that `--out` can be written before sending anything.
- `MSA().get()` gives clear errors for bytes, `None` or an empty chain, and
  suggests a list when a sequence contains `:`.
- Ctrl-C while waiting says that running the same command again resumes the
  job, and exits 130.
- `write_model(model, chains, out)` works as the README shows: `name` and
  `quiet` have defaults and `out` can be a str.

## 0.2.2

- The `version` command reports the installed version. 0.2.1 reported 0.2.0.

## 0.2.1

- `write_model` is exported from the package and writes any model's inputs
  with one call.
- `from_a3m` errors say whether the argument was read as a path or as text.

## 0.2.0

- First release on PyPI.
