# Changelog

## 0.2.6

Changes you may notice:

- AlphaFold 3: every paired row now sits at the same row index in every
  chain that has it. Before, a chain without the first pairs' species and
  with too few rows of its own put its later paired rows right after the
  query, against other chains' rows of other species, while the notice
  said they were paired. Where a chain has no row of its own left to stand
  in, it gets a gap row, as Boltz does; the pairs are put in the order that
  needs the fewest, and the notice says how many each chain got.
- AlphaFold 3: rows that differ only in insertions are kept. AlphaFold 3
  reads a user MSA without removing repeats, so they are not repeats to it.
- Every MSA file is named, or put in a folder named, for its content. A
  second alignment of the same sequence in one `--out` (`convert` with
  another file and another `--name`, or `run` with `--host` after the
  public server) no longer rewrites the first job's MSA for Boltz, Chai-1,
  OpenFold3 and Protenix. Identical files are still shared. New paths:
  Boltz `<sha256>.<hash>.a3m` and `.csv`, Chai-1
  `msas/unpaired-<hash>/` for a job without pairing (as `paired-<hash>/`
  before), OpenFold3 `main/<sha256>/<hash>/`, Protenix
  `<sha256>/<hash>/`, AlphaFold 3 `inline=False`
  `<sha256>.<hash>.unpaired.a3m` and `.paired.a3m`. The printed commands
  and returned paths point there.
- Boltz gets at most 8191 pair keys, the most it pairs besides the query.
  Before, the rows of a pair past that were dropped by Boltz; now they are
  unpaired rows.

Fixes:

- `https://api.colabfold.com//.` and other paths with repeated slashes
  that the server reads as `/` are the public server, with its warning, cap
  and cache.
- The pairing notice counts chains, not distinct sequences, and names them
  in order: A, B, A, C reads "across 4 chains", not "across 3 chains", and
  "chains A, B, C, D", not "chains A, C, B, D".

## 0.2.5

Changes you may notice:

- Species shared by only some chains of a complex are paired. Before,
  AlphaFold 3, Boltz, Chai-1, ColabFold and OpenFold3 paired only species
  present in every chain that had species, so one chain from another
  organism removed all pairing for the others (Protenix already paired
  them). Rows shared by more chains come first, and the notice says which
  chains paired and how many rows each set got. `Pairing.rows` holds `None`
  for a chain without a row of that species.
- AlphaFold 3: where a chain has no row for a paired species, one of its
  own unpaired rows stands in, so the rows after it stay aligned. It never
  gets a gap row.
- Writing a second complex into the same folder (A+B, then A+C, or then A
  alone) no longer rewrites the first complex's MSA files and breaks its
  pairing. Boltz CSVs with pair keys are named `<sha256>.<hash>.csv`.
  Chai-1 files of a complex with paired rows go in
  `<out>/chai/msas/paired-<hash>/`, which `Chai.prepare` returns and the
  printed command uses. OpenFold3's paired folder and AlphaFold 3's
  `inline=False` unpaired file with paired rows are named for their
  content too. Files without pairing keep the plain name and are shared.
- `run` refuses what the model's writer would refuse before it sends
  anything: a Protenix chain with `U`, `B`, `Z`, `J` or `O`, and Chai-1
  without pyarrow. Before, the sequence went to the server first. In a
  `--batch`, only the refused job is skipped, and the closing message says
  its input must change instead of asking to run again.
- A server URL with a non-ASCII character is refused, with its ASCII
  spelling when it has one. Before, `run` said it was sending, then failed
  with a latin-1 error.

Fixes:

- `https://api.colabfold.com/.`, `/./` and other paths of only dot segments
  are the public server, with its warning, cap and cache.
- An attribution file with an odd entry no longer crashes the write after
  the model's files are written. Replacing an attribution file that cannot
  be read prints a warning.
- `--batch` gives records whose names differ only in case or Unicode form
  (`ß` and `SS`, a Hangul syllable and its jamo) their own folders. On macOS
  the second one overwrote the first.
- `COM¹`, `LPT³` and the like, and `COM0` and `LPT0`, get a `_` in front as
  Windows reserves them, and a name cut to 200 bytes no longer ends in `.`.

## 0.2.4

Changes you may notice:

- A complex with a chain that has no species information (environmental or
  clustered hits only) keeps pairing for the other chains with AlphaFold 3,
  Boltz, Chai-1, ColabFold and OpenFold3. Before, they silently lost all
  pairing. Each model gets the species-less chain in the form its own code
  expects, and the notice names that chain.
- Pairing notices name chains by id (`chain C`), not by their position
  among the distinct sequences (`chain(s) 3`). `Pairing.rows` holds `None`
  for every row of a chain without species, and `pair_by_species` takes
  `names=` for its message.
- `scigantic-msa-attribution.json` keeps one entry per write under
  `outputs`, and `sources` lists every distinct credit. Before, a second
  write into the same folder (two models from Python, or a second `convert`
  or `run` with another `--name`) replaced the earlier credit. The top-level
  `chains`, `files` and `notes` keys are now inside each entry. A file from
  0.2.3 is kept as the first entry.
- `--name` with `--batch` is a prefix for every job: `--name run1` writes
  `<out>/run1_<record>/<model>/`. Before, it was ignored.
- `SCIGANTIC_MSA_NO_REMOTE` blocks on any value but `0`, `false`, `no` or
  `off`. Before, `on` or `y` still sent the sequence.
- Any spelling of the public server's URL (another case, `:443`, a trailing
  slash, spaces, or `http://`) is the public server, reached over https, and
  gets its warning. `--host` refuses a URL with a space, query or fragment.
- One `run` sends at most 100 new sequences to the public server.
- Protenix: a protein letter other than the 20 standard amino acids and `X`
  (`U`, `B`, `Z`, `J`, `O`) is refused before any file is written, and
  `Protenix.check` reports it. Protenix 2.0.0 stops on it with `KeyError`.
- Job names in file names are cut to 200 bytes, not 100 characters, and
  Windows reserved names (`CON`, `nul.txt`) get a `_` in front.

Fixes:

- `scigantic-msa --version` works, as well as `scigantic-msa version`.
- Parallel `Chai.prepare` calls into one folder no longer fail or leave a
  corrupt parquet file.
- Chai-1 gets at most 8191 paired rows, the most it reads besides the query.
- `run` checks the model's folders under `--out`, and each job's folder in a
  batch, before sending anything.
- `--server nvidia` without `NVIDIA_API_KEY` stops before asking to send.
- A result download that is not a tar.gz fails only that job of a batch.
- The 10-residue error no longer echoes a long sequence.
- Depth warnings in a batch name the job.
- `write_model` with `name=""` returns the files it wrote.
- `convert --uniprot ""` is refused instead of ignored.
- A model that refuses the input leaves no empty folders behind.
- The time estimate for the public server says "or more", and lengths that
  are all equal read "length 60 each".
- Tests: an opt-in test runs Protenix's own JSON parser and MSA
  featurisation on the files, with `SCIGANTIC_MSA_TEST_PROTENIX_PYTHON`.

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
