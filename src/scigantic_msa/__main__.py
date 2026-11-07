"""python -m scigantic_msa, for when the scigantic-msa script is not on PATH."""
from .cli import main

raise SystemExit(main())
