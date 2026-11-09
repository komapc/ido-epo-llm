"""Repository layout. The repo is cloned into the apertium-dev workspace as
`llm/`, so the Apertium pair is a sibling directory (WORKSPACE)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # this repo
WORKSPACE = ROOT.parent                          # apertium-dev/
CORPUS = ROOT / "corpus"
CACHE = CORPUS / "cache"                         # downloaded dumps (gitignored)
OUT = CORPUS / "out"                             # built pairs and splits (gitignored)
DATASETS = ROOT / "datasets"                     # published splits, one dir per dataset

# The split the published model komapc/nllb-ido-epo-600m was trained and scored on.
# Frozen: never rebuild it in place; a new dataset gets a new directory.
DEFAULT_DATASET = "v1-io-eo"


def dataset_dir(name: str = DEFAULT_DATASET) -> Path:
    return DATASETS / name
