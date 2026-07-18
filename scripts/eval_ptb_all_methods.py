"""Compatibility wrapper for Penn Treebank all-method evaluation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_all_methods import run


if __name__ == "__main__":
    run(
        default_datasets=("ptb",),
        default_output_csv="outputs/ptb_all_methods.csv",
        description="Evaluate all methods on Penn Treebank only.",
    )
