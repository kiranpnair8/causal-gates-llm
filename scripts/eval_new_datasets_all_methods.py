"""Compatibility wrapper for the unified all-method evaluator."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_all_methods import run


if __name__ == "__main__":
    run(
        default_datasets=("openbookqa", "winogrande", "lambada", "c4"),
        default_output_csv="outputs/new_dataset_all_methods.csv",
        description="Evaluate all methods on OpenBookQA, WinoGrande, LAMBADA, and C4.",
    )
