"""Validate the independent all-open KL scan shared by Figures 2 and 7.

These scores are not the dynamic, soft-gated targets used during training.
"""

import csv
import math
import re
from pathlib import Path


CANONICAL_CSV = "outputs/canonical_allopen_kl_tinyllama.csv"
FIELDS = ("module", "layer", "module_type", "mean_kl", "kl_rank")
MODULE_RE = re.compile(r"^L(\d{2})\.(attn|mlp)$")


def validate_rows(rows):
    expected = {f"L{layer:02d}.{kind}" for layer in range(22) for kind in ("attn", "mlp")}
    names = [row["module"] for row in rows]
    if len(rows) != 44 or set(names) != expected or len(set(names)) != 44:
        raise ValueError("Canonical scan must contain each of the 44 TinyLlama modules exactly once")

    ranks = [row["kl_rank"] for row in rows]
    if set(ranks) != set(range(1, 45)):
        raise ValueError("Canonical KL ranks must be exactly 1 through 44")

    for row in rows:
        match = MODULE_RE.fullmatch(row["module"])
        if match is None or int(match[1]) != row["layer"] or match[2] != row["module_type"]:
            raise ValueError(f"Module metadata mismatch: {row['module']}")
        if not math.isfinite(row["mean_kl"]) or row["mean_kl"] < 0:
            raise ValueError(f"Invalid mean KL for {row['module']}")

    ordered = sorted(rows, key=lambda row: row["kl_rank"])
    by_score = sorted(rows, key=lambda row: (-row["mean_kl"], row["module"]))
    if [row["module"] for row in ordered] != [row["module"] for row in by_score]:
        raise ValueError("KL ranks do not match descending mean KL (module name breaks ties)")
    return ordered


def load_canonical_kl(path=CANONICAL_CSV):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain: {', '.join(FIELDS)}")
        rows = [
            {
                "module": row["module"],
                "layer": int(row["layer"]),
                "module_type": row["module_type"],
                "mean_kl": float(row["mean_kl"]),
                "kl_rank": int(row["kl_rank"]),
            }
            for row in reader
        ]
    return validate_rows(rows)
