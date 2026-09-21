"""Run the independent all-open Oracle-KL scan for Figures 2 and 7."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from canonical_kl import CANONICAL_CSV, FIELDS, load_canonical_kl, validate_rows
from eval_oracle_kl_baseline import (
    build_wikitext_loader,
    compute_oracle_kl_ranking,
    get_module_names,
    load_gate_checkpoint,
    load_tinyllama_with_gates,
    load_config,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--output-csv", default=CANONICAL_CSV)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output = Path(args.output_csv)
    if output.exists():
        raise FileExistsError(f"Canonical scan already exists; refusing to overwrite {output}")
    checkpoint = Path(args.checkpoint_dir)
    if not any((checkpoint / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"Trained CausalGate checkpoint not found in {checkpoint}")

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, checkpoint)
    names = get_module_names(model)
    if len(names) != 44:
        raise ValueError(f"Expected 44 attention/MLP modules, found {len(names)}")

    loader = build_wikitext_loader(config, tokenizer, "test", 32)
    if len(loader.dataset) != 32 or loader.batch_size != 1:
        raise ValueError("Canonical scan requires exactly 32 filtered examples at batch size 1")
    # The existing Oracle routine sets every gate to sigmoid(20), calls eval(),
    # intervenes on one full residual branch at a time, and averages last-token KL.
    ranking = sorted(
        compute_oracle_kl_ranking(model, loader, names),
        key=lambda item: (-item["delta"], item["module"]),
    )
    rows = []
    for rank, item in enumerate(ranking, start=1):
        layer, kind = item["module"].split(".")
        rows.append({
            "module": item["module"],
            "layer": int(layer[1:]),
            "module_type": kind,
            "mean_kl": item["delta"],
            "kl_rank": rank,
        })
    validate_rows(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "mean_kl": format(row["mean_kl"], ".17g")})
    reloaded = load_canonical_kl(output)
    if reloaded != rows:
        raise AssertionError("Canonical CSV failed full-precision round-trip validation")

    print(f"Saved {output}; seed={args.seed}; 32 WikiText-2 test examples; 44 all-open modules")
    print("Independent all-open intervention analysis; not dynamic CausalGate training targets.")
    print("Top 10 modules by mean KL:")
    for row in reloaded[:10]:
        print(f"{row['kl_rank']:2d} {row['module']:8s} {row['mean_kl']:.17g}")
    l19 = next(row for row in reloaded if row["module"] == "L19.mlp")
    print(f"L19.mlp mean_kl={l19['mean_kl']:.17g} kl_rank={l19['kl_rank']}")


if __name__ == "__main__":
    main()
