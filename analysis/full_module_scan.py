import csv
import torch
import torch.nn.functional as F

from utils.config import load_config
from models.load_model import load_tinyllama_with_gates
from models.intervention import set_intervention, clear_intervention


def compute_kl_delta(original_logits, intervened_logits):
    original_last = original_logits[:, -1, :].float()
    intervened_last = intervened_logits[:, -1, :].float()

    log_p = F.log_softmax(original_last, dim=-1)
    q = F.softmax(intervened_last, dim=-1)

    return F.kl_div(log_p, q, reduction="batchmean")


@torch.no_grad()
def main():
    config = load_config("utils/gate.yaml")

    model, tokenizer = load_tinyllama_with_gates(config)
    model.eval()

    text = (
        "The capital of France is Paris. "
        "France is a country in Europe. "
        "The Eiffel Tower is located in"
    )

    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config["data"]["max_length"],
    ).to(model.device)

    clear_intervention()
    original_outputs = model(**batch)
    original_logits = original_outputs.logits.detach()

    results = []

    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        for module in ["attn", "mlp"]:
            set_intervention(
                layer_idx=layer_idx,
                module=module,
                mode="module",
            )

            intervened_outputs = model(**batch)
            intervened_logits = intervened_outputs.logits.detach()

            clear_intervention()

            delta = compute_kl_delta(
                original_logits,
                intervened_logits,
            ).item()

            layer = model.model.layers[layer_idx]

            if module == "attn":
                gate_value = layer.last_attn_gate.float().mean().item()
            else:
                gate_value = layer.last_mlp_gate.float().mean().item()

            results.append(
                {
                    "layer": layer_idx,
                    "module": module,
                    "delta": delta,
                    "gate": gate_value,
                }
            )

    results = sorted(
        results,
        key=lambda x: x["delta"],
        reverse=True,
    )

    print("\nTop causal modules:")
    for row in results[:10]:
        print(
            f"L={row['layer']:02d} "
            f"M={row['module']:4s} "
            f"delta={row['delta']:.6f} "
            f"gate={row['gate']:.4f}"
        )

    print("\nLeast causal modules:")
    for row in results[-10:]:
        print(
            f"L={row['layer']:02d} "
            f"M={row['module']:4s} "
            f"delta={row['delta']:.6f} "
            f"gate={row['gate']:.4f}"
        )

    output_path = "outputs/full_module_scan.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "module", "delta", "gate"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved scan to: {output_path}")


if __name__ == "__main__":
    main()