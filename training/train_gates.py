import torch
import random
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm

from models.load_model import load_tinyllama_with_gates
from models.intervention import set_intervention, clear_intervention
from training.losses import gate_sparsity_loss

from utils.config import load_config

config = load_config("utils/gate.yaml")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_fn(example, tokenizer, max_length=config["data"]["max_length"]):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

## Intervention Helper Functions

def sample_intervention(model):
    num_layers = len(model.model.layers)
    layer_idx = random.randint(0, num_layers - 1)
    module = random.choice(["attn", "mlp"])
    return layer_idx, module

def get_gate_value(model, layer_idx, module):
    layer = model.model.layers[layer_idx]

    if module == "attn":
        gate_values = layer.last_attn_gate
    else:
        gate_values = layer.last_mlp_gate

    return gate_values.float().mean()

def get_module_gate_value(model, layer_idx, module):
    layer = model.model.layers[layer_idx]

    if module == "attn":
        gate_values = layer.last_attn_gate
    else:
        gate_values = layer.last_mlp_gate

    return gate_values.float().mean()

def get_all_module_gates(model):
    gates = []

    for layer in model.model.layers:
        gates.append(layer.attn_gate.gate_values_scalar())
        gates.append(layer.mlp_gate.gate_values_scalar())

    return torch.stack(gates)


def get_module_names(model):
    names = []

    for layer_idx in range(len(model.model.layers)):
        names.append(f"L{layer_idx:02d}.attn")
        names.append(f"L{layer_idx:02d}.mlp")

    return names


def print_gate_rankings(model, top_k=10):
    gates = get_all_module_gates(model).detach().float().cpu()
    module_names = get_module_names(model)
    ranked = sorted(
        zip(module_names, gates.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nTop learned gates:")
    for name, value in ranked[:top_k]:
        print(f"{name:8s} gate={value:.4f}")

    print("\nLowest learned gates:")
    for name, value in ranked[-top_k:]:
        print(f"{name:8s} gate={value:.4f}")

@torch.no_grad()
def compute_module_delta(model, batch, original_logits, layer_idx, module):
    set_intervention(
        layer_idx=layer_idx,
        module=module,
        mode="module",
    )

    intervened_outputs = model(**batch)
    intervened_logits = intervened_outputs.logits

    clear_intervention()

    return compute_kl_delta(
        original_logits,
        intervened_logits,
    )

@torch.no_grad()
def compute_all_module_deltas(model, batch, original_logits):
    deltas = []

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
            )

            deltas.append(delta)

    return torch.stack(deltas)

def causal_ranking_loss(delta_a, delta_b, gate_a, gate_b, margin=0.05):
    if delta_a > delta_b:
        return torch.relu(margin - (gate_a - gate_b))
    else:
        return torch.relu(margin - (gate_b - gate_a))


def compute_kl_delta(original_logits, intervened_logits):
    # use last-token distribution for now
    original_last = original_logits[:, -1, :].float()
    intervened_last = intervened_logits[:, -1, :].float()

    log_p = F.log_softmax(original_last, dim=-1)
    q = F.softmax(intervened_last, dim=-1)

    kl = F.kl_div(log_p, q, reduction="batchmean")

    return kl.detach()


def compute_gate_target_corr(gates, targets):
    gates = gates.detach().float()
    targets = targets.detach().float()

    centered_gates = gates - gates.mean()
    centered_targets = targets - targets.mean()
    denom = centered_gates.norm() * centered_targets.norm()

    if denom.item() <= 1e-8:
        return 0.0

    return float((centered_gates * centered_targets).sum() / denom)

## Intervention Helper Functions end


def main():
    seed = int(config.get("system", {}).get("seed", 42))
    set_seed(seed)
    print(f"Using seed={seed}")

    model, tokenizer = load_tinyllama_with_gates(config)

    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["dataset_config"],
        split=config["data"]["split"],
    )
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)

    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    loader = DataLoader(
        tokenized,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
    )

    model.train()

    gate_params = []
    for layer in model.model.layers:
        gate_params += list(layer.attn_gate.parameters())
        gate_params += list(layer.mlp_gate.parameters())

    optimizer = torch.optim.AdamW(
        gate_params,
        lr=float(config["training"]["learning_rate"]),
        eps=float(config["training"].get("adam_eps", 1e-6)),
    )

    lambda_lm = float(config["loss"].get("lambda_lm", 0.0))
    lambda_sparsity = config["loss"]["lambda_sparsity"]
    lambda_causal = float(config["causal"]["lambda_causal"])
    target_floor = float(config["causal"].get("target_floor", 0.0))
    grad_accum_steps = config["training"]["grad_accum_steps"]
    max_steps = int(config["training"].get("max_steps", 1000))
    log_every = int(config.get("logging", {}).get("log_every", 50))

    optimizer.zero_grad()

    avg_gate_grad_norm = 0.0

    for step, batch in enumerate(tqdm(loader)):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        clear_intervention()

        outputs = model(**batch)
        lm_loss = outputs.loss
        original_logits = outputs.logits.detach()

        deltas = compute_all_module_deltas(
            model,
            batch,
            original_logits,
        )

        normalized_targets = deltas / (deltas.max() + 1e-8)
        targets = target_floor + (1.0 - target_floor) * normalized_targets
        targets = targets.to(model.device)

        gates = get_all_module_gates(model)

        causal_loss = torch.nn.functional.mse_loss(
            gates.float(),
            targets.float(),
        )

        sparse_loss = gate_sparsity_loss(model)

        loss = (
            lambda_lm * lm_loss
            + lambda_sparsity * sparse_loss
            + lambda_causal * causal_loss
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf detected")
            print("lm_loss:", lm_loss)
            print("sparse_loss:", sparse_loss)
            print("causal_loss:", causal_loss)
            break

        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            total_grad_norm = 0.0
            num_grads = 0

            for layer in model.model.layers:
                for gate in [layer.attn_gate, layer.mlp_gate]:
                    if gate.gate_logit.grad is not None:
                        grad_norm = gate.gate_logit.grad.abs().item()
                        total_grad_norm += grad_norm
                        num_grads += 1

            avg_gate_grad_norm = total_grad_norm / max(num_grads, 1)

            torch.nn.utils.clip_grad_norm_(
                gate_params,
                max_norm=float(config["training"]["max_grad_norm"]),
            )

            optimizer.step()
            optimizer.zero_grad()

        if step % log_every == 0:
            gate_range = float(gates.max() - gates.min())
            gate_target_corr = compute_gate_target_corr(gates, targets)

            print(
                f"step={step} "
                f"lm_loss={lm_loss.item():.4f} "
                f"sparsity={float(sparse_loss):.4f} "
                f"causal_loss={float(causal_loss):.4f} "
                f"delta_max={float(deltas.max()):.4f} "
                f"delta_mean={float(deltas.mean()):.4f} "
                f"target_mean={float(targets.mean()):.4f} "
                f"gate_mean={float(gates.mean()):.4f} "
                f"gate_min={float(gates.min()):.4f} "
                f"gate_max={float(gates.max()):.4f} "
                f"gate_range={gate_range:.4f} "
                f"gate_target_corr={gate_target_corr:.4f} "
                f"grad_norm={avg_gate_grad_norm:.8f}"
            )

        if step >= max_steps:
            break

    print_gate_rankings(model)

    model.save_pretrained("outputs/tinyllama_gated")
    tokenizer.save_pretrained("outputs/tinyllama_gated")


if __name__ == "__main__":
    main()