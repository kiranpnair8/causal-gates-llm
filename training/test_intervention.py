import torch
import torch.nn.functional as F

from utils.config import load_config
from models.load_model import load_tinyllama_with_gates
from models.intervention import set_intervention, clear_intervention


@torch.no_grad()
def main():
    config = load_config("utils/gate.yaml")
    model, tokenizer = load_tinyllama_with_gates(config)
    model.eval()

    text = "The capital of France is"
    batch = tokenizer(text, return_tensors="pt").to(model.device)

    clear_intervention()
    original = model(**batch).logits[:, -1, :]

    set_intervention(
        layer_idx=10,
        module="mlp",
        token_idx=3,
    )

    intervened = model(**batch).logits[:, -1, :]
    clear_intervention()

    p = F.log_softmax(original.float(), dim=-1)
    q = F.softmax(intervened.float(), dim=-1)

    kl = F.kl_div(p, q, reduction="batchmean")

    print("KL causal delta:", kl.item())


if __name__ == "__main__":
    main()