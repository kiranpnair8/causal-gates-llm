import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class InterventionConfig:
    enabled: bool = False
    layer_idx: Optional[int] = None
    module: Optional[str] = None  # "attn" or "mlp"
    mode: str = "token"
    token_idx: Optional[int] = None


# global intervention state
CURRENT_INTERVENTION = InterventionConfig()


def set_intervention(layer_idx: int, module: str, mode="token", token_idx=None):
    if module not in ["attn", "mlp"]:
        raise ValueError("module must be either 'attn' or 'mlp'")

    CURRENT_INTERVENTION.enabled = True
    CURRENT_INTERVENTION.layer_idx = layer_idx
    CURRENT_INTERVENTION.module = module
    CURRENT_INTERVENTION.mode = mode
    CURRENT_INTERVENTION.token_idx = token_idx


def clear_intervention():
    CURRENT_INTERVENTION.enabled = False
    CURRENT_INTERVENTION.layer_idx = None
    CURRENT_INTERVENTION.module = None
    CURRENT_INTERVENTION.token_idx = None


def should_intervene(layer_idx: int, module: str) -> bool:
    return (
        CURRENT_INTERVENTION.enabled
        and CURRENT_INTERVENTION.layer_idx == layer_idx
        and CURRENT_INTERVENTION.module == module
    )


def apply_token_intervention(module_output, token_idx: int):
    """
    Zero out the module output for one token position.

    module_output shape:
        [batch, seq_len, hidden_size]
    """

    if token_idx < 0 or token_idx >= module_output.shape[1]:
        return module_output

    module_output = module_output.clone()
    module_output[:, token_idx, :] = 0.0

    return module_output

def apply_module_intervention(module_output):
    """
    Zero out the ENTIRE module output.

    Shape:
        [batch, seq_len, hidden_size]
    """

    return torch.zeros_like(module_output)