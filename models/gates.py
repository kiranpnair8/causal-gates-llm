import torch
import torch.nn as nn

from utils.config import load_config

config = load_config("utils/gate.yaml")

class ResidualGate(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()

        init_bias = float(config["gate"]["init_bias"])

        # one learnable scalar logit per module
        self.gate_logit = nn.Parameter(
            torch.tensor(init_bias, dtype=torch.float32)
        )

    def gate_values_scalar(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, hidden_states, module_output):
        gate_value = torch.sigmoid(self.gate_logit)

        gate_value = gate_value.to(
            device=module_output.device,
            dtype=module_output.dtype,
        )

        gated_output = module_output * gate_value

        # return shape similar to old gate: [B, S, 1]
        gate_values = torch.ones(
            module_output.shape[0],
            module_output.shape[1],
            1,
            device=module_output.device,
            dtype=module_output.dtype,
        ) * gate_value

        return gated_output, gate_values