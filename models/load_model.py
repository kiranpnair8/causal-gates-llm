import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.gates import ResidualGate

from models.intervention import (
    CURRENT_INTERVENTION,
    should_intervene,
    apply_token_intervention,
    apply_module_intervention,
)


def make_gated_forward(layer, layer_idx):
    def gated_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states_norm = layer.input_layernorm(hidden_states)
        attn_outputs = layer.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        attn_output = attn_outputs[0]
        if should_intervene(layer_idx, "attn"):

            if CURRENT_INTERVENTION.mode == "token":
                attn_output = apply_token_intervention(
                    attn_output,
                    CURRENT_INTERVENTION.token_idx,
                )

            elif CURRENT_INTERVENTION.mode == "module":
                attn_output = apply_module_intervention(
                    attn_output
                )
        gated_attn, attn_gate_values = layer.attn_gate(residual, attn_output)
        hidden_states = residual + gated_attn

        residual = hidden_states
        hidden_states_norm = layer.post_attention_layernorm(hidden_states)
        mlp_output = layer.mlp(hidden_states_norm)
        if should_intervene(layer_idx, "mlp"):

            if CURRENT_INTERVENTION.mode == "token":
                mlp_output = apply_token_intervention(
                    mlp_output,
                    CURRENT_INTERVENTION.token_idx,
                )

            elif CURRENT_INTERVENTION.mode == "module":
                mlp_output = apply_module_intervention(
                    mlp_output
                )

        gated_mlp, mlp_gate_values = layer.mlp_gate(residual, mlp_output)
        hidden_states = residual + gated_mlp

        layer.last_attn_gate = attn_gate_values
        layer.last_mlp_gate = mlp_gate_values

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        if use_cache:
            outputs += (attn_outputs[-1],)

        return outputs

    return gated_forward


def add_gates_to_tinyllama(model, config):
    hidden_size = model.config.hidden_size

    for layer_idx, layer in enumerate(model.model.layers):
        device = next(layer.parameters()).device

        layer.attn_gate = ResidualGate(hidden_size, config).to(device=device)
        layer.mlp_gate = ResidualGate(hidden_size, config).to(device=device)

        layer.forward = make_gated_forward(layer, layer_idx)

    return model


def freeze_backbone_train_gates(model):
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        for param in layer.attn_gate.parameters():
            param.requires_grad = True
        for param in layer.mlp_gate.parameters():
            param.requires_grad = True

    return model


def load_tinyllama_with_gates(config):
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = add_gates_to_tinyllama(model, config)
    model = freeze_backbone_train_gates(model)

    return model, tokenizer