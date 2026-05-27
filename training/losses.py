def gate_sparsity_loss(model):
    losses = []

    for layer in model.model.layers:
        if hasattr(layer, "last_attn_gate"):
            losses.append(layer.last_attn_gate.float().pow(2).mean())

        if hasattr(layer, "last_mlp_gate"):
            losses.append(layer.last_mlp_gate.float().pow(2).mean())

    if len(losses) == 0:
        return 0.0

    return sum(losses) / len(losses)