# src/diag/hooks.py
from __future__ import annotations
import torch


def grad_stats(model, step: int, every: int = 100):
    """
    Print aggregate grad norms occasionally to catch vanishing/exploding/NaN.
    """
    if every <= 0 or (step % every):
        return
    norms = []
    has_nan = False
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if torch.isnan(g).any() or torch.isinf(g).any():
            print(f"[grad][naninf] {name}")
            has_nan = True
        norms.append(float(g.norm().item()))
    if norms:
        norms_sorted = sorted(norms)
        mid = norms_sorted[len(norms_sorted) // 2]
        print(f"[grad] step={step} L2 min/med/max: "
              f"{min(norms):.3e}/{mid:.3e}/{max(norms):.3e} (nan/inf={has_nan})")
