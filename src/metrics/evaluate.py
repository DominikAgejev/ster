import torch
from torch.utils.data import DataLoader

def _apply_eval_activation(preds: torch.Tensor, color_space: str, pred_activation: str, eps: float):
    if color_space.lower() != "rgb":
        return preds
    if pred_activation == "sigmoid":
        return torch.sigmoid(preds).clamp(eps, 1 - eps)
    if pred_activation == "sigmoid_eps":
        s = torch.sigmoid(preds)
        return eps + (1 - 2 * eps) * s
    if pred_activation == "tanh01":
        y = 0.5 * (torch.tanh(preds) + 1.0)
        return y.clamp(eps, 1 - eps)
    return preds  # "none"

def evaluate(model, loader: DataLoader, device, criterion,
             color_space: str = "lab",
             pred_activation: str = "none",
             activation_eps: float = 1e-3):
    model.eval()
    totals = {"loss": 0.0, "de00": 0.0, "mse_lab": 0.0, "mse_rgb": 0.0}
    counts = {"n": 0, "mse_lab": 0, "mse_rgb": 0}

    with torch.no_grad():
        for b in loader:
            img = b["image"].to(device)
            meta = b["metadata"].to(device)
            y   = b["label"].to(device)

            preds = model(img, meta)
            preds = _apply_eval_activation(preds, color_space, pred_activation, activation_eps)

            loss_t, metrics = criterion(preds, y)

            bs = img.size(0)
            totals["loss"] += float(metrics["loss"]) * bs
            totals["de00"] += float(metrics["de00"]) * bs
            counts["n"]    += bs

            if metrics["mse_lab"] is not None:
                totals["mse_lab"] += float(metrics["mse_lab"]) * bs
                counts["mse_lab"] += bs
            if metrics["mse_rgb"] is not None:
                totals["mse_rgb"] += float(metrics["mse_rgb"]) * bs
                counts["mse_rgb"] += bs

    def _avg(total, n): return (total / n) if n > 0 else None
    return {
        "loss":    _avg(totals["loss"], counts["n"]),
        "de00":    _avg(totals["de00"], counts["n"]),
        "mse_lab": _avg(totals["mse_lab"], counts["mse_lab"]),
        "mse_rgb": _avg(totals["mse_rgb"], counts["mse_rgb"]),
        # Optional single “mse” for older code (current active space):
        "mse":     _avg(totals["mse_rgb"], counts["mse_rgb"]) if criterion.mse_space == "rgb"
                   else _avg(totals["mse_lab"], counts["mse_lab"]),
    }

def test_model(model, test_loader, criterion, color_space="rgb", pred_activation="sigmoid_eps", activation_eps=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metrics = evaluate(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        color_space=color_space,
        pred_activation=pred_activation,
        activation_eps=activation_eps
    )

    print("[Test] Metrics:", metrics)
    return metrics
