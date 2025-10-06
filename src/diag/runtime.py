# src/diag/runtime.py
from __future__ import annotations
import json, time
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional
import torch

# ------------------------------ internals ------------------------------------
@contextmanager
def _preserve_model_mode(model):
    """Restore model.train()/eval() to whatever it was before the diag."""
    was_training = model.training
    try:
        yield
    finally:
        model.train(was_training)

def _append_jsonl(diag_dir: Optional[str], filename: str, record: Dict[str, Any]) -> None:
    if not diag_dir:
        return
    p = Path(diag_dir) / "diag"
    p.mkdir(parents=True, exist_ok=True)
    with (p / filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    # Defensive: keep this no-grad and cheap
    with torch.no_grad():
        isnan = torch.isnan(t).sum().item() if t.is_floating_point() else 0
        isinf = torch.isinf(t).sum().item() if t.is_floating_point() else 0
        # Guard min/max on empty tensors
        tmin = float(t.min().item()) if t.numel() > 0 else None
        tmax = float(t.max().item()) if t.numel() > 0 else None
        return {
            "min": tmin,
            "max": tmax,
            "nan": int(isnan),
            "inf": int(isinf),
            "shape": tuple(int(x) for x in t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
        }

def _batch_stats(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in ("image", "label", "metadata"):
        v = batch.get(k, None) if isinstance(batch, dict) else None
        if isinstance(v, torch.Tensor):
            out[k] = _tensor_stats(v)
    return out

def _forward_check(model, batch: Dict[str, Any]) -> Optional[str]:
    """
    Side-effect-free forward:
      - no grad
      - model eval within a context that restores the original mode
      - returns an error string on anomaly; returns None if everything is ok
    """
    try:
        x = batch["image"]
        meta = batch.get("metadata", None) if isinstance(batch, dict) else None

        # batch dimension sanity
        if not isinstance(x, torch.Tensor):
            return "image tensor missing or wrong type"
        if meta is not None and isinstance(meta, torch.Tensor) and meta.shape[0] != x.shape[0]:
            return f"metadata batch size mismatch: image B={x.shape[0]} vs meta B={meta.shape[0]}"

        with _preserve_model_mode(model), torch.no_grad():
            model.eval()
            if meta is not None and isinstance(meta, torch.Tensor):
                out = model(x.detach(), meta.detach())
            else:
                out = model(x.detach())

            if not isinstance(out, torch.Tensor):
                # allow tuple/list outputs but check tensors inside
                if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                    y = out[0]
                else:
                    return "model output is not a tensor (or first element of tuple/list)"
            else:
                y = out

            if torch.isnan(y).any():
                return "NaN in model output"
            if torch.isinf(y).any():
                return "Inf in model output"

    except Exception as e:
        return f"{type(e).__name__}: {e}"

    return None
# -----------------------------------------------------------------------------

def maybe_step_diag(
    *,
    model,
    batch: Dict[str, Any],
    step_in_epoch: int,
    stage: str,                        # "train" | "val" | "test"
    diag_every_steps: int = 0,         # -1=off, 0=first step only, N>0 = every N steps
    write_stats_dir: Optional[str] = None  # e.g. cfg.ckpt.outdir; None = don't write
) -> None:
    """
    Quiet + safe diagnostics. Emits NOTHING unless an anomaly is detected.
    Optionally writes a single JSONL line with batch stats when it runs.
    """
    if diag_every_steps == -1:
        return
    do_check = (step_in_epoch == 0) if diag_every_steps == 0 else (step_in_epoch % diag_every_steps == 0)
    if not do_check:
        return

    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    # write compact batch stats (no prints)
    if write_stats_dir:
        stats = _batch_stats(batch)
        if stats:
            _append_jsonl(write_stats_dir, "batch_stats.jsonl", {
                "ts": ts, "event": "batch_stats", "stage": stage, **stats
            })

    # run forward check (prints ONLY if a real problem is found)
    err = _forward_check(model, batch)
    if err:
        print(f"[diag][ALERT] forward check ({stage}) failed: {err}")
