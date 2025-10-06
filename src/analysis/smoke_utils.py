# src/analysis/smoke_utils.py
from __future__ import annotations
import json, os, hashlib, time
from typing import Dict, Any

_KEEP_KEYS_FOR_SIG = {
    "model","backbone","features","color_space",
    "epochs","batch_size","val_split","group_split",
    "excluded_folders","mse_space","mse_weight_start","mse_weight_epochs",
}

def _now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def config_signature(cfg: Dict[str, Any]) -> str:
    j = json.dumps({k: cfg.get(k) for k in sorted(_KEEP_KEYS_FOR_SIG)}, sort_keys=True)
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:12]

def write_summary_and_check(*, run_cfg: Dict[str, Any], sizes: Dict[str,int],
                            best_val_de00: float, best_epoch: int,
                            test_metrics: Dict[str, float] | None,
                            smoke_cfg) -> Dict[str, Any]:
    os.makedirs(smoke_cfg.baseline_dir, exist_ok=True)
    baseline_path = os.path.join(smoke_cfg.baseline_dir, "smoke_baseline.json")
    result_path   = os.path.join(smoke_cfg.baseline_dir, smoke_cfg.result_filename)

    sig = config_signature(run_cfg)
    try:
        with open(baseline_path, "r") as f:
            baselines = json.load(f)
    except Exception:
        baselines = {}

    prev = baselines.get(sig)
    status, reason = "ok", ""
    if prev is not None:
        tol = max(float(smoke_cfg.abs_tol), float(smoke_cfg.rel_tol) * float(prev))
        if best_val_de00 > float(prev) + tol and not smoke_cfg.update_baseline:
            status = "regression"
            reason = f"ΔE00 {best_val_de00:.3f} > baseline {prev:.3f} + tol {tol:.3f}"
        else:
            status, reason = "ok", f"meets baseline {prev:.3f} ± tol {tol:.3f}"
    else:
        status, reason = "new-baseline", "no previous baseline"

    if smoke_cfg.update_baseline or prev is None:
        baselines[sig] = float(best_val_de00)
        with open(baseline_path, "w") as f:
            json.dump(baselines, f, indent=2, sort_keys=True)

    summary = {
        "time": _now(),
        "config_signature": sig,
        "config": {k: run_cfg.get(k) for k in sorted(_KEEP_KEYS_FOR_SIG)},
        "sizes": sizes,
        "best": {"val_de00": float(best_val_de00), "epoch": int(best_epoch)},
        "test": test_metrics,
        "baseline_status": status,
        "baseline_reason": reason,
        "baseline_val_de00": prev,
    }
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[smoke] status={status}  best_val_de00={best_val_de00:.3f}  -> {result_path}")
    if status == "regression":
        print(f"[smoke][WARN] {reason}")
    return summary
