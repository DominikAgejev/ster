# src/engine/sweep/exec.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import os, json, time

from .utils import (
    build_experiment_cmd, state_dir_for_config, marker_path, write_marker, run_subprocess
)
from ..split_utils import ensure_auto_splits_for_run

def execute_runs(
    config_path: str,
    runs: List[Tuple[str, Dict[str, Any]]],
    *,
    start_index: int = 1,
    stop_index:   int | None = None,
    split_root:   str = "./splits/sweep",
    force_auto_test: bool = False,
    skip_existing: bool = False,
    verbose: int = 1,
) -> Tuple[int, int, int]:
    """
    Execute a slice of runs, optionally provisioning per-filter AUTO splits,
    and optionally skipping any run that already has a SUCCESS marker.
    Returns: (success_count, fail_count, skip_count)
    """
    if not runs:
        return (0, 0, 0)

    n = len(runs)
    lo = max(1, int(start_index))
    hi = int(stop_index or n)
    hi = max(lo, min(hi, n))

    state_dir = state_dir_for_config(config_path)
    success = fail = skip = 0

    for idx in range(lo, hi + 1):
        run_name, run_cfg = runs[idx - 1]
        run_id = f"{idx:03d}_{run_name}"

        if skip_existing:
            p = marker_path(state_dir, run_id, "success")
            if os.path.exists(p):
                if verbose:
                    print(f"[sweep] SKIP existing success: {run_id}")
                skip += 1
                continue

        if verbose:
            print(f"[sweep] === Run {idx}/{n} ===")
            print(f"[sweep] id={run_id}")

        cfg = dict(run_cfg)  # shallow copy

        if force_auto_test or str(cfg.get("test_split_file", "")).upper() == "AUTO":
            test_json, train_full_json = ensure_auto_splits_for_run(
                cfg, split_root=split_root, test_per_class=cfg.get("test_per_class")
            )
            cfg["test_split_file"] = test_json
            cfg.setdefault("train_full_split_file", train_full_json)

        cmd = build_experiment_cmd(cfg)

        if verbose:
            print("[run] ", " ".join(cmd))

        t0 = time.time()
        rc = run_subprocess(cmd)
        dt = time.time() - t0

        payload = {"run_id": run_id, "rc": rc, "secs": round(dt, 2)}
        if rc == 0:
            success += 1
            write_marker(marker_path(state_dir, run_id, "success"), payload)
        else:
            fail += 1
            write_marker(marker_path(state_dir, run_id, "fail"), payload)

    return (success, fail, skip)
