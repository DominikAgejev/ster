# src/engine/sweep/plan.py
from __future__ import annotations
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple
import os, yaml


def load_runs(config_path: str) -> tuple[list[tuple[str, dict]], list[str], bool]:
    """
    Parse a sweep YAML. Supports either:
      - base + grid: cartesian expansion
      - base + runs: explicit list

    Returns:
        (runs: [(name, merged_cfg)], grid_keys, used_grid_flag)
    """
    with open(config_path, "r") as f:
        spec = yaml.safe_load(f) or {}

    base: Dict[str, Any] = spec.get("base", {}) or {}
    grid: Dict[str, Any] = spec.get("grid", {}) or {}
    runs_spec: List[Dict[str, Any]] = spec.get("runs", []) or []

    if runs_spec:
        runs: list[tuple[str, dict]] = []
        for i, r in enumerate(runs_spec, 1):
            r = r or {}
            name = r.get("name", f"run{i}")
            merged = {**base, **r}
            runs.append((name, merged))
        return runs, [], False

    if grid:
        items = list(grid.items())
        keys = [k for k, _ in items]
        values = [v for _, v in items]
        runs: list[tuple[str, dict]] = []
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            merged = {**base, **overrides, "tag_keys": keys}
            name = "__".join(f"{k}={merged.get(k)}" for k in keys)
            runs.append((name, merged))
        return runs, keys, True

    raise ValueError("Sweep config must contain either 'runs:' or 'grid:' at top-level.")
