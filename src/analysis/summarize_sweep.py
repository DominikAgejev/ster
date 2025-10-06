# src/analysis/summarize_sweep.py
"""
Summarize a sweep: best runs, best combinations, and elimination candidates.

It walks one or more checkpoint directories, loads .pt/.pth files, and extracts:
- best validation metric (e.g., val_de00),
- epoch,
- config (for model/backbone/features/... grouping).

Outputs:
- CSV with per-run bests,
- CSV with group aggregates,
- Markdown report with ranked tables + elimination suggestions.

Usage (examples):
  python -m src.engine.summarize_sweep \
    --ckpt_dir ./checkpoints/compare \
    --monitor de00 --mode min \
    --out_csv ./results/sweep/compare/summary_runs.csv \
    --out_groups_csv ./results/sweep/compare/summary_groups.csv \
    --md_report ./results/sweep/compare/summary_report.md

  # Multiple roots:
  python -m src.engine.summarize_sweep \
    --ckpt_dir ./checkpoints/compare ./more_ckpts \
    --monitor de00 --mode min --skip_bad
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# -----------------------
# Helpers
# -----------------------
def sha12(d: Dict[str, Any]) -> str:
    """Stable short hash for a dict (used to group checkpoints of the same run)."""
    def _norm(x):
        if isinstance(x, dict):
            return {k: _norm(x[k]) for k in sorted(x.keys())}
        if isinstance(x, list):
            return [_norm(v) for v in x]
        return x
    payload = json.dumps(_norm({k: v for k, v in d.items() if not str(k).startswith("_")}),
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

def _is_better(a: float, b: Optional[float], mode: str) -> bool:
    try:
        a_f = float(a)
    except Exception:
        return False
    if b is None:
        return True
    try:
        b_f = float(b)
    except Exception:
        return True
    if not math.isfinite(a_f):  # NaN/inf never beats a finite value
        return False
    if not math.isfinite(b_f):
        return True
    return (a_f < b_f) if mode == "min" else (a_f > b_f)

def find_files(roots: Iterable[str], exts=(".pt", ".pth")) -> List[str]:
    out: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        # Skip 'final_test' unless the root itself *is* final_test
        allow_final = os.path.basename(os.path.normpath(root)) == "final_test"
        for dirpath, dirnames, filenames in os.walk(root):
            if not allow_final:
                dirnames[:] = [d for d in dirnames if d != "final_test"]            
            for fn in filenames:
                if fn.endswith(exts):
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)

def find_json_summaries(roots: Iterable[str]) -> List[str]:
    """
    Find JSON sidecar summaries, e.g. fold summaries or final_test summaries.
    Matches '*_summary.json' and 'final_summary.json'.
    """
    out: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
           continue
        allow_final = os.path.basename(os.path.normpath(root)) == "final_test"
        for dirpath, dirnames, filenames in os.walk(root):
            if not allow_final:
                dirnames[:] = [d for d in dirnames if d != "final_test"]
            for fn in filenames:
                if fn.endswith("_summary.json") or fn == "final_summary.json":
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)

def try_parse_metric_from_filename(path: str, monitor: str) -> Optional[float]:
    """
    Fallback: extract metric from filenames like "...de00=4.31..." or "...val_de00=4.31...".
    """
    base = os.path.basename(path)
    patterns = [
        rf"val_{re.escape(monitor)}=([0-9]+(?:\.[0-9]+)?)",
        rf"{re.escape(monitor)}=([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, base)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

def _to_hashable(v):
    # Lists/sets -> stable, readable string; dict -> JSON
    if isinstance(v, list):
        try:
            return "none" if len(v) == 0 else "+".join(map(str, sorted(v)))
        except Exception:
            return json.dumps(v, sort_keys=True, separators=(",", ":"))
    if isinstance(v, set):
        return "none" if len(v) == 0 else "+".join(map(str, sorted(v)))
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return v

def flatten_config(cfg: Dict[str, Any], include: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Flatten *all* config keys (or a provided subset) into hashable scalars.
    """
    if not isinstance(cfg, dict):
        return {}
    keys = list(include) if include is not None else [k for k in cfg.keys() if not str(k).startswith("_")]
    out = {}
    for k in keys:
        try:
            out[k] = _to_hashable(cfg[k])
        except Exception:
            out[k] = str(cfg.get(k))
    return out

@dataclass
class Record:
    run_id: str
    ckpt_path: str
    metric: float
    epoch: Optional[int]
    config: Dict[str, Any]
    metric_kind: str  # 'val', 'test', 'cvmean', 'metric'


# -----------------------
# Core load
# -----------------------
def load_ckpt_info(path: str, monitor: str, mode: str, skip_bad: bool) -> Optional[Record]:
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        if skip_bad:
            print(f"[warn] failed to load {path}: {e}")
            return None
        raise

    cfg = {}
    metric = None
    epoch = None

    if isinstance(obj, dict):
        cfg = obj.get("config", {}) or {}
        epoch = obj.get("epoch", None)
        metric = obj.get(f"val_{monitor}", None)

    # Fallback to filename parsing
    if metric is None:
        metric = try_parse_metric_from_filename(path, monitor)

    # ---- NEW: normalize non-finite metrics to None so we can skip them ----
    try:
        if metric is not None and (not math.isfinite(float(metric))):
            metric = None
    except Exception:
        metric = None

    if metric is None:
        if skip_bad:
            return None
        raise ValueError(f"Checkpoint {path} missing usable metric for '{monitor}' and filename fallback failed.")

    if not cfg:
        cfg = {}

    kind = "cvmean" if os.path.basename(path) == "cvmean.pt" else "val"
    run_id = sha12(cfg) if cfg else f"nofcfg::{os.path.basename(path)}"
    return Record(run_id=run_id, ckpt_path=path, metric=float(metric), epoch=epoch, config=cfg, metric_kind=kind)

def load_json_info(path: str, monitor: str, prefer_test: bool, skip_bad: bool) -> Optional[Record]:
    """
    Load a JSON summary and emit a Record.
    Priority (when prefer_test=True): test_<monitor> -> val_<monitor> -> best_metric/best_value/metric/value.
    Robust to missing/non-finite metrics and tags metric_kind as 'test'/'val' using JSON or path heuristics.
    """
    try:
        with open(path, "r") as f:
            rec = json.load(f) or {}
    except Exception as e:
        if skip_bad:
            print(f"[warn] failed to read {path}: {e}")
            return None
        raise

    cfg   = rec.get("config", {}) or {}
    epoch = rec.get("best_epoch", rec.get("epoch", None))

    # Determine 'kind' from JSON if available; otherwise use path heuristics.
    lp = path.replace("\\", "/")
    kind = rec.get("metric_kind", rec.get("kind"))    
    if not kind:
        if "/final_test/" in lp:
            kind = "test"
        elif "cvmean" in os.path.basename(lp):
            kind = "cvmean"
        else:
            kind = None  # leave undecided for now

    # Candidate metrics
    def _first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    test_metric = _first(
        rec.get(f"test_{monitor}"),
        rec.get("test_best_metric"),
        rec.get("best_test_metric"),
    )
    val_metric = _first(
        rec.get(f"val_{monitor}"),
        rec.get("val_best_metric"),
        rec.get("best_val_metric"),
    )
    generic_metric = _first(
        rec.get("best_metric"),
        rec.get("best_value"),
        rec.get("metric"),
        rec.get("value"),
    )

    metric = None
    metric_kind = None

    if prefer_test and test_metric is not None:
        metric = test_metric
        metric_kind = "test"

    if metric is None and val_metric is not None:
        metric = val_metric
        metric_kind = "val"

    if metric is None and generic_metric is not None:
        metric = generic_metric
        # If generic but we know kind from JSON/path, use it; else default sensibly.
        if kind in {"test", "val", "cvmean"}:
            metric_kind = kind
        else:
            metric_kind = "test" if "/final_test/" in lp else "val"

    if metric is None:
        if skip_bad:
            return None
        raise ValueError(f"JSON summary {path} lacks usable metric (test_{monitor}/val_{monitor}/best_metric/best_value/metric/value).")

    # Ensure finite float
    try:
        metric = float(metric)
        if not math.isfinite(metric):
            if skip_bad:
                return None
            raise ValueError(f"Non-finite metric in {path}: {metric}")
    except Exception as e:
        if skip_bad:
            return None
        raise

    # If still unset, fall back to detected 'kind'
    if not metric_kind:
        metric_kind = kind or ("test" if "/final_test/" in lp else "val")

    run_id = sha12(cfg) if cfg else f"json::{os.path.basename(path)}"
    return Record(run_id=run_id, ckpt_path=path, metric=metric, epoch=epoch, config=cfg, metric_kind=metric_kind)


def aggregate_best(records: List[Record], mode: str) -> List[Record]:
    """
    For each (run_id, metric_kind), keep the single best record according to mode ('min' or 'max').
    """
    best: Dict[Tuple[str, str], Record] = {}

    for r in records:
        key = (r.run_id, r.metric_kind)
        if key not in best or _is_better(r.metric, best[ key ].metric, mode):
            best[key] = r

    return list(best.values())

def consolidate_best(records: List[Record], mode: str, prefer_test: bool) -> List[Record]:
    """
    Reduce to ONE record per run_id with a clear preference order.
    Steps:
      1) Keep the best record within each (run_id, metric_kind) according to mode.
      2) For each run_id, pick by preference order:
         - if prefer_test: ['test', 'cvmean', 'val', 'metric']
         - else          : ['cvmean', 'val', 'test', 'metric']
    """
    best_per_kind = {}
    better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    for r in records:
        key = (r.run_id, r.metric_kind)
        if key not in best_per_kind or better(r.metric, best_per_kind[key].metric):
            best_per_kind[key] = r
    by_run: Dict[str, Dict[str, Record]] = {}
    for (rid, kind), rec in best_per_kind.items():
        by_run.setdefault(rid, {})[kind] = rec
    order = ["test", "cvmean", "val", "metric"] if prefer_test else ["cvmean", "val", "test", "metric"]
    out: List[Record] = []
    for rid, kinds in by_run.items():
        for k in order:
            if k in kinds:
                out.append(kinds[k])
                break
    return out

# -----------------------
# Analysis
# -----------------------
def to_dataframe(recs: List[Record], monitor: str) -> pd.DataFrame:
    rows = []
    for r in recs:
        flat = flatten_config(r.config)
        rows.append({
            "run_id": r.run_id,
            "metric_kind": r.metric_kind,
            "ckpt_path": r.ckpt_path,
            "best_metric": r.metric,
            "best_epoch": r.epoch,
            **flat,
        })
    df = pd.DataFrame(rows)
    # Reorder columns
    first = ["run_id", "metric_kind", "best_metric", "best_epoch", "ckpt_path"]    
    others = [c for c in df.columns if c not in first]
    # Drop obvious runtime-only decorations that shouldn't be treated as tunables
    drop_cols = ["device", "train_size", "val_size"]
    cols = first + [c for c in others if c not in drop_cols]
    return df[cols]


def group_and_rank(df: pd.DataFrame, mode: str, group_keys: List[str]) -> pd.DataFrame:
    keys = [k for k in group_keys if k in df.columns]
    if not keys:
        return pd.DataFrame()
    safe = df.copy()
    for k in keys:
        safe[k] = safe[k].map(_to_hashable)
    agg = safe.groupby(keys, dropna=False)["best_metric"].agg(["count", "median", "mean", "std"]).reset_index()
    agg = agg.sort_values("median", ascending=(mode == "min")).reset_index(drop=True)
    return agg


def suggest_eliminations(groups: pd.DataFrame, mode: str, threshold: float, min_count: int) -> pd.DataFrame:
    """
    Mark groups whose median is worse than the best median by > threshold.
    """
    if groups.empty:
        return groups

    best_med = groups["median"].min() if mode == "min" else groups["median"].max()
    if mode == "min":
        diff = groups["median"] - best_med
    else:
        diff = best_med - groups["median"]

    elim = groups.copy()
    elim["delta_vs_best"] = diff
    elim = elim.loc[(elim["count"] >= min_count) & (elim["delta_vs_best"] > threshold)]
    elim = elim.sort_values(["delta_vs_best", "count"], ascending=[False, False]).reset_index(drop=True)
    return elim


def render_markdown_report(
    runs_df: pd.DataFrame,
    groups: Dict[str, pd.DataFrame],
    eliminations: Dict[str, pd.DataFrame],
    monitor: str,
    mode: str,
    top_k: int = 25,
    constants: Optional[Dict[str, Any]] = None,
    grid_keys: Optional[List[str]] = None,
    top_cols: Optional[List[str]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"# Sweep Summary (monitor: `{monitor}`, mode: `{mode}`)")
    lines.append("")
    lines.append(f"Total unique runs: **{len(runs_df)}**")
    if "best_metric" in runs_df:
        best_val = runs_df["best_metric"].min() if mode == "min" else runs_df["best_metric"].max()
        # Clarify that 'best_metric' might be 'test_<monitor>' when --prefer_test is used
        src_hint = ""
        if "metric_kind" in runs_df.columns:
            mk = runs_df.sort_values("best_metric", ascending=(mode=="min")).iloc[0]["metric_kind"]
            src_hint = f" ({'test' if mk=='test' else mk})"
        lines.append(f"Best run metric{src_hint}: **{best_val:.4f}**")
    lines.append("")


    # Constants block
    if constants:
        lines.append("## Constants")
        if constants:
            const_df = pd.DataFrame([constants])
            # Keep a deterministic, readable order
            cols = list(constants.keys())
            lines.append(const_df[cols].to_markdown(index=False))
        else:
            lines.append("_none detected_")
        lines.append("")

    # Grid keys block
    if grid_keys:
        lines.append("## Grid keys (vary across runs)")
        lines.append(", ".join(f"`{k}`" for k in grid_keys))
        lines.append("")
 

    # Top runs
    lines.append("## Top Runs")
    top = runs_df.sort_values("best_metric", ascending=(mode == "min")).head(top_k)
    if top.empty:
        lines.append("_No runs found._")
    else:
        cols = [c for c in (top_cols or [c for c in top.columns if c not in ("ckpt_path",)]) if c in top.columns]
        lines.append(top[cols].to_markdown(index=False))
    lines.append("")

    # Group sections
    for name, df in groups.items():
        lines.append(f"## Grouped by {name}")
        if df.empty:
            lines.append("_not enough info to group_")
            lines.append("")
            continue
        lines.append(df.head(top_k).to_markdown(index=False))
        lines.append("")

        elim = eliminations.get(name)
        if elim is not None and not elim.empty:
            lines.append(f"### Elimination candidates ({name})")
            lines.append(elim.to_markdown(index=False))
            lines.append("")
    return "\n".join(lines)


# -----------------------
# CLI
# -----------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Summarize checkpoints from a sweep.")
    p.add_argument("--ckpt_dir", nargs="+", required=True,
                   help="One or more root directories to scan for .pt/.pth files.")
    p.add_argument("--monitor", default="de00",
                   help="Validation metric key saved as 'val_<monitor>' in checkpoints. Default: de00")
    p.add_argument("--mode", choices=["min", "max"], default="min",
                   help="Whether lower (min) or higher (max) is better for the monitor.")
    p.add_argument("--include_json", action="store_true",
                   help="Also scan for JSON summaries (*_summary.json, final_summary.json).")
    p.add_argument("--prefer_test", action="store_true",
                   help="When loading JSON summaries, prefer test_<monitor> over val_<monitor> if available.")
    p.add_argument("--only_test", action="store_true",
               help="Keep only test-evaluation records (metric_kind == 'test').")
    p.add_argument("--out_csv", default="sweep_summary_runs.csv",
                   help="Path to write per-run CSV.")
    p.add_argument("--out_groups_csv", default="sweep_summary_groups.csv",
                   help="Path to write grouped CSV (multi-index flattened).")
    p.add_argument("--md_report", default="sweep_report.md",
                   help="Path to write markdown report.")
    p.add_argument("--elim_threshold", type=float, default=0.5,
                   help="Delta vs best median to mark elimination candidates (mode=min).")
    p.add_argument("--elim_min_count", type=int, default=3,
                   help="Minimum count within a group to consider elimination.")
    p.add_argument("--skip_bad", action="store_true",
                   help="Skip unreadable checkpoints instead of failing.")
    p.add_argument("--prefer_cvmean", action="store_true",
                   help="If true and cvmean.pt files exist in any ckpt_dir, only load those.")
    p.add_argument("--group_keys", nargs="*", default=None,
                   help="Config keys to group by. If omitted and --auto_group is set, use detected grid keys.")
    p.add_argument("--auto_group", action="store_true",
                   help="Automatically group by detected grid keys (varying columns).")
    p.add_argument("--extra_group_keys", nargs="*", default=[],
                   help="Force-additional group keys (on top of detected grid keys).")
    p.add_argument("--grid_keys", nargs="*", default=None,
                   help="Explicit grid keys to prefer over auto-detection.")
    p.add_argument("--report_keep", nargs="*", default=[],
                   help="Base constants to display in the Constants block and keep in the top table.")
    p.add_argument("--top_k", type=int, default=25, help="Rows to show in markdown tables.")
    return p


def main():
    args = build_argparser().parse_args()
    explicit_test_root = any(os.path.basename(os.path.normpath(p)) == "final_test"
                             for p in args.ckpt_dir)
    paths = find_files(args.ckpt_dir, exts=(".pt", ".pth"))
    if args.prefer_cvmean:
        cvmeans = [p for p in paths if os.path.basename(p) == "cvmean.pt"]
        if cvmeans:
            paths = cvmeans
    json_paths: List[str] = []
    if args.include_json:
        json_paths = find_json_summaries(args.ckpt_dir)
    # It's valid to summarize final_test dirs that only have JSON summaries.
    if not paths and not json_paths:
        print("[error] No checkpoint or JSON summary files found in:", args.ckpt_dir, file=sys.stderr)
        sys.exit(2)
    print(f"[info] Found {len(paths)} checkpoint files and {len(json_paths)} json summaries.")
    # Load all ckpts -> best per run
    recs: List[Record] = []
    for pth in paths:
        r = load_ckpt_info(pth, monitor=args.monitor, mode=args.mode, skip_bad=args.skip_bad)
        if r is not None:
            recs.append(r)
    # Load JSON sidecars as records
    for pth in json_paths:
        r = load_json_info(pth, monitor=args.monitor, prefer_test=args.prefer_test, skip_bad=args.skip_bad)
        if r is not None:
            recs.append(r)

    if not recs:
        print("[error] No usable checkpoints (missing config/metrics).", file=sys.stderr)
        sys.exit(3)

    # DIAG: counts by kind *before* consolidation
    from collections import Counter
    kinds = Counter([getattr(r, "metric_kind", "?") for r in recs])
    print(f"[diag] loaded records by kind: {dict(kinds)}")
    best = consolidate_best(recs, mode=args.mode, prefer_test=args.prefer_test)    
    runs_df = to_dataframe(best, monitor=args.monitor)

    # --- NaN coalescing: if a column has exactly one non-NaN value across rows,
    # backfill all NaNs with that value. This removes spurious variation caused by
    # JSON/ckpt config asymmetry.
    for col in [c for c in runs_df.columns if c not in ("run_id","best_metric","best_epoch","ckpt_path","metric_kind")]:
        vals = runs_df[col].dropna().unique().tolist()
        if len(vals) == 1:
            runs_df[col] = runs_df[col].fillna(vals[0])

    # Optional: normalize string "nan"/"None" coming from odd configs
    runs_df = runs_df.replace({"nan": np.nan, "None": np.nan, "null": np.nan})

    # --- NEW: de-duplicate identical configs (ckpt vs json for the same run) ---
    # Sort so we keep the better metric (and earlier epoch as a tiebreaker).
    conf_cols = [c for c in runs_df.columns if c not in ("run_id","best_metric","best_epoch","ckpt_path","metric_kind")]
    if conf_cols:
        asc = (args.mode == "min")
        runs_df = runs_df.sort_values(
            by=["best_metric","best_epoch"],
            ascending=[asc, True],
            kind="mergesort"  # stable
        ).drop_duplicates(subset=conf_cols, keep="first").reset_index(drop=True)
    
    if "metric_kind" in runs_df.columns and (runs_df["metric_kind"] == "test").any():
        sample = runs_df.loc[runs_df["metric_kind"] == "test"].head(3)[["ckpt_path","best_metric"]]
        print("[diag] sample TEST rows:")
        for p, m in sample.values.tolist():
            print(f"        {m:.4f} <- {p}")
    # --- Final-test cleanup ---
    # If we preferred test and any test rows exist, keep only those.
    if explicit_test_root and args.prefer_test and "metric_kind" in runs_df.columns:
        if (runs_df["metric_kind"] == "test").any():
            runs_df = runs_df[runs_df["metric_kind"] == "test"].reset_index(drop=True)

    # Keep core columns even if NaN; drop only truly useless extras
    _protect = {"run_id", "best_metric", "best_epoch", "ckpt_path", "metric_kind"}
    cols_to_keep = [c for c in runs_df.columns if (c in _protect) or not runs_df[c].isna().all()]
    runs_df = runs_df.loc[:, cols_to_keep]


    # Save per-run CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    runs_df.to_csv(args.out_csv, index=False)
    print(f"[ok] Wrote per-run summary: {args.out_csv}")

    # --- Detect grid keys vs constants ---
    conf_cols = [c for c in runs_df.columns
              if c not in ("run_id","best_metric","best_epoch","ckpt_path","metric_kind")]

    if args.grid_keys:
        grid_keys = [k for k in args.grid_keys if k in conf_cols]
    else:
        grid_keys = [c for c in conf_cols if runs_df[c].nunique(dropna=False) > 1]
    grid_keys = list(dict.fromkeys(grid_keys))  # stable
  
    def _first_non_nan(col):
        s = runs_df[col].dropna()
        return s.iloc[0] if len(s) else None

    # Constants: exactly what the caller asked to keep_from_base
    constants_map = {k: _first_non_nan(k) for k in (args.report_keep or []) if k in runs_df.columns}

    # Columns for the "Top Runs" table: report_keep + grid_keys + metric/epoch
    top_cols = ["run_id", "metric_kind"] \
            + [c for c in (args.report_keep or []) if c in runs_df.columns] \
            + grid_keys + ["best_metric","best_epoch"]
    # de-duplicate, drop all-NaN columns
    top_cols = [c for c in dict.fromkeys(top_cols)
                if c in runs_df.columns and not runs_df[c].isna().all()]
    # if metric_kind is constant (e.g., all "test"), hide it to reduce noise
    if "metric_kind" in runs_df.columns and runs_df["metric_kind"].nunique(dropna=False) <= 1:
        top_cols = [c for c in top_cols if c != "metric_kind"]

    # Also emit: best by grid combo (dedup)
    if grid_keys:
        best_by_grid = runs_df.sort_values("best_metric", ascending=(args.mode=="min")).drop_duplicates(subset=grid_keys, keep="first")
        best_by_grid_path = os.path.splitext(args.out_csv)[0] + "_best_by_grid.csv"
        best_by_grid.to_csv(best_by_grid_path, index=False)
        print(f"[ok] Wrote best-by-grid: {best_by_grid_path}")

    # Columns meta sidecar
    meta_json = os.path.splitext(args.out_csv)[0] + "_columns_meta.json"
    with open(meta_json, "w") as f:
        json.dump({
            "grid_keys": grid_keys,
            "report_keep": args.report_keep,
            "constants": list(constants_map.keys()),
            "top_cols": top_cols
        }, f, indent=2)
    print(f"[ok] Wrote columns meta: {meta_json}")

    # Group tables
    groups: Dict[str, pd.DataFrame] = {}
    eliminations: Dict[str, pd.DataFrame] = {}

    # Determine group keys to use
    if args.auto_group and (args.group_keys is None):
        # Only the declared grid (from YAML) + extra_group_keys
        use_group_keys = [k for k in (list(dict.fromkeys(grid_keys + args.extra_group_keys))) if k in runs_df.columns]
    else:
        use_group_keys = args.group_keys or []

    # Single-key groupings
    for key in use_group_keys:
        gdf = group_and_rank(runs_df, mode=args.mode, group_keys=[key])
        groups[key] = gdf
        if not gdf.empty:
            elim = suggest_eliminations(gdf, mode=args.mode,
                                        threshold=args.elim_threshold,
                                        min_count=args.elim_min_count)
            eliminations[key] = elim

    # Two-key combinations (useful sweet spot)
    pair_keys: List[Tuple[str, str]] = []
    gks = [k for k in use_group_keys if k in runs_df.columns]    
    for i in range(len(gks)):
        for j in range(i + 1, len(gks)):
            pair_keys.append((gks[i], gks[j]))

    # Flatten multi-group results for CSV
    group_rows: List[pd.DataFrame] = []
    for name, df in groups.items():
        if df.empty:
            continue
        tag = pd.DataFrame({"_group_by": [name] * len(df)})
        group_rows.append(pd.concat([tag, df], axis=1))

    for a, b in pair_keys:
        df = group_and_rank(runs_df, mode=args.mode, group_keys=[a, b])
        key_name = f"{a}+{b}"
        groups[key_name] = df
        if not df.empty:
            group_rows.append(pd.concat([pd.DataFrame({"_group_by": [key_name] * len(df)}), df], axis=1))

    if group_rows:
        grp_all = pd.concat(group_rows, ignore_index=True)
        grp_all.to_csv(args.out_groups_csv, index=False)
        print(f"[ok] Wrote grouped summary: {args.out_groups_csv}")
    else:
        grp_all = pd.DataFrame()
        print("[warn] No groups to write (missing config keys in checkpoints).")

    # Markdown report
    report = render_markdown_report(
        runs_df, groups, eliminations,
        monitor=args.monitor, mode=args.mode, top_k=args.top_k,
        constants=constants_map, grid_keys=grid_keys, top_cols=top_cols
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.md_report)) or ".", exist_ok=True)
    with open(args.md_report, "w") as f:
        f.write(report)
    print(f"[ok] Wrote markdown report: {args.md_report}")

    # Console hint for eliminations
    if eliminations:
        print("\n[ELIMINATION CANDIDATES]")
        for name, df in eliminations.items():
            if df is not None and not df.empty:
                print(f"- by {name}: top {min(5, len(df))} worst vs best median")
                print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
