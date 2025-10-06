#!/usr/bin/env python3
# src/engine/sweep/cli.py
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple, Set, Union
import yaml

from .plan import load_runs
from .exec import execute_runs
from .utils import (
    state_dir_for_config,
    _coerce_list, _if_token, _resolve_auto_test_split, _default_results_root_for_config,
    run_subprocess,
)
from .winners import stage_outdirs, select_winners_from_csv, eval_winners_on_test
from ...analysis.delta_distribution import analyze_from_csv
import torch

from .winners import DEFAULT_ALLOWED_KEYS


# -----------------------
# Summarizer wrapper
# -----------------------
def run_auto_summary(
    cfg_or_path: Union[str, Dict[str, Any]],
    ckpt_dirs,
    monitor,
    *,
    label: str = None,
    summary_module: str = "src.analysis.summarize_sweep",
    prefer_test: bool = True,
    only_test: bool = False,
    summary_outdir: str = None,
):
    """
    Produce CSV + grouped CSV + MD for the given ckpt roots.
    - ckpt_dirs: list of folders (e.g., ["./checkpoints/.../full"] or ["./checkpoints/.../full","./checkpoints/.../full/final_test"])
    - label: suffix in filenames: <cfg_name>_<label>_summary_*.{csv,md}
    - prefer_test: pass --prefer_test (keep TEST when a 'final_test' root is present)
    - only_test: additionally pass --only_test (forces test-only rows)
    """

    # --- load cfg if a path was passed ---
    if isinstance(cfg_or_path, str):
        with open(cfg_or_path, "r") as _f:
            cfg = yaml.safe_load(_f) or {}
        cfg["__config_path__"] = cfg_or_path
    else:
        cfg = cfg_or_path or {}

    # --- reporting hints from YAML ---
    rep      = cfg.get("report", {}) if isinstance(cfg, dict) else {}
    keep     = rep.get("keep_from_base", []) or []
    extra_gk = rep.get("extra_group_keys", []) or []
    grid     = sorted((cfg.get("grid") or {}).keys())

    # --- output paths ---
    cfg_path  = cfg.get("__config_path__", "")
    cfg_name  = os.path.splitext(os.path.basename(cfg_path or "config"))[0]
    label     = label or "summary"
    # results root alongside configs unless overridden
    report_root  = os.path.abspath(summary_outdir) if summary_outdir else _default_results_root_for_config(cfg_path)
    os.makedirs(report_root, exist_ok=True)

    base = os.path.join(report_root, f"{cfg_name}_{label}_summary")
    out_csv = f"{base}_runs.csv"
    out_grp = f"{base}_groups.csv"
    out_md  = f"{base}_report.md"

    # --- build command ---
    cmd = [
        sys.executable, "-m", summary_module,
        "--ckpt_dir", *list(map(str, ckpt_dirs)),
        "--monitor", monitor,
        "--mode", "min",
        "--skip_bad",
        "--auto_group",
        "--report_keep", *map(str, keep),
        "--extra_group_keys", *map(str, extra_gk),
        "--include_json",
        "--out_csv", out_csv,
        "--out_groups_csv", out_grp,
        "--md_report", out_md,
    ]
    if grid:
        cmd.extend(["--grid_keys", *map(str, grid)])
    if prefer_test:
        cmd.append("--prefer_test")
    if only_test:
        cmd.append("--only_test")

    print("[summary] RUN:", " ".join(cmd))
    run_subprocess(cmd)
    return out_csv, out_grp, out_md


# -----------------------
# NEW: Analyzer wrapper
# -----------------------
def run_validation_analysis(
    spec_or_path: Union[str, Dict[str, Any]],
    runs_csv: str,
    *,
    split_root: str,
    test_split_file_arg: str | None,
    outdir_root: str | None = None,
    want_mode: str = "both",
) -> str:
    """
    Call ΔE00 distribution analyzer on the checkpoints listed in `runs_csv`.

    - Resolves TEST split:
        * if test_split_file_arg == "AUTO" -> compute deterministic path
        * if test_split_file_arg is a file -> use as-is
        * else -> fall back to VAL-only analysis
    - Writes outputs under:
        outdir_root (if provided) OR <results>/<cfg_name>/analysis
    - Returns the outdir where analysis was written.
    """
    # Load spec + base
    if isinstance(spec_or_path, str):
        with open(spec_or_path, "r") as _f:
            spec = yaml.safe_load(_f) or {}
        spec["__config_path__"] = spec_or_path
    else:
        spec = spec_or_path or {}

    base_cfg = (spec.get("base") or {})
    cfg_path = spec.get("__config_path__", "")

    # Where to write
    outdir = outdir_root or _default_results_root_for_config(cfg_path, label_dir="analysis")
    os.makedirs(outdir, exist_ok=True)

    # Data roots
    images_dir = base_cfg.get("images_dir")
    json_dir   = base_cfg.get("json_dir")
    labels_csv = base_cfg.get("labels_csv")
    if not all([images_dir, json_dir, labels_csv]):
        raise RuntimeError("analysis: images_dir/json_dir/labels_csv must be set in base config.")

    # Resolve TEST split (AUTO / explicit / missing)
    mode = want_mode
    test_split_file: str | None = None
    if want_mode in ("test", "both"):
        if test_split_file_arg and str(test_split_file_arg).upper() == "AUTO":
            # derive deterministic test split path
            test_split_file = _resolve_auto_test_split(base_cfg, split_root=split_root)
            if not os.path.isfile(test_split_file):
                print(f"[analysis][warn] AUTO test split not found at {test_split_file}; falling back to VAL-only.")
                mode = "val"
                test_split_file = None
        elif test_split_file_arg and os.path.isfile(str(test_split_file_arg)):
            test_split_file = str(test_split_file_arg)
        else:
            print("[analysis][warn] --test_split_file not set or does not exist; falling back to VAL-only.")
            mode = "val"
            test_split_file = None

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run analyzer
    # Prefer ckpt rows; drop JSON-only rows if present
    runs_csv_for_analysis = runs_csv
    try:
        import pandas as pd  # local import to avoid hard dep elsewhere
        df = pd.read_csv(runs_csv)
        if "ckpt_path" in df.columns:
            m = df["ckpt_path"].astype(str).str.endswith((".pt", ".pth"))
            df2 = df[m]
            if not df2.empty:
                runs_csv_for_analysis = os.path.join(outdir, "analysis_input.csv")
                df2.to_csv(runs_csv_for_analysis, index=False)
    except Exception as e:
        print(f"[analysis][warn] prefilter failed: {e}; continuing with original CSV.")

    print(f"[analysis] runs_csv={runs_csv_for_analysis}")
    print(f"[analysis] mode={mode} test_split_file={test_split_file or '—'} outdir={outdir}")
    analyze_from_csv(
        runs_csv=runs_csv_for_analysis,
        images_dir=images_dir,
        json_dir=json_dir,
        labels_csv=labels_csv,
        mode=mode,
        outdir_root=outdir,
        test_split_file=test_split_file,
        val_split_file=None,
        device=device,
        bins=60,
    )
    return outdir


# -----------------------
# CLI
# -----------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a sweep from a YAML config (grid or explicit runs), with optional staged winners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to sweep YAML (with base + grid or base + runs).")
    p.add_argument("--start", type=int, default=1, help="1-based start index")
    p.add_argument("--stop", type=int, default=None, help="1-based inclusive stop index")
    p.add_argument("--split-root", default="./splits/sweep", help="Root where AUTO splits are persisted")
    p.add_argument("--force-auto-test", action="store_true",
                   help="Force AUTO test/train_full materialization for every run (overrides run cfg).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs with existing SUCCESS markers for the selected index range.")
    p.add_argument("--verbose", type=int, default=1, help="Verbosity level (0/1)")

    # Summary
    p.add_argument("--auto-summary", action="store_true",
                   help="After each stage, call summarize on the stage's outdirs.")
    p.add_argument("--summary-outdir", default=None,
                   help="Directory for summary CSVs/Markdown (default: <config_dir>/../results/<cfg>/).")
    p.add_argument("--summary-module", default="src.analysis.summarize_sweep",
                   help="Module to run for summarizing (default: src.analysis.summarize_sweep).")
    p.add_argument("--monitor", default="de00",
                   help="Primary metric name used by the summarizer and eval (e.g., de00).")

    # Two-stage + winners
    p.add_argument(
        "--only_full",
        action="store_true",
        help="Skip the fast stage: train FULL for every run, then summarize and (optionally) eval winners/test."
    )
    p.add_argument("--two_stage", action="store_true",
                   help="If YAML has stages[], run stage-1 then stage-2 on winners.")
    p.add_argument("--eval-winners", action="store_true",
                   help="After final stage, evaluate winners on TEST.")
    p.add_argument("--winners_stage", type=str, default=None,
            help="Which stage's winners to evaluate (default: last stage name).")
    p.add_argument("--eval_top_k", type=int, default=None,
                help="Number of winners to evaluate on TEST (overrides stage/report policies).")
    p.add_argument("--retrain_for_test", action="store_true",
                   help="Legacy: retrain on train_full and eval on TEST (instead of best-ckpt test-eval).")
    p.add_argument("--test_split_file", type=str, default=None,
                   help="Path to persistent test split JSON, or 'AUTO' to auto-create per winner.")
    p.add_argument("--train_full_split_file", type=str, default=None,
                   help="Optional train-full JSON (if omitted and test is AUTO, will be auto-created).")
    p.add_argument("--eval_subdir", type=str, default="final_test",
                   help="Subdirectory under the stage outdir for final test-eval runs.")
    p.add_argument("--winner_group_keys", nargs="*", default=None,
                   help="Group winners by these keys. Default: ALL YAML grid keys. "
                        "Use 'none' to group by nothing.")

    p.add_argument("--analysis", action="store_true",
                   help="After eval/summary, run ΔE00 distribution analysis (val+test)")
    return p


def _expand_runs(base: Dict[str, Any], grid: Dict[str, Any] | None, runs_spec: List[Dict[str, Any]] | None
                 ) -> List[Tuple[str, Dict[str, Any]]]:
    if runs_spec:
        out = []
        for i, r in enumerate(runs_spec, 1):
            r = r or {}
            name = r.get("name", f"run{i}")
            merged = {**base, **r}
            out.append((name, merged))
        return out
    if grid:
        from itertools import product
        items = list(grid.items())
        keys = [k for k, _ in items]
        vals = [v for _, v in items]
        out = []
        for combo in product(*vals):
            overrides = dict(zip(keys, combo))
            merged = {**base, **overrides}
            # ensure experiment.py can include grid knobs in run_tag
            merged["tag_keys"] = keys
            name = "__".join(f"{k}={merged.get(k)}" for k in keys)
            out.append((name, merged))
        return out
    return [("base", dict(base))]


# --- Small stage helpers ------------------------------------------------------

def run_stage_runs(
    spec: Dict[str, Any],
    base: Dict[str, Any],
    stage_name: str,
    overrides: Dict[str, Any] | None,
    grid: Dict[str, Any],
    runs_l: List[Dict[str, Any]],
    args,
) -> tuple[Dict[str, Any], List[tuple[str, Dict[str, Any]]]]:
    """Prepare outdirs, expand runs, execute them. Returns (base_stage_cfg, runs)."""
    over = overrides or {}
    base_stage = stage_outdirs({**base, **over}, stage_name)
    runs_stage = _expand_runs(base_stage, grid, runs_l)

    execute_runs(
        args.config, runs_stage,
        start_index=1, stop_index=len(runs_stage),
        split_root=args.split_root,
        force_auto_test=args.force_auto_test,
        skip_existing=args.skip_existing,
        verbose=args.verbose,
    )
    return base_stage, runs_stage


def summarize_stage(
    spec: Dict[str, Any],
    outdir: str,
    args,
    label: str,
    prefer_test: bool = True,
    only_test: bool = False,
) -> str | None:
    """Run summarize_sweep for a single stage dir and return the CSV path."""
    csv, _, _ = run_auto_summary(
        spec, [outdir], args.monitor, label=label,
        prefer_test=prefer_test, only_test=only_test,
        summary_outdir=args.summary_outdir,
        summary_module=args.summary_module,
    )
    return csv


def winners_from_csv(
    csv_path: str,
    spec: Dict[str, Any],
    stage_spec: Dict[str, Any] | None,
    allowed_keys: set[str],
) -> List[Dict[str, Any]]:
    """Pick winners using keep_top_k / keep_top_frac in a stage spec (if given)."""
    keep_top_k    = (stage_spec or {}).get("keep_top_k")
    keep_top_frac = (stage_spec or {}).get("keep_top_frac")
    return select_winners_from_csv(
        csv_path,
        keep_top_k=keep_top_k, keep_top_frac=keep_top_frac,
        mode="min",
        allowed_keys=allowed_keys,
    )


def eval_winners_flow(
    winners: List[Dict[str, Any]],
    base_cfg: Dict[str, Any],
    stage_outdir: str,
    stage_logdir: str,
    args,
    tag_keys: List[str],
) -> tuple[int, int]:
    # Pass winners as the first positional argument (or use keyword 'winners=')
    return eval_winners_on_test(
        winners,
        base_cfg=base_cfg,
        stage_outdir=stage_outdir,
        stage_logdir=stage_logdir,
        monitor=args.monitor,
        test_split_file=args.test_split_file or "AUTO",
        train_full_split_file=args.train_full_split_file,
        split_root=args.split_root,
        retrain_for_test=args.retrain_for_test,
        eval_subdir=args.eval_subdir,
        tag_keys=tag_keys,
    )


def re_summarize_after_test(
    spec: Dict[str, Any],
    stage_outdir: str,
    args,
) -> tuple[str | None, str | None]:
    """
    1) PRETEST/SINGLE summary (has real ckpt paths) — good for analysis
    2) TEST-only summary — good for reporting
    Returns (pretest_csv, test_csv)
    """
    pre_csv, _, _ = run_auto_summary(
        spec, [stage_outdir], args.monitor, label="single",
        prefer_test=True, only_test=False,
        summary_outdir=args.summary_outdir,
        summary_module=args.summary_module,
    )
    final_test_dir = os.path.join(stage_outdir, args.eval_subdir or "final_test")
    test_csv, _, _ = run_auto_summary(
        spec, [stage_outdir, final_test_dir], args.monitor, label="test",
        prefer_test=True, only_test=True,
        summary_outdir=args.summary_outdir,
        summary_module=args.summary_module,
    )
    return pre_csv, test_csv

def _dedup_top_per_group(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Assumes `rows` are already sorted best→worst by the monitor metric.
    Keeps at most one row per (group_keys...) combination — the first (best).
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = tuple((k, r.get(k)) for k in group_keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def _resolve_eval_top_k(args, spec: Dict[str, Any], stage_spec: Dict[str, Any] | None, *, default_all: bool = True) -> int | None:
    """
    Priority:
      1) --eval_top_k (CLI)
      2) stage_spec.keep_top_k  (e.g., from stage-2)
      3) report.top_k           (YAML)
      4) if default_all: evaluate all (large sentinel); else None
    """
    if getattr(args, "eval_top_k", None) is not None:
        return int(args.eval_top_k)
    if stage_spec and stage_spec.get("keep_top_k") is not None:
        return int(stage_spec["keep_top_k"])
    rep = (spec.get("report") or {})
    if rep.get("top_k") is not None:
        return int(rep["top_k"])
    return 1_000_000 if default_all else None

def run_only_full(spec: Dict[str, Any], args) -> int:
    """
    FULL-only sweep:
      - train ALL runs at the 'full' stage (or stage-2 overrides if provided)
      - summarize full
      - optionally evaluate winners on TEST
      - optionally run analysis
    """
    base   = spec.get("base", {}) or {}
    grid   = spec.get("grid", {}) or {}
    runs_l = spec.get("runs", []) or []
    stages = spec.get("stages", []) or []

    # Prefer the 2nd stage overrides if YAML defines stages; else use base.
    if len(stages) >= 2:
        s2 = stages[1]
        s2_name = s2.get("name", "full")
        s2_over = s2.get("overrides", {}) or {}
    else:
        s2_name = "full"
        s2_over = {}

    # Train full for all runs
    print("[sweep] FULL-only mode: training full stage for all runs.")
    base_full, runs_full = run_stage_runs(
        spec=spec, base=base, stage_name=s2_name, overrides=s2_over,
        grid=grid, runs_l=runs_l, args=args,
    )

    # Summarize full
    full_csv = None
    if args.auto_summary:
        full_csv = summarize_stage(spec, base_full["outdir"], args, label="full")

    # If no winner eval requested: optionally analyze now (VAL + best VAL ckpts)
    if args.analysis and not args.eval_winners and full_csv and os.path.isfile(full_csv):
        run_validation_analysis(
            spec_or_path=spec,
            runs_csv=full_csv,
            split_root=args.split_root,
            test_split_file_arg=args.test_split_file,
            outdir_root=None,
            want_mode="both",
        )

    # Winner TEST eval flow
    if args.eval_winners:
        # Build a PRETEST summary to select winners from the FULL stage
        chosen_csv = summarize_stage(spec, base_full["outdir"], args, label="full_pretest")
        if not chosen_csv or not os.path.isfile(chosen_csv):
            print("[eval][warn] No summary CSV to select winners from; skipping.")
            return 0

        grid_keys = set((spec.get("grid") or {}).keys())
        allowed = grid_keys | DEFAULT_ALLOWED_KEYS

        s2_for_policy = stages[1] if len(stages) >= 2 else {}
        eval_top_k = _resolve_eval_top_k(args, spec, s2_for_policy, default_all=True)

        # 1) get ALL ranked candidates (ckpt-level)
        candidates = select_winners_from_csv(
            chosen_csv,
            keep_top_k=None, keep_top_frac=None,
            mode="min",
            allowed_keys=allowed,
        )
        # 2) collapse to top-1 per group (best row per group)
        #    default: ALL 'grid:' keys; CLI override supported
        grid_key_list = list((spec.get("grid") or {}).keys())
        if args.winner_group_keys is None:
            group_keys = grid_key_list or ["model","backbone","features","color_space"]
        else:
            group_keys = [] if (len(args.winner_group_keys) == 1 and args.winner_group_keys[0].lower() == "none") \
                            else list(args.winner_group_keys)
        winners = _dedup_top_per_group(candidates, group_keys)
        # 3) then cut to eval_top_k groups
        if eval_top_k is not None:
            winners = winners[:int(eval_top_k)]

        print(f"[eval] Selecting up to top_k={eval_top_k} unique group winner(s) for TEST in FULL-only mode.")

        if not winners:
            print("[eval][warn] No winners to evaluate in FULL-only mode; skipping.")
            return 0

        print("[eval][debug] winners (grouped):", [{k: w.get(k) for k in group_keys} for w in winners])
        ok, fail = eval_winners_flow(
            winners=winners,
            base_cfg=base,
            stage_outdir=base_full["outdir"],
            stage_logdir=base_full["logdir"],
            args=args,
            # IMPORTANT: preserve YAML insertion order to match training run tags
            tag_keys=list((spec.get("grid") or {}).keys()),
        )
        print(f"[eval] winners(full-only): ok={ok} fail={fail}")

        # Re-summarize to include TEST results; optionally run analysis
        pre_csv, test_csv = (None, None)
        if args.auto_summary:
            pre_csv, test_csv = re_summarize_after_test(spec, base_full["outdir"], args)

        runs_csv_for_analysis = pre_csv or test_csv
        if args.analysis and runs_csv_for_analysis and os.path.isfile(runs_csv_for_analysis):
            outdir_root = os.path.join(args.summary_outdir, "analysis") if args.summary_outdir else None
            run_validation_analysis(
                spec_or_path=spec,
                runs_csv=runs_csv_for_analysis,
                split_root=args.split_root,
                test_split_file_arg=args.test_split_file or "AUTO",
                outdir_root=outdir_root,
                want_mode="both",
            )

    print("[sweep] FULL-only mode complete.")
    return 0


def main(argv: List[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    with open(args.config, "r") as f:
        spec = yaml.safe_load(f) or {}
    spec["__config_path__"] = args.config

    # --- NEW: short-circuit to full-only mode if requested ---
    if getattr(args, "only_full", False):
        return run_only_full(spec, args)
    
    # -------- Two-stage --------
    base   = spec.get("base", {}) or {}
    grid   = spec.get("grid", {}) or {}
    runs_l = spec.get("runs", []) or {}
    stages = spec.get("stages", [])
    if not stages:
        print("[sweep] --two_stage set but 'stages' missing in YAML.")
        return 2

    # Stage 1 (fast)
    s1 = stages[0]
    s1_name = s1.get("name", "stage1")
    s1_over = s1.get("overrides", {}) or {}
    base_s1, runs_s1 = run_stage_runs(spec, base, s1_name, s1_over, grid, runs_l, args)

    winners_csv = summarize_stage(spec, base_s1["outdir"], args, label="fast")
    grid_keys = set((spec.get("grid") or {}).keys())
    allowed = grid_keys | DEFAULT_ALLOWED_KEYS
    winners = winners_from_csv(winners_csv, spec, s1, allowed)

    print(f"[sweep] Stage-1 winners: {len(winners)} / {len(runs_s1)}")
    if not winners:
        print("[sweep][warn] No winners from Stage-1; stopping.")
        return 0

    # Stage 2 (full)
    s2 = stages[1] if len(stages) > 1 else {"name": "stage2", "overrides": {}}
    s2_name = s2.get("name", "stage2")
    s2_over = s2.get("overrides", {}) or {}
    base_s2 = stage_outdirs({**base, **s2_over}, s2_name)

    # Build stage-2 run list from winners
    runs_s2: List[Tuple[str, Dict[str, Any]]] = []
    for w in winners:
        cfg = dict(base_s2); cfg.update(w)
        for k, v in list(cfg.items()):
            if isinstance(v, str) and v.lower() in ("true", "false"):
                cfg[k] = (v.lower() == "true")
        tag = f"{s2_name}::" + "__".join(
            f"{k}={w[k]}" for k in sorted(w.keys()) if k in ("model","backbone","features","color_space")
        )
        runs_s2.append((tag, cfg))

    execute_runs(
        args.config, runs_s2,
        start_index=1, stop_index=len(runs_s2),
        split_root=args.split_root,
        force_auto_test=args.force_auto_test,
        skip_existing=args.skip_existing,
        verbose=args.verbose,
    )

    # Full (stage-2) summary
    full_csv = summarize_stage(spec, base_s2["outdir"], args, label="full") if args.auto_summary else None

    # Optional analysis when not evaluating winners
    if args.analysis and not args.eval_winners and full_csv and os.path.isfile(full_csv):
        run_validation_analysis(
            spec_or_path=spec,
            runs_csv=full_csv,
            split_root=args.split_root,
            test_split_file_arg=args.test_split_file,
            outdir_root=None,
            want_mode="both",
        )

    # Evaluate winners on TEST and re-summarize
    if args.eval_winners:
        chosen = (args.winners_stage or s2_name).lower()
        chosen_outdir = (base_s2 if chosen == s2_name.lower() else base_s1)["outdir"]
        chosen_csv    = summarize_stage(spec, chosen_outdir, args, label=f"{chosen}_pretest")

        # Resolve policy: prefer CLI --eval_top_k, else chosen stage keep_top_k, else report.top_k, else ALL
        chosen_stage_spec = (s2 if chosen == s2_name.lower() else s1) or {}
        eval_top_k = _resolve_eval_top_k(args, spec, chosen_stage_spec, default_all=True)

        # get ALL ranked candidates first
        candidates = select_winners_from_csv(
            chosen_csv,
            keep_top_k=None, keep_top_frac=None,
            mode="min",
            allowed_keys=allowed,
        )
        # collapse to top-1 per grid group
        grid_key_list = list((spec.get("grid") or {}).keys())
        if args.winner_group_keys is None:
            group_keys = grid_key_list or ["model","backbone","features","color_space"]
        else:
            if len(args.winner_group_keys) == 1 and args.winner_group_keys[0].lower() == "none":
                group_keys = []
            else:
                group_keys = list(args.winner_group_keys)
        winners_for_eval = _dedup_top_per_group(candidates, group_keys)
        # take up to eval_top_k groups
        if eval_top_k is not None:
            winners_for_eval = winners_for_eval[:int(eval_top_k)]

        print(f"[eval] Selecting up to top_k={eval_top_k} unique group winner(s) from stage='{chosen}'.")

        if not winners_for_eval:
            print("[eval][warn] No winners to evaluate for selected stage; skipping.")
            return 0

        stage_out = (base_s2 if chosen == s2_name.lower() else base_s1)["outdir"]
        stage_log = (base_s2 if chosen == s2_name.lower() else base_s1)["logdir"]

        print("[eval][debug] winners (grouped):", [{k: w.get(k) for k in group_keys} for w in winners_for_eval])
        ok, fail = eval_winners_flow(
            winners=winners_for_eval,
            base_cfg=base,
            stage_outdir=stage_out,
            stage_logdir=stage_log,
            args=args,
            # IMPORTANT: preserve YAML insertion order to match training run tags
            tag_keys=list((spec.get("grid") or {}).keys()),
        )
        print(f"[eval] winners: ok={ok} fail={fail}")

        pre_csv, test_csv = (None, None)
        if args.auto_summary:
            pre_csv, test_csv = re_summarize_after_test(spec, stage_out, args)

        runs_csv_for_analysis = pre_csv or test_csv
        if args.analysis and runs_csv_for_analysis and os.path.isfile(runs_csv_for_analysis):
            outdir_root = os.path.join(args.summary_outdir, "analysis") if args.summary_outdir else None
            run_validation_analysis(
                 spec_or_path=spec,
                 runs_csv=runs_csv_for_analysis,
                 split_root=args.split_root,
                 test_split_file_arg=args.test_split_file or "AUTO",
                 outdir_root=outdir_root,
                 want_mode="both",
             )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
