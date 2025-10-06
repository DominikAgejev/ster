# src/engine/cvgrid.py
#!/usr/bin/env python3
from __future__ import annotations

import os, json, itertools, subprocess, sys
from typing import Dict, Any, List
import yaml
import numpy as np

from .datamodule import DataModule, DataModuleConfig
from ..data.splitters import get_test_indices, save_splits

def make_cli_args(d: Dict[str, Any]) -> List[str]:
    """Convert a dict of hparams into CLI args, handling bools and lists like sweep.py."""
    args: List[str] = []
    for k, v in d.items():
        if k in ("images_dir","json_dir","labels_csv","outdir","logdir"):
            # These are passed explicitly elsewhere where needed
            continue
        # Your argparse expects underscores in option names
        flag = f"--{k}"
        # BooleanOptionalAction pairs used in train.py
        if k == "restore_best_on_stop":
            args.append("--restore_best_on_stop" if v else "--no-restore_best_on_stop")
            continue
        if k == "pretrained":
            args.append("--pretrained" if v else "--no-pretrained")
            continue
        if isinstance(v, bool):
            if v:
                args.append(flag)
            continue
        if isinstance(v, (list, tuple)):
            vals = list(map(str, v))
            # nargs-style flags: --included_folders a b c
            args.append(flag)
            args.extend(vals)
            # If any list item starts with '-', join as CSV and use --flag=<csv>
            if any(s.startswith("-") for s in vals):
                args.append(f"{flag}=" + ",".join(vals))
            else:
                # nargs-style flags: --included_folders a b c
                args.append(flag)
                args.extend(vals)
            continue
        s = str(v)
        # Standard "--flag value"
        args += [flag, s]
        # If scalar value starts with '-', force --flag=<value> to avoid argparse mis-parsing
        if s.startswith("-"):
            args.append(f"{flag}={s}")
        else:
            args += [flag, s]
    return args

def _grid_items(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not grid: return [dict()]
    keys = list(grid.keys())
    vals = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    out = []
    for combo in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, combo)})
    return out

def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    z = dict(base or {})
    z.update(over or {})
    return z

def _run_cli(mod: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", mod] + args
    print("[cvgrid] RUN:", " ".join(cmd))
    return subprocess.call(cmd, env=os.environ.copy())

def _fold_summary_paths(outdir: str, k: int) -> List[str]:
    return [os.path.join(outdir, f"fold{i}-of-{k}_summary.json") for i in range(1, k+1)]

def _collect_fold_metric(outdir: str, monitor: str) -> float | None:
    """Recursively scan for per-fold *_summary.json files and average 'metric'."""
    vals = []
    if not os.path.isdir(outdir):
        return None
    for root, _dirs, files in os.walk(outdir):
        for name in files:
            if not name.endswith("_summary.json"):
                continue
            p = os.path.join(root, name)
            try:
                with open(p, "r") as f:
                    rec = json.load(f)
                m = rec.get("metric", rec.get(f"val_{monitor}"))
                if m is not None:
                    vals.append(float(m))
            except Exception:
                pass
    if not vals:
        return None
    return float(np.mean(vals))

def _missing_folds(outdir: str, k: int) -> List[int]:
    """Check both top-level and fold subdirs for each fold's summary."""
    missing = []
    for i in range(1, k + 1):
        top = os.path.join(outdir, f"fold{i}-of-{k}_summary.json")
        sub = os.path.join(outdir, f"fold{i}-of-{k}", f"fold{i}-of-{k}_summary.json")
        if not (os.path.isfile(top) or os.path.isfile(sub)):
            missing.append(i)
    return missing

def main():
    import argparse
    ap = argparse.ArgumentParser("Cross-validation grid orchestrator")
    ap.add_argument("--config", required=True, help="cvgrid YAML with 'base' and 'groups'")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--split_dir", type=str, default="./splits/cvgrid",
                    help="Where to write per-combo fold split JSONs")
    ap.add_argument("--test_split_file", type=str, required=True,
                    help="Persistent test split JSON, or 'AUTO' to auto-create one.")
    ap.add_argument("--train_full_split_file", type=str, default=None,
                    help="Train-full split JSON (train = all non-test). If omitted and test is AUTO, will auto-create.")
    ap.add_argument("--summary_out", type=str, default="./cvgrid_summary.json")
    # NEW: skip-existing + autosummary
    ap.add_argument("--skip-existing", action="store_true", default=False,
                    help="Skip combos whose outdir already has all fold summaries; "
                         "skip final train if final_summary.json exists.")
    ap.add_argument("--autosummary", action="store_true", default=False,
                    help="After writing summary_out JSON, also produce CSV and/or Markdown.")
    ap.add_argument("--autosummary_csv", type=str, default=None,
                    help="Optional path for autosummary CSV (defaults to summary_out with .csv).")
    ap.add_argument("--autosummary_md", type=str, default=None,
                    help="Optional path for autosummary Markdown (defaults to summary_out with .md).")
    ap.add_argument("--autosummary_module", type=str, default="src.analysis.summarize_sweep",
                    help="Module to run for autosummary. Use 'src.analysis.summarize_sweep' to reuse the sweep summarizer.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        spec = yaml.safe_load(f) or {}

    # Extract top-level config parts early (needed by AUTO below)
    groups: Dict[str, Dict[str, Any]] = spec["groups"]
    base_common: Dict[str, Any] = spec.get("base", {}) or {}
    report_spec: Dict[str, Any] = spec.get("report", {}) or {}

    # --- optional: auto-create test/train_full splits ---
    if str(args.test_split_file).upper() == "AUTO":
        base_for_splits = base_common  # test split should be invariant across combos
        split_root = os.path.join(args.split_dir, "_GLOBAL")
        os.makedirs(split_root, exist_ok=True)
        auto_test_p  = os.path.join(split_root, "test.json")
        auto_train_p = os.path.join(split_root, "train_full.json")

        # Build a dataset once to compute indices
        dm = DataModule(DataModuleConfig(
            images_dir=base_for_splits["images_dir"],
            json_dir=base_for_splits["json_dir"],
            labels_csv=base_for_splits["labels_csv"],
            batch_size=base_for_splits.get("batch_size", 32),
            val_split=base_for_splits.get("val_split", 0.15),
            workers=base_for_splits.get("workers", 0),
            seed=args.seed,
            hidden_classes_cnt=base_for_splits.get("hidden_classes_cnt", 0),
            group_split=base_for_splits.get("group_split", None),
            color_space=base_for_splits.get("color_space", "rgb"),
            features=base_for_splits.get("features", "image+mean+meta"),
            pretrained=base_for_splits.get("pretrained", True),
            # Use a safe default if backbone is not in top-level base (since grid may vary it)
            backbone_name=base_for_splits.get("backbone", "resnet18"),
            include_test=False,
            test_per_class=base_for_splits.get("test_per_class", 3),
            split_file=None,
            save_splits_flag=False,
        )).setup()

        ds = dm.ds
        n_test_per_class = base_for_splits.get("test_per_class", 3)
        test_idx = get_test_indices(ds, n_samples_per_class=n_test_per_class, seed=args.seed)
        all_idx  = list(range(len(ds)))
        train_full_idx = sorted(set(all_idx) - set(test_idx))

        save_splits(auto_test_p, train_idx=[], val_idx=[], test_idx=test_idx,
                    meta={"note":"AUTO from cvgrid", "seed":int(args.seed), "per_class":int(n_test_per_class)})
        save_splits(auto_train_p, train_idx=train_full_idx, val_idx=[], test_idx=None,
                    meta={"note":"AUTO from cvgrid (non-test indices)", "seed":int(args.seed)})

        args.test_split_file = auto_test_p
        if not args.train_full_split_file:
            args.train_full_split_file = auto_train_p

    report_keep: List[str] = list(report_spec.get("keep_from_base", []) or [])
    extra_group_keys: List[str] = list(report_spec.get("extra_group_keys", []) or [])

    results: List[Dict[str, Any]] = []

    all_grid_keys: set[str] = set()
    for gname, block in groups.items():
        base = _merge(base_common, block.get("base", {}))
        grid = block.get("grid", {})
        if isinstance(grid, dict):
            all_grid_keys |= set(grid.keys())
        combos = _grid_items(grid)
        print(f"[cvgrid] Group {gname}: {len(combos)} combo(s)")

        outdir_root = base.get("outdir", f"./checkpoints/{gname}")
        logdir_root = base.get("logdir", f"./runs/{gname}")
        os.makedirs(outdir_root, exist_ok=True)
        os.makedirs(logdir_root, exist_ok=True)

        group_records: List[tuple[Dict[str, Any], float]] = []

        for h in combos:
            cfg = _merge(base, h)
            # tag like "backbone=resnet18__lr=0.001"
            tag = "__".join([f"{k}={v}" for k, v in sorted(h.items())])
            outdir = os.path.join(outdir_root, tag)
            logdir = os.path.join(logdir_root, tag)
            os.makedirs(outdir, exist_ok=True); os.makedirs(logdir, exist_ok=True)

            split_dir = os.path.join(args.split_dir, gname, tag)
            os.makedirs(split_dir, exist_ok=True)

            # --- skip-existing for folds
            missing = _missing_folds(outdir, args.folds)
            if args.skip_existing and not missing:
                print(f"[cvgrid] skip-existing: all {args.folds} fold summaries present for {gname}/{tag}")
            else:
                kfold_args = [
                    "--images_dir", cfg["images_dir"], "--json_dir", cfg["json_dir"], "--labels_csv", cfg["labels_csv"],
                    "--kfold_splits", str(args.folds), "--kfold_seed", str(args.seed),
                    "--save_split_dir", split_dir,
                    "--outdir", outdir, "--logdir", logdir,
                    "--monitor", base.get("ckpt_monitor", "de00"),
                ]
                # if some folds are missing and skip-existing is ON, run only those
                if args.skip_existing and missing:
                    kfold_args += ["--folds"] + [str(i) for i in missing]

                # add remaining hparams once, robustly
                # Do NOT pass include_test into kfold folds
                kfold_args += make_cli_args({
                    k: v for k, v in cfg.items()
                    if k not in ("images_dir","json_dir","labels_csv","outdir","logdir","include_test")
                })

                rc = _run_cli("src.engine.kfold", kfold_args)
                if rc != 0:
                    print(f"[cvgrid][warn] kfold failed for {tag} (rc={rc}); skipping")
                    continue

            mean_cv = _collect_fold_metric(outdir, base.get("ckpt_monitor", "de00"))
            if mean_cv is None:
                print(f"[cvgrid][warn] no fold summary metrics found in {outdir}; skipping")
                continue
            group_records.append((cfg, mean_cv))
            results.append({"group": gname, "hparams": cfg, "mean_cv": mean_cv})

        if not group_records:
            print(f"[cvgrid][warn] no successful CV combos in {gname}; continuing")
            continue

        # pick best by minimal mean_cv
        best_cfg, best_cv = min(group_records, key=lambda t: t[1])
        print(f"[cvgrid] {gname} best mean_cv={best_cv:.4f} with {best_cfg}")

        # final train on full train + test
        final_out = os.path.join(outdir_root, "final_fulltrain")
        final_log = os.path.join(logdir_root, "final_fulltrain")
        os.makedirs(final_out, exist_ok=True); os.makedirs(final_log, exist_ok=True)

        final_summary_p = os.path.join(final_out, "final_summary.json")
        if args.skip_existing and os.path.isfile(final_summary_p):
            print(f"[cvgrid] skip-existing: found {final_summary_p}; skipping final train for {gname}")
        else:
            exp_args = [
                "--images_dir", best_cfg["images_dir"], "--json_dir", best_cfg["json_dir"], "--labels_csv", best_cfg["labels_csv"],
                "--outdir", final_out, "--logdir", final_log,
                "--split_file", args.train_full_split_file,
                "--eval_test_after", "--test_split_file", args.test_split_file,
            ]
            # Here we DO pass include_test (so your final model can train/eval with test)
            exp_args += make_cli_args({
                k: v for k, v in best_cfg.items()
                if k not in ("images_dir","json_dir","labels_csv","outdir","logdir")
            })
            rc = _run_cli("src.engine.experiment", exp_args)
            if rc != 0:
                print(f"[cvgrid][warn] final train failed for group {gname} (rc={rc})")

        results.append({"group": gname, "best_hparams": best_cfg,
                        "best_mean_cv": best_cv, "final_run_dir": final_out})

    # write cvgrid summary json
    try:
        with open(args.summary_out, "w") as f:
            json.dump({
                "results": results,
                "report": {
                    "keep_from_base": report_keep,
                    "extra_group_keys": extra_group_keys,
                    "grid_keys": sorted(list(all_grid_keys)),
                }
            }, f, indent=2)
        print(f"[cvgrid] wrote summary -> {args.summary_out}")
    except Exception as e:
        print(f"[cvgrid][warn] could not write summary_out: {e}")

    # --- autosummary ---
    if args.autosummary:
        # Branch: prefer provided module
        if args.autosummary_module == "src.analysis.summarize_sweep":
            # Summarize across final_fulltrain outdirs with the same sweep summarizer
            final_dirs = [rec["final_run_dir"] for rec in results if "final_run_dir" in rec]
            csv_out = args.autosummary_csv or (os.path.splitext(args.summary_out)[0] + "_sweep_summary.csv")
            grp_out = os.path.splitext(csv_out)[0] + "_groups.csv"
            md_out  = args.autosummary_md  or (os.path.splitext(args.summary_out)[0] + "_sweep_summary.md")
            cmd = [sys.executable, "-m", "src.analysis.summarize_sweep",
                   "--ckpt_dir", *final_dirs,
                   "--monitor", base_common.get("ckpt_monitor", "de00"),
                   "--mode", "min", "--skip_bad", "--auto_group",
                  "--out_csv", csv_out, "--out_groups_csv", grp_out, "--md_report", md_out]
            if report_keep: cmd += ["--report_keep"] + report_keep
            if all_grid_keys: cmd += ["--grid_keys"] + sorted(list(all_grid_keys))
            if extra_group_keys: cmd += ["--extra_group_keys"] + extra_group_keys
            print("[cvgrid] autosummary(sweep) RUN:", " ".join(cmd))
            rc = subprocess.call(cmd, env=os.environ.copy())
        else:
            csv_out = args.autosummary_csv or (os.path.splitext(args.summary_out)[0] + ".csv")
            md_out  = args.autosummary_md  or (os.path.splitext(args.summary_out)[0] + ".md")
            rc = _run_cli(args.autosummary_module, [
                "--cvgrid_json", args.summary_out,
                "--out_csv", csv_out,
                "--out_md", md_out
            ])
        if rc == 0:
            print(f"[cvgrid] autosummary wrote:\n  - {csv_out}\n  - {md_out}")
        else:
            print(f"[cvgrid][warn] autosummary failed (rc={rc})")

if __name__ == "__main__":
    main()
