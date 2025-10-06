# src/engine/kfold.py
#!/usr/bin/env python3
"""
K-Fold runner with class-wise grouping (Pantone ID from basename).

Usage examples:
  # 5-fold CV, write split files, fold-aware outdirs/logdirs
  python -m src.engine.kfold \
      --images_dir ... --json_dir ... --labels_csv ... \
      --model film --backbone resnet18 --color_space rgb \
      --kfold_splits 5 --kfold_seed 100 \
      --save_split_dir ./splits/cv5 \
      --outdir ./checkpoints/cv5 --logdir ./runs/cv5

  # Run only folds 2 and 4 (useful for resuming)
  python -m src.engine.kfold ... --folds 2 4
"""
from __future__ import annotations

import os, json
from typing import List, Sequence, Tuple, Optional
import numpy as np
from sklearn.model_selection import GroupKFold
from torch.utils.data import Subset

from .config import RunConfig, DataConfig
from .train import fit, build_argparser as build_train_argparser
from .datamodule import DataModule, DataModuleConfig
from ..data.dataset import ImageMetadataDataset
from ..data.splitters import save_splits  # uses the same JSON format as experiment/sweeps

# -----------------------
# Helpers
# -----------------------
def _class_labels(ds: ImageMetadataDataset) -> List[int]:
    # Pantone class from basename: "001.jpg" -> 1
    return [int(s.basename.lstrip("0") or "0") for s in ds.samples]

def _folder_labels(ds: ImageMetadataDataset) -> List[str]:
    # folder names of samples
    return [s.folder for s in ds.samples]

def _fold_paths(root: str, k: int, fold_idx: int) -> str:
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"fold{fold_idx}.json")

def _fold_tag(fold_idx: int, k: int) -> str:
    return f"fold{fold_idx}-of-{k}"

# -----------------------
# CLI
# -----------------------
def build_argparser():
    p = build_train_argparser()  # reuse all train/experiment flags
    p.add_argument("--kfold_splits", type=int, default=5, help="Number of folds.")
    p.add_argument("--kfold_seed", type=int, default=100, help="Seed (used for class ordering).")
    p.add_argument("--folds", type=int, nargs="*", default=None,
                   help="Subset of folds to run (1-based). If omitted, run all.")
    p.add_argument("--save_split_dir", type=str, default=None,
                   help="If set, write per-fold JSON indices here and reuse them for each fold.")
    p.add_argument("--no_suffix_outdirs", action="store_true", default=False,
                   help="Do not append '/foldX' to outdir/logdir.")
    p.add_argument("--monitor", type=str, default="de00",
                   help="Metric to report across folds (use the validation key, default: de00).")
    return p

# -----------------------
# Main
# -----------------------
def main():
    args = build_argparser().parse_args()

    # Build top-level RunConfig (single source of truth)
    run = RunConfig.from_namespace(args)

    # Prepare a base DataModule to access the fully-built dataset once
    d: DataConfig = run.data
    base_dm = DataModule(DataModuleConfig(
        images_dir=d.images_dir,
        json_dir=d.json_dir,
        labels_csv=d.labels_csv,
        batch_size=run.loop.batch_size,
        val_split=d.val_split,           # not used when we override indices
        workers=d.workers,
        seed=args.kfold_seed,
        hidden_classes_cnt=d.hidden_classes_cnt,
        group_split=d.group_split or 'class',               # KFold controls split below
        color_space=d.color_space,
        features=d.features,
        pretrained=d.pretrained,
        backbone_name=d.backbone_name or run.model.backbone,
        include_test=False,              # keep test for a final holdout (optional)
        test_per_class=d.test_per_class,
        split_file=None,
        save_splits_flag=False,
    )).setup()

    ds = base_dm.ds

    if d.group_split == 'folder':
        groups = np.array(_folder_labels(ds))
    else:
        groups = np.array(_class_labels(ds))
    k = int(args.kfold_splits)

    gkf = GroupKFold(n_splits=k)
    # Deterministic fold order: sklearn GroupKFold is deterministic; seed governs any prior shuffling you might add.
    X_dummy = np.zeros((len(ds), 1))
    folds: List[Tuple[np.ndarray, np.ndarray]] = list(gkf.split(X_dummy, y=None, groups=groups))

    # Select subset of folds if requested
    selected: Sequence[int] = range(1, k + 1) if not args.folds else sorted(set(int(x) for x in args.folds))
    metrics: List[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        if fold_idx not in selected:
            continue

        tag = _fold_tag(fold_idx, k)
        print(f"[kfold] {tag}: train={len(tr_idx)} val={len(va_idx)}")

        # Option A: persistent split files (best for reproducibility + sweeps)
        split_file: Optional[str] = None
        if args.save_split_dir:
            split_file = _fold_paths(args.save_split_dir, k, fold_idx)
            save_splits(
                split_file,
                train_idx=tr_idx.tolist(),
                val_idx=va_idx.tolist(),
                test_idx=None,
                meta={"kfold_splits": k, "fold": fold_idx, "seed": int(args.kfold_seed),
                      "group": d.group_split}
            )
            # Build a *fresh* DataModule that reads the split file
            dm = DataModule(DataModuleConfig(
                images_dir=d.images_dir,
                json_dir=d.json_dir,
                labels_csv=d.labels_csv,
                batch_size=run.loop.batch_size,
                val_split=d.val_split,
                workers=d.workers,
                seed=d.seed,
                hidden_classes_cnt=d.hidden_classes_cnt,
                group_split=d.group_split,
                color_space=d.color_space,
                features=d.features,
                pretrained=d.pretrained,
                backbone_name=d.backbone_name or run.model.backbone,
                include_test=False,
                test_per_class=d.test_per_class,
                split_file=split_file,
                save_splits_flag=False,
            )).setup()
        else:
            # Option B: reuse the in-memory dataset and override subsets
            dm = base_dm
            dm.train_ds = Subset(dm.ds, tr_idx.tolist())
            dm.val_ds   = Subset(dm.ds, va_idx.tolist())
            #dm.n_train, dm.n_val = len(tr_idx), len(va_idx)

        # Per-fold outdirs/logdirs unless suppressed
        if args.no_suffix_outdirs:
            fold_run = run
        else:
            from dataclasses import replace
            fold_run = replace(
                run,
                log=replace(run.log, logdir=os.path.join(run.log.logdir, tag)),
                ckpt=replace(run.ckpt, outdir=os.path.join(run.ckpt.outdir, tag)),
            )

        # Train this fold
        result = fit(fold_run, dm)
        metric_key = f"val_{args.monitor}_last" if f"val_{args.monitor}_last" in result else "val_de00_best"
        fold_metric = result.get(metric_key) or result.get("val_de00_best")
        if fold_metric is not None:
            metrics.append(float(fold_metric))

        # Persist a tiny fold report next to checkpoints
        try:
            where = fold_run.ckpt.outdir if hasattr(fold_run, "ckpt") else run.ckpt.outdir
            os.makedirs(where, exist_ok=True)
            with open(os.path.join(where, f"{tag}_summary.json"), "w") as f:
                json.dump({"fold": fold_idx, "kfold_splits": k, "monitor": args.monitor,
                           "metric": fold_metric, "result": result}, f, indent=2)
        except Exception:
            pass

    # Aggregate
    if metrics:
        mu = float(np.mean(metrics))
        sd = float(np.std(metrics, ddof=1)) if len(metrics) > 1 else 0.0
        print(f"[kfold] {len(metrics)}/{len(selected)} folds finished. {args.monitor}: "
              f"mean={mu:.4f} std={sd:.4f}")
    else:
        print("[kfold] No fold metrics collected.")

if __name__ == "__main__":
    main()
