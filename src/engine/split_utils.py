# src/engine/split_utils.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple

from .datamodule import DataModule, DataModuleConfig
from ..data.splitters import get_test_indices, save_splits
from ..data.folder_filters import normalize_folder_filters

def _namespace_for_run(run_cfg: Dict[str, Any]) -> str:
    """
    Build a stable namespace string to store per-filter splits.
    Example: IF=focus+pixel-6__EF=NONE__HC=0__SEED=101
    """
    def _norm_list(x: Any) -> str:
        xs = normalize_folder_filters(x)
        return "NONE" if not xs else "+".join(sorted(xs))

    inc = _norm_list(run_cfg.get("included_folders"))
    exc = _norm_list(run_cfg.get("excluded_folders"))
    hid = int(run_cfg.get("hidden_classes_cnt", 0) or 0)
    seed = int(run_cfg.get("seed", 0) or 0)
    return f"IF={inc}__EF={exc}__HC={hid}__SEED={seed}"


def ensure_auto_splits_for_run(
    run_cfg: Dict[str, Any],
    *,
    split_root: str,
    test_per_class: int | None = None,
) -> Tuple[str, str]:
    """
    Create (idempotently) persistent TEST and TRAIN_FULL JSON split files
    for the *filtered* dataset implied by `run_cfg`.

    Returns:
        (test_json_path, train_full_json_path)
    """
    ns = _namespace_for_run(run_cfg)
    out_dir = os.path.join(split_root, ns)
    os.makedirs(out_dir, exist_ok=True)
    test_json = os.path.join(out_dir, "test.json")
    train_full_json = os.path.join(out_dir, "train_full.json")

    # Already present → reuse
    if os.path.exists(test_json) and os.path.exists(train_full_json):
        return test_json, train_full_json

    # Materialize dataset once via DataModule so we respect *exactly*
    # the same filters/transforms/sanitization used in training.
    dm = DataModule(DataModuleConfig(
        images_dir=run_cfg["images_dir"],
        json_dir=run_cfg["json_dir"],
        labels_csv=run_cfg["labels_csv"],
        batch_size=int(run_cfg.get("batch_size", 32)),
        val_split=0.0,
        workers=int(run_cfg.get("workers", 0)),
        seed=int(float(run_cfg.get("seed", 100))),
        hidden_classes_cnt=int(float(run_cfg.get("hidden_classes_cnt", 0))),
        group_split=run_cfg.get("group_split"),
        color_space=str(run_cfg.get("color_space", "rgb")),
        features=str(run_cfg.get("features", "image")),
        pretrained=bool(run_cfg.get("pretrained", True)),
        backbone=str(run_cfg.get("backbone", "smallcnn")),
        include_test=False,
        excluded_folders=normalize_folder_filters(run_cfg.get("excluded_folders")),
        included_folders=normalize_folder_filters(run_cfg.get("included_folders")),
        test_per_class=int(run_cfg.get("test_per_class", 3)),
        split_file=None,
        save_splits_flag=False,

        # IMPORTANT: disable meta text encoder during split provisioning
        meta_encoder="none",
        # (do NOT pass meta_model_name/layers/template/batch; defaults are fine but unused)
    )).setup()

    ds = dm.ds  # full pool under the requested filters

    test_idx = get_test_indices(
        ds=ds,
        n_samples_per_class=int(test_per_class or run_cfg.get("test_per_class", 3)),
        seed=int(run_cfg.get("seed", 101)),
        hidden_class_cnt=int(run_cfg.get("hidden_classes_cnt", 0) or 0),
        excluded_folders=normalize_folder_filters(run_cfg.get("excluded_folders")),
        included_folders=normalize_folder_filters(run_cfg.get("included_folders")),
    )

    # Persist TEST
    save_splits(test_json, train_idx=[], val_idx=[], test_idx=test_idx)    

    # TRAIN_FULL = everything not in test
    all_idx = list(range(len(ds)))
    test_set = set(test_idx)
    train_full_idx = [i for i in all_idx if i not in test_set]

    save_splits(train_full_json, train_idx=train_full_idx, val_idx=[], test_idx=[])

    return test_json, train_full_json
