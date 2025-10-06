# src/engine/datamodule.py
"""Centralized data prep: dataset construction, splitting, normalization, and loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Import from your repo packages
from ..models.backbones import resolve_preprocess
from ..data.dataset import ImageMetadataDataset
from ..data.transforms import ToTensorTransform, make_rgb_normalizer
from ..features.meta_text import build_meta_cache_path, render_meta_texts, embed_texts
from ..data.splitters import (
    get_test_indices, train_val_split, make_test_album,
    save_splits, load_splits
)
from ..data.folder_filters import normalize_folder_filters

from ..diag import D

@dataclass
class DataModuleConfig:
    images_dir: str
    json_dir: str
    labels_csv: str
    batch_size: int = 32
    val_split: float = 0.15
    workers: int = 0
    seed: int = 100
    hidden_classes_cnt: int = 0
    group_split: str | None = None
    color_space: str = "lab"            # "lab" or "rgb"
    features: str = "image+mean+meta"   # preset selector
    pretrained: bool = True             # for timm backbones
    backbone: str = "smallcnn"     # to resolve RGB mean/std when pretrained
    include_test: bool = False
    test_per_class: int = 3
    excluded_folders:  list[str] = field(default_factory=list)
    included_folders:  list[str] = field(default_factory=list)
    split_file: str | None = None
    save_splits_flag: bool = False      # write split_file if provided
    # --- NEW: text meta encoder options (used in setup()) ---
    meta_encoder: str = "none"               # none | flair
    meta_model_name: str = "jhu-clsp/ettin-encoder-17m"
    meta_layers: str = "-1,-2,-3,-4"
    meta_text_template: str = "compact"      # compact | kv | json
    meta_batch_size: int = 64


class DataModule:
    def __init__(self, cfg: DataModuleConfig):
        self.cfg = cfg
        self.ds = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._meta_cols: List[str] = []
        self._meta_dim: int = 0

    # ---- Properties ----
    @property
    def meta_cols(self):
        return getattr(self, "_meta_cols", [])

    @property
    def meta_dim(self):
        return getattr(self, "_meta_dim", 0)

    @property
    def n_train(self) -> int:
        return len(self.train_ds) if self.train_ds is not None else 0

    @property
    def n_val(self) -> int:
        return len(self.val_ds) if self.val_ds is not None else 0
    
    # ---- Setup ----
    def setup(self) -> "DataModule":
        # ============ Dataset (no transforms yet) ============
        self.ds = ImageMetadataDataset(
            images_dir=self.cfg.images_dir,
            json_dir=self.cfg.json_dir,
            labels_path=self.cfg.labels_csv,
            color_space=self.cfg.color_space,
            transform=None,  # attach after we decide normalization
        )

        # ============ Normalize folder filters exactly once ============
        exc = normalize_folder_filters(self.cfg.excluded_folders)
        inc = normalize_folder_filters(self.cfg.included_folders)

        D.splits.filters(excluded=exc, included=inc)

        # ============ Splits: load or create ============
        train_idx = val_idx = test_idx = None
        loaded = None
        if self.cfg.split_file:
            try:
                loaded = load_splits(self.cfg.split_file)
            except Exception:
                loaded = None

        if loaded:
            train_idx = loaded.get("train", []) or []
            val_idx   = loaded.get("val",   []) or []
            test_idx  = loaded.get("test",  None)
            if test_idx is not None:
                self.test_ds = Subset(self.ds, test_idx)
                # Optional: album for quick visual sanity
                make_test_album(self.test_ds)
            # NEW: if only TEST present, derive train/val from remainder
            if (not train_idx) and (not val_idx) and (test_idx is not None):
                exc = normalize_folder_filters(self.cfg.excluded_folders)
                inc = normalize_folder_filters(self.cfg.included_folders)
                train_idx, val_idx = train_val_split(
                    ds=self.ds, val_split=float(self.cfg.val_split), seed=int(self.cfg.seed),
                    test_indices=test_idx, hidden_classes_cnt=int(self.cfg.hidden_classes_cnt),
                    excluded_folders=exc, included_folders=inc, group_split=self.cfg.group_split,
                )
        else:
            # Optional dedicated TEST slice per class under the same filters
            if self.cfg.include_test:
                test_idx = get_test_indices(
                    ds=self.ds,
                    n_samples_per_class=int(self.cfg.test_per_class),
                    seed=self.cfg.seed,
                    hidden_class_cnt=int(self.cfg.hidden_classes_cnt),
                    excluded_folders=exc,
                    included_folders=inc,
                )
                self.test_ds = Subset(self.ds, test_idx)
                make_test_album(self.test_ds)

            # Train/val from the remaining pool
            train_idx, val_idx = train_val_split(
                ds=self.ds,
                val_split=float(self.cfg.val_split),
                seed=int(self.cfg.seed),
                test_indices=test_idx,
                hidden_classes_cnt=int(self.cfg.hidden_classes_cnt),
                excluded_folders=exc,
                included_folders=inc,
                group_split=self.cfg.group_split,
            )

            # Persist splits if requested
            if self.cfg.split_file and self.cfg.save_splits_flag:
                save_splits(
                    self.cfg.split_file,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    meta={
                        "seed": int(self.cfg.seed),
                        "val_split": float(self.cfg.val_split),
                        "group_split": bool(self.cfg.group_split),
                        "hidden_classes_cnt": int(self.cfg.hidden_classes_cnt),
                        "test_per_class": int(self.cfg.test_per_class),
                        "included_folders": inc,
                        "excluded_folders": exc,
                    },
                )

        # DIAG: split counts, folder hist, overlap guards
        image_paths = [s.image_path for s in self.ds.samples]
        D.splits.report_counts(n_train=len(train_idx), n_val=len(val_idx), n_test=(len(test_idx) if test_idx is not None else None))
        D.splits.overlap_check(image_paths=image_paths, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        D.splits.folder_hist(image_paths, train_idx, name="train")
        D.splits.folder_hist(image_paths, val_idx,   name="val")
        if test_idx is not None:
            D.splits.folder_hist(image_paths, test_idx, name="test")
        D.splits.group_overlap(self.ds.metadatas, train_idx, val_idx, group_key=self.cfg.group_split)

        # Materialize subsets
        self.train_ds = Subset(self.ds, train_idx)
        self.val_ds   = Subset(self.ds, val_idx)

        print(
            f"[split] train={len(self.train_ds)} val={len(self.val_ds)}"
            + (f" test={len(self.test_ds)}" if self.test_ds is not None else "")
        )

        # ============ Feature selection ============
        df = self.ds.metadatas

        # Image mean triplet depends on color space
        if self.cfg.color_space == "lab":
            img_mean_cols = [c for c in ("L_mean_img", "a_mean_img", "b_mean_img") if c in df.columns]
        else:
            img_mean_cols = [c for c in ("R_mean_img", "G_mean_img", "B_mean_img") if c in df.columns]

        # All numeric (drop basename + chosen mean triplet)
        all_numeric = df.select_dtypes(include=["number"]).columns.tolist()
        for c in ["basename", *img_mean_cols]:
            if c in all_numeric:
                all_numeric.remove(c)

        # Keep robust engineered meta + camera make one-hots
        allowed_base = ["iso_log", "shutter_s_log", "brightness_ev", "white_balance_auto", "meta_missing"]
        allowed_prefixes = ["Make_"]

        def _is_allowed(col: str) -> bool:
            return (col in allowed_base) or any(col.startswith(p) for p in allowed_prefixes)

        meta_only = [c for c in all_numeric if _is_allowed(c)]

        mode = self.cfg.features  # "image" | "image+mean" | "image+meta" | "image+mean+meta"
        if mode == "image":
            used_cols: List[str] = []
        elif mode == "image+mean":
            used_cols = img_mean_cols
        elif mode == "image+meta":
            used_cols = meta_only
        elif mode == "image+mean+meta":
            used_cols = img_mean_cols + meta_only
        else:
            raise ValueError(f"Unknown features preset: {mode}")

        # ============ OPTIONAL: text meta encoder (replace numeric meta) ============
        if self.cfg.meta_encoder == "flair" and ("meta" in mode):
            # 1) Render rows -> text
            texts = render_meta_texts(self.ds.metadatas, template=self.cfg.meta_text_template)

            # 2) Stable sample IDs (aligned with df index)
            sample_ids = [s.image_path for s in self.ds.samples]

            # 3) Collision-proof cache path
            cache_dir = os.path.join(self.cfg.images_dir, ".meta_cache")
            inc_key = ",".join(inc) if inc else None
            cache_key_path = build_meta_cache_path(
                cache_dir=cache_dir,
                model_name=self.cfg.meta_model_name,
                layers=self.cfg.meta_layers,
                template=self.cfg.meta_text_template,
                color_space=self.cfg.color_space,
                features=self.cfg.features,
                included_folders=inc_key,
                seed=self.cfg.seed,
                sample_ids=sample_ids,
            )

            # 4) Embed (CPU/GPU auto) with on-disk cache
            embs = embed_texts(
                texts,
                model_name=self.cfg.meta_model_name,
                layers=self.cfg.meta_layers,
                batch_size=int(self.cfg.meta_batch_size),
                device=None,
                cache_key_path=cache_key_path,
            )

            # 5) Replace numeric meta with embedding columns
            d = int(embs.shape[1])
            txt_cols = [f"meta_txt_{i:04d}" for i in range(d)]
            embs_df = pd.DataFrame(embs, columns=txt_cols, index=self.ds.metadatas.index)

            # Drop numeric meta we were going to use (img_mean + meta_only) to avoid duplication
            drop_cols = [c for c in (img_mean_cols + meta_only) if c in self.ds.metadatas.columns]
            if drop_cols:
                self.ds.metadatas = self.ds.metadatas.drop(columns=drop_cols)

            # Concatenate and defragment
            self.ds.metadatas = pd.concat([self.ds.metadatas, embs_df], axis=1, copy=False)
            self.ds.metadatas = self.ds.metadatas.copy()

            # From here on, "meta" means the text embedding only
            used_cols = txt_cols
            
            # DIAG: which meta columns are in play
            D.norm.meta_columns(used_cols=used_cols, img_mean_cols=img_mean_cols, meta_only_cols=meta_only)


        # ============ Standardize continuous meta (fit on TRAIN only) ============
        # Only standardize true continuous fields; keep binary flags (0/1) intact.
        df = self.ds.metadatas  # refresh after any embedding/drop
        cont_candidates = ["iso_log", "shutter_s_log", "brightness_ev"] + img_mean_cols
        cont_cols = [c for c in cont_candidates if (c in used_cols) and (c in df.columns)]

        if cont_cols:
            if hasattr(self.train_ds, "indices"):
                tr_rows = self.train_ds.indices
            else:
                tr_rows = range(len(self.train_ds))

            tr_df = df.loc[tr_rows, cont_cols].astype(float)
            mu = tr_df.mean(0, skipna=True)
            sigma = tr_df.std(0, skipna=True).replace(0, 1.0)

            # Impute NaNs with train means, then standardize (applied to full df)
            self.ds.metadatas[cont_cols] = df[cont_cols].astype(float).fillna(mu)
            self.ds.metadatas[cont_cols] = (self.ds.metadatas[cont_cols] - mu) / sigma

            # DIAG: record params and validate train standardization
            D.norm.record_mu_sigma(mu, sigma)
            D.norm.check_standardization(df=self.ds.metadatas, cont_cols=cont_cols, train_rows=tr_rows)
            D.norm.check_binary_flags(self.ds.metadatas, cols=["meta_missing","white_balance_auto"], only_if_used=self.ds.meta_numeric_cols)

        # Expose meta columns/dim to callers
        self.ds.meta_numeric_cols = used_cols
        self._meta_cols = used_cols
        self._meta_dim = len(used_cols)

        # Lightweight echo so it's also in diag.jsonl
        D.norm.meta_columns(used_cols=self._meta_cols, img_mean_cols=img_mean_cols, meta_only_cols=meta_only)

        # ============ Image normalization & transform ============
        img_norm_fn = None
        if self.cfg.color_space == "rgb":
            if self.cfg.pretrained and self.cfg.backbone != "smallcnn":
                stats = resolve_preprocess(self.cfg.backbone)
                img_norm_fn = make_rgb_normalizer(stats["mean"], stats["std"])
            else:
                img_norm_fn = None  # keep raw [0,1] tensors
        else:
            img_norm_fn = None    # LAB: typically keep raw in your pipeline

        # DIAG: report image transform choice
        rgb_mean = stats["mean"] if (self.cfg.color_space == "rgb" and self.cfg.pretrained and self.cfg.backbone != "smallcnn") else None
        rgb_std  = stats["std"]  if (self.cfg.color_space == "rgb" and self.cfg.pretrained and self.cfg.backbone != "smallcnn") else None
        D.images.transform_report(color_space=self.cfg.color_space,
                                pretrained=self.cfg.pretrained,
                                backbone=self.cfg.backbone,
                                rgb_mean=rgb_mean, rgb_std=rgb_std)

        self.ds.transform = ToTensorTransform(img_norm_fn=img_norm_fn)

        # Optional small sample stats (cheap; skip if dataset is huge)
        if D.state.enabled:
            D.images.sample_stats(self.train_ds, n=min(8, len(self.train_ds)))

        return self


    # ---- Loaders ----
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.workers, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.workers, pin_memory=True
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.workers, pin_memory=True
        )
