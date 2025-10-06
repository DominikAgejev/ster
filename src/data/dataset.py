# src/data/dataset.py
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

# Color conversions
from skimage.color import rgb2lab, lab2rgb

from .metadata import process_metadata_json_file, make_missing_metadata_df


def _list_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[str]:
    out = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(base, f))
    out.sort()
    return out


def _basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _flatten_exif(d: dict, prefix="") -> dict:
    """Flatten nested dicts/lists into a 1-level dict with dot/idx notation."""
    flat = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            flat.update(_flatten_exif(v, kk))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            kk = f"{prefix}[{i}]"
            flat.update(_flatten_exif(v, kk))
    else:
        flat[prefix] = d
    return flat


def _lab_triplet_from_df(row: pd.Series) -> Tuple[float, float, float]:
    """
    Try to extract (L, a, b) from a row with flexible column names.
    Expected primary names: 'L', 'a', 'b'
    Fallbacks: 'lab_L','lab_a','lab_b' or 'L*','a*','b*'
    """
    candidates = [
        ("L", "a", "b"),
        ("lab_L", "lab_a", "lab_b"),
        ("L*", "a*", "b*"),
    ]
    for Lc, ac, bc in candidates:
        if Lc in row and ac in row and bc in row:
            return float(row[Lc]), float(row[ac]), float(row[bc])
    raise KeyError("Could not find Lab columns in labels CSV (tried L/a/b, lab_L/lab_a/lab_b, L*/a*/b*).")


@dataclass
class Sample:
    image_path: str
    json_path: Optional[str]
    basename: str
    folder: str


class ImageMetadataDataset(Dataset):
    """
    Loads (image, metadata features, label) triplets.

    - color_space: 'lab' or 'rgb'
      * 'lab': image is converted to Lab in __getitem__, labels are Lab (3 floats).
               We also compute per-image Lab means (L_mean_img, a_mean_img, b_mean_img).
      * 'rgb': image is returned in RGB float [0,1], labels are converted to RGB.
               We also compute per-image RGB means (R_mean_img, G_mean_img, B_mean_img).

    - metadatas: pandas DataFrame aligned to samples (row i corresponds to sample i).
      DataModule will choose which columns to pass to the model via `meta_numeric_cols`.

    Public attributes the rest of the pipeline expects:
      - self.metadatas: pd.DataFrame
      - self.meta_numeric_cols: List[str] (set by DataModule)
      - self.meta_dim: int (derived from meta_numeric_cols length)
    """

    def __init__(
        self,
        images_dir: str,
        json_dir: str,
        labels_path: str,
        color_space: str = "lab",
        transform=None
        ):
        super().__init__()
        assert color_space in ("lab", "rgb")
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.labels_path = labels_path
        self.color_space = color_space
        self.transform = transform

        # 1) enumerate images
        self.image_paths: List[str] = _list_images(images_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under: {images_dir}")

        # 2) build per-sample descriptors
        self.samples: List[Sample] = []
        for p in self.image_paths:
            bn = _basename_noext(p)
            folder = os.path.relpath(os.path.dirname(p), images_dir)
            jp = os.path.join(json_dir, folder, f"{bn}.json")
            self.samples.append(Sample(p, jp if os.path.exists(jp) else None, bn, folder))

        # 3) read labels CSV and align to sample order (INDEX-BASED)
        labels_df = pd.read_csv(labels_path)
        if "Index" not in labels_df.columns:
            raise KeyError("Labels CSV must include an 'Index' column when using Index-based mapping.")

        # Coerce Index to int (handles '1.0' etc.) and ensure uniqueness
        labels_df["Index_int"] = labels_df["Index"].apply(lambda x: int(float(x)))
        dups = labels_df["Index_int"].duplicated(keep=False)
        if dups.any():
            bad = labels_df.loc[dups, "Index_int"].tolist()
            raise ValueError(f"Duplicate Index values in labels CSV (first few): {bad[:10]}")

        labels_df = labels_df.set_index("Index_int", drop=False)

        def _basename_to_index(bn: str) -> int:
            # '001' -> 1, '010' -> 10, '0' -> 0 (if ever present)
            s = os.path.splitext(os.path.basename(bn))[0]
            s = s.lstrip("0") or "0"
            return int(s)

        labs = []
        rgbs = []
        missing = []

        for s in self.samples:
            idx_int = _basename_to_index(s.basename)
            if idx_int not in labels_df.index:
                missing.append((s.basename, idx_int))
                continue

            row = labels_df.loc[idx_int]

            # ----- Lab label (prefer explicit L*,a*,b* columns; otherwise fallback helper) -----
            if all(c in row for c in ["L*", "a*", "b*"]):
                L, a, b = float(row["L*"]), float(row["a*"]), float(row["b*"])
            else:
                L, a, b = _lab_triplet_from_df(row)
            labs.append((L, a, b))

            # ----- RGB label -----
            # Prefer CSV-provided R_web/G_web/B_web (assumed 0..255); else convert from Lab.
            if all(c in row for c in ["R_web", "G_web", "B_web"]):
                r = float(row["R_web"]) / 255.0
                g = float(row["G_web"]) / 255.0
                b_ = float(row["B_web"]) / 255.0
                rgbs.append((r, g, b_))
            else:
                rgb = lab2rgb(np.array([[[L, a, b]]], dtype=np.float32)).reshape(3)
                rgbs.append((float(rgb[0]), float(rgb[1]), float(rgb[2])))

        if missing:
            raise KeyError(
                "Missing ground-truth for these images (basename -> parsed Index): "
                f"{missing[:10]} (showing up to 10). "
                "Ensure filenames like '001.jpg' map to CSV Index=1."
            )

        self.labels_lab = np.asarray(labs, dtype=np.float32)  # [N,3]
        self.labels_rgb = np.asarray(rgbs, dtype=np.float32)  # [N,3] in [0,1]


        # 4) metadata table (from metadata.py whitelist: ISO, Shutter, Brightness, Make_*)
        rows = []
        for s in self.samples:
            if s.json_path is None:
                df = make_missing_metadata_df()
            else:
                df = process_metadata_json_file(s.json_path)
                if df is None or df.empty:
                    df = make_missing_metadata_df()
            # each df is 1-row; append that row (union of columns handled by DataFrame below)
            rows.append(df.iloc[0])

        meta_df = pd.DataFrame(rows)
        meta_df["basename"] = [s.basename for s in self.samples]
        meta_df = meta_df.replace([np.inf, -np.inf], np.nan)
        # Only one-hots (Make_*) are safe to zero-fill here; leave continuous as NaN.
        onehots = [c for c in meta_df.columns if c.startswith("Make_")]
        meta_df[onehots] = meta_df[onehots].fillna(0.0)
        # Keep a copy for debugging; this is already numeric + one-hots from metadata.py
        self.meta_raw = meta_df.copy()
        meta_num = meta_df


        # 5) compute per-image mean in active color space and attach as metadata columns
        #    (these are used when the preset asks for image means)
        if color_space == "lab":
            means = self._compute_image_means_lab()
            meta_num[["L_mean_img", "a_mean_img", "b_mean_img"]] = means
        else:
            means = self._compute_image_means_rgb()
            meta_num[["R_mean_img", "G_mean_img", "B_mean_img"]] = means

        # 6) finalize table
        self.metadatas = meta_num.reset_index(drop=True)
        self.meta_numeric_cols: List[str] = []  # DataModule will set this
        self.meta_dim: int = 0

    # ----------------- helper computations -----------------
    def _open_pil_rgb(self, path: str) -> np.ndarray:
        # returns HxWx3 float32 in [0,1]
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB")) / 255.0

    @staticmethod
    def _center_crop(arr: np.ndarray, size: int = 128) -> np.ndarray:
        """
        Return the centered size×size crop if both dimensions >= size.
        Otherwise, return the original array unchanged.
        """
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if h >= size and w >= size:
            y0 = (h - size) // 2
            x0 = (w - size) // 2
            return arr[y0:y0+size, x0:x0+size, ...]
        return arr

    def _compute_image_means_rgb(self) -> np.ndarray:
        out = np.zeros((len(self.samples), 3), dtype=np.float32)
        for i, s in enumerate(self.samples):
            rgb = self._open_pil_rgb(s.image_path)   # [H,W,3] in [0,1]
            rgb = self._center_crop(rgb, 128)        # always center 128×128 when possible
            out[i] = rgb.reshape(-1, 3).mean(axis=0)
        return out

    def _compute_image_means_lab(self) -> np.ndarray:
        out = np.zeros((len(self.samples), 3), dtype=np.float32)
        for i, s in enumerate(self.samples):
            rgb = self._open_pil_rgb(s.image_path)   # [H,W,3] in [0,1]
            rgb = self._center_crop(rgb, 128)        # always center 128×128 when possible
            lab = rgb2lab(rgb).astype(np.float32)    # [H,W,3]
            out[i] = lab.reshape(-1, 3).mean(axis=0)
        return out

    # ----------------- dataset API -----------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        rgb = self._open_pil_rgb(s.image_path)  # [H,W,3] in [0,1]

        if self.color_space == "lab":
            image_arr = rgb2lab(rgb).astype(np.float32)        # [H,W,3], not normalized
            label = self.labels_lab[idx].astype(np.float32)    # [3]
        else:
            image_arr = rgb.astype(np.float32)                 # [H,W,3] in [0,1]
            label = self.labels_rgb[idx].astype(np.float32)    # [3] in [0,1]

        # metadata vector is selected upstream by DataModule (columns set to meta_numeric_cols)
        if self.meta_numeric_cols:
            meta_vec = self.metadatas.loc[idx, self.meta_numeric_cols].to_numpy(dtype=np.float32)
        else:
            meta_vec = np.zeros((0,), dtype=np.float32)

        sample = {
            "image": image_arr,             # HxWx3 float32
            "metadata": meta_vec,           # K float32
            "label": label,                 # 3 float32
            "basename": s.basename,
            "folder": s.folder,
        }
        # Optional: attach the rendered meta text if present (for debug/logging)
        if hasattr(self, "meta_texts"):
            try:
                sample["meta_text"] = self.meta_texts[idx]
            except Exception:
                pass

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

def to_batch(batch, device=None):
    """
    Simplest path: batch is the dict collated by the default DataLoader.
    Expects:
      batch["image"]   -> [B,3,H,W] float-ish
      batch["label"]   -> [B,3]
      batch["metadata"]-> [B,K] or missing
    """
    images = batch["image"].float()
    labels = batch["label"].float()
    meta   = batch.get("metadata", None)

    if meta is None:
        metadata = torch.zeros((images.size(0), 0), dtype=torch.float32)
    else:
        metadata = meta.float()
        if metadata.ndim == 1:  # rare, but keep it robust
            metadata = metadata.unsqueeze(1)

    if device is not None:
        images   = images.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

    out = {"image": images, "metadata": metadata, "label": labels}
    if "meta_text" in batch:  # a list of strings
        out["meta_text"] = batch["meta_text"]
    return out

