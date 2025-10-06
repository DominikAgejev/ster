# src/data/splitters.py
import shutil
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, Subset
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import os

from .folder_filters import normalize_folder_filters
# --------------------------
# Basic helpers 
# --------------------------



def _class_from_basename(bn: str) -> int:
    # '001' -> 1, '010' -> 10, '0' -> 0
    s = str(bn).lstrip("0") or "0"
    return int(s)

# ---------------------------
# Dataset cleaner
# ---------------------------

def _norm_relpath(p: str) -> str:
    """Normalize to POSIX-ish, strip leading './'."""
    if p is None:
        return ""
    return os.path.normpath(str(p)).replace("\\", "/").lstrip("./")

def _folder_matches(folder: str, pattern: str) -> bool:
    """
    Match rules:
      - If 'pattern' has a slash: match exact path or any subpath that starts with it.
          e.g., pattern='pixel/pixel-1' matches 'pixel/pixel-1' and 'pixel/pixel-1/foo'
      - If 'pattern' has no slash: match any PATH SEGMENT equal to it.
          e.g., pattern='pixel' matches 'pixel', 'pixel/pixel-1', 'other/pixel/foo'
                pattern='pixel-1' matches 'pixel/pixel-1', 'x/pixel-1/y'
    """
    f = _norm_relpath(folder)
    pat = _norm_relpath(pattern)
    if not pat:
        return False
    if "/" in pat:
        return f == pat or f.startswith(pat + "/")
    # segment match anywhere
    return pat in f.split("/")

def clean_dataset(
    ds: Dataset,
    hidden_class_cnt: int = 0,
    excluded_folders: list = None,
    included_folders: list = None,
    seed: int = 42
) -> List[int]:
    remaining_idx = list(range(len(ds)))

    if included_folders:
        pats = normalize_folder_filters(included_folders)
        before = len(remaining_idx)
        remaining_idx = [
            i for i in remaining_idx
            if any(_folder_matches(ds.samples[i].folder, pat) for pat in pats)
        ]
        kept = len(remaining_idx)
        print(f"[split] Included {kept}/{before} samples by folders {pats} (subdirs included)")

    # Exclude folders (with subdirectories support)
    if excluded_folders:
        pats = normalize_folder_filters(excluded_folders)
        before = len(remaining_idx)
        remaining_idx = [
            i for i in remaining_idx
            if not any(_folder_matches(ds.samples[i].folder, pat) for pat in pats)
        ]
        removed = before - len(remaining_idx)
        print(f"[split] Excluded {removed} samples by folders {pats} (subdirs included)")

    # Hide random classes
    if hidden_class_cnt > 0:
        indices_by_class = defaultdict(list)
        for i in remaining_idx:
            cls = _class_from_basename(ds.samples[i].basename)
            indices_by_class[cls].append(i)

        if hidden_class_cnt > len(indices_by_class):
            raise ValueError(f"Cannot hide {hidden_class_cnt} classes, only {len(indices_by_class)} available.")

        rng = np.random.default_rng(seed)
        hidden_classes = rng.choice(list(indices_by_class.keys()), size=hidden_class_cnt, replace=False)
        print(f"[split] Hidden classes [{hidden_class_cnt}]: {hidden_classes}")

        remaining_idx = [i for i in remaining_idx if _class_from_basename(ds.samples[i].basename) not in hidden_classes]

    return sorted(remaining_idx)

# ---------------------------
# Splitters
# ---------------------------

def split_indices(indices, val_split: float, seed: int):
    rng = np.random.RandomState(seed)
    indices = np.array(indices)
    rng.shuffle(indices)

    n_val = int(round(val_split * len(indices)))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return train_idx, val_idx


def get_test_indices(
    ds: Dataset,
    n_samples_per_class: int,
    seed: int = 42,
    hidden_class_cnt: int = 0,
    excluded_folders: list = None,
    included_folders: list = None,
) -> list:
    """
    Return a list of dataset samples to use as a test set.

    Filters the dataset by included/excluded folders and optionally hides classes,
    then selects `n_samples_per_class` per remaining class.

    Returns:
        Sorted list of dataset indices (ints).
    """
    # Step 1: Filter dataset like clean_dataset
    remaining_idx = clean_dataset(
        ds,
        hidden_class_cnt=hidden_class_cnt,
        excluded_folders=excluded_folders,
        included_folders=included_folders,
        seed=seed,
    )

    if not remaining_idx:
        raise ValueError("After filtering, no samples remain to draw a test set from. "
                         "Check included/excluded folders and hidden classes.")

    # Step 2: Group by class
    rng = np.random.RandomState(seed)
    class_to_indices = defaultdict(list)
    for i in remaining_idx:
        cls = _class_from_basename(ds.samples[i].basename)
        class_to_indices[cls].append(i)

    # Step 3: Sample per class
    selected_indices = []
    for cls, indices in class_to_indices.items():
        if len(indices) < n_samples_per_class:
            raise ValueError(
                f"Class {cls} only has {len(indices)} samples after filtering, "
                f"but need {n_samples_per_class} for test set"
            )
        chosen = rng.choice(indices, size=n_samples_per_class, replace=False)
        selected_indices.extend(chosen.tolist())

    selected_indices = sorted(set(selected_indices))

    # Step 4: Return actual Sample objects
    #selected_samples = [ds.samples[i] for i in selected_indices]
    return selected_indices


def train_val_split(
    ds: Dataset,
    val_split: float = 0.2,
    seed: int = 42,
    test_indices: List[int] | None = None,
    hidden_classes_cnt: int = 0,
    excluded_folders: list = None,
    included_folders: list = None,
    group_split: Optional[str] = None,
):
    rng = np.random.RandomState(seed)

    remaining_idx = clean_dataset(ds, hidden_class_cnt=hidden_classes_cnt, excluded_folders=excluded_folders, included_folders=included_folders, seed=seed)
    test_set = set(test_indices) if test_indices is not None else set()
    remaining_idx = [i for i in remaining_idx if i not in test_set]


    if group_split is None:
        train_idx, val_idx = split_indices(remaining_idx, val_split, seed)
    else:
        group_to_indices = defaultdict(list)
        for i in remaining_idx:
            if group_split == "class":
                key = _class_from_basename(ds.samples[i].basename)
            elif group_split == "folder":
                key = ds.samples[i].folder
            else:
                raise ValueError(f"Invalid group_split={group_split}, must be 'class', 'folder', or None")
            group_to_indices[key].append(i)

        groups = list(group_to_indices.keys())
        n_val_groups = max(1, int(round(val_split * len(groups))))
        val_groups = set(rng.choice(groups, size=n_val_groups, replace=False))
        train_groups = set(groups) - val_groups
        train_idx = [i for g in train_groups for i in group_to_indices[g]]
        val_idx = [i for g in val_groups for i in group_to_indices[g]]

    return sorted(train_idx), sorted(val_idx)

# --------------------------
# NEW: persistence & sanity
# --------------------------
def make_test_album(ds: Subset, path='/home/data/Pantone/test-dominik/', csv_name='test.csv'):
    if path is not None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    samples = [ds.dataset.samples[i] for i in ds.indices]
    labels_lab = ds.dataset.labels_lab[ds.indices]
    labels_rgb = ds.dataset.labels_rgb[ds.indices]
    metadatas = ds.dataset.metadatas.iloc[ds.indices].reset_index(drop=True)
    rows = []
    for i, s in enumerate(samples):
        row = {
            "basename": s.basename,
            "folder": s.folder,
            "image_path": s.image_path,
            "json_path": s.json_path,
        }

        # add labels
        row.update({
            "L": labels_lab[i][0],
            "a": labels_lab[i][1],
            "b": labels_lab[i][2],
            "R": labels_rgb[i][0],
            "G": labels_rgb[i][1],
            "B": labels_rgb[i][2],
        })

        rows.append(row)

        subfolder = os.path.join(path, s.folder)
        os.makedirs(subfolder, exist_ok=True)
        dst_file = os.path.join(subfolder, os.path.basename(s.image_path))
        shutil.copyfile(s.image_path, dst_file)

    df_labels = pd.DataFrame(rows)
    df = pd.concat([df_labels.reset_index(drop=True), metadatas.reset_index(drop=True)], axis=1)

    csv_path = os.path.join(path, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"[info] Saved copy of {len(df)} test samples to {csv_path}")

def save_splits(path: str,
                *,
                train_idx: List[int],
                val_idx: List[int],
                test_idx: List[int] | None = None,
                meta: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "train": sorted(list(map(int, set(train_idx)))),
        "val":   sorted(list(map(int, set(val_idx)))),
        "test":  sorted(list(map(int, set(test_idx)))) if test_idx is not None else None,
        "meta":  meta or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[split] saved -> {path}")


def load_splits(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    # normalize
    out = {
        "train": data.get("train", []),
        "val":   data.get("val", []),
        "test":  data.get("test", None),
        "meta":  data.get("meta", {}),
    }
    print(f"[split] loaded <- {path}")
    return out
