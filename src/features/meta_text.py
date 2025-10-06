# src/features/meta_text.py
from __future__ import annotations
from typing import List, Iterable, Optional, Sequence
import os
import re
import hashlib
import math

import numpy as np
import pandas as pd
import torch


# =============================================================
# Meta → text rendering
# =============================================================

def _pick_make(meta_row: pd.Series) -> Optional[str]:
    """Collapse one-hot columns like Make_Apple, Make_Samsung → 'Apple'/'Samsung'."""
    makes = [c for c in meta_row.index if c.startswith("Make_") and float(meta_row[c]) > 0.5]
    if not makes:
        return None
    makes.sort(key=len)  # prefer shortest, stable
    return makes[0].split("Make_")[-1]


def _safe_get(meta_row: pd.Series, key: str) -> Optional[float]:
    try:
        v = meta_row[key]
    except Exception:
        return None
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def render_meta_text_row(meta_row: pd.Series, template: str = "compact") -> str:
    """
    Turn a single metadata row into a compact, deterministic text string.
    Supported templates: 'compact' (default), 'kv', 'csv'.
    """
    iso_log = _safe_get(meta_row, "iso_log")
    sh_log = _safe_get(meta_row, "shutter_s_log")
    ev = _safe_get(meta_row, "brightness_ev")      # use normalized brightness EV as exposure info
    wb_auto  = meta_row.get("white_balance_auto", None)
    make = _pick_make(meta_row)

    parts: List[str] = []
    if template == "csv":
        if iso_log is not None: parts.append(f"{iso_log:.3f}")
        if sh_log  is not None: parts.append(f"{sh_log:.3f}")
        if ev  is not None: parts.append(f"{ev:.3f}")
        if wb_auto is not None: parts.append("auto" if wb_auto > 0.5 else "manual")
        if make: parts.append(make)
        return ",".join(parts)

    if template == "kv":
        if iso_log is not None: parts.append(f"ISO log = {iso_log:.3f}")
        if sh_log  is not None: parts.append(f"Shutter Speed log = {sh_log:.3f}")
        if ev  is not None: parts.append(f"Exposure value = {ev:.3f}")
        if wb_auto is not None: parts.append(f"White Balance = {'auto' if wb_auto>0.5 else 'manual'}")
        if make: parts.append(f"Make = {make}")
        return " ".join(parts)

    # default: compact
    if iso_log is not None: parts.append(f"iso log {iso_log:.2f}")
    if sh_log  is not None: parts.append(f"shutter log {sh_log:.2f}")
    if ev  is not None: parts.append(f"ev {ev:.2f}")
    if wb_auto is not None: parts.append("wb auto" if wb_auto>0.5 else "wb manual")
    if make: parts.append(f"make {make}")
    return "; ".join(parts)


def render_meta_texts(df: pd.DataFrame, template: str = "compact") -> List[str]:
    """Vectorized wrapper: DataFrame → list[str]."""
    return [render_meta_text_row(df.iloc[i], template=template) for i in range(len(df))]


# =============================================================
# Cache key construction (collision-proof)
# =============================================================

def _slug(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(x)).strip("-")


def _hash_ids(sample_ids: Iterable[str]) -> str:
    """Stable fingerprint for an exact set/order of sample identifiers."""
    h = hashlib.sha1()
    for s in sample_ids:
        h.update((s if isinstance(s, str) else str(s)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:10]


def build_meta_cache_path(
    cache_dir: str,
    *,
    model_name: str,
    layers: str,
    template: str,
    color_space: str,
    features: str,
    included_folders: Optional[str],
    seed: int,
    sample_ids: Iterable[str],
) -> str:
    """
    Compose a cache file path that uniquely identifies:
      - text formation params (template)
      - embedding config (model_name, layers)
      - pipeline config (color_space, features incl. means/no-means)
      - data slice (included_folders, seed, and the exact sample id list/order)
    """
    use_means = 1 if ("mean" in (features or "")) else 0
    ids_hash  = _hash_ids(sample_ids)
    tag = "__".join([
        _slug(model_name.replace("/", "_")),
        f"L={_slug(layers)}",
        f"T={_slug(template)}",
        f"CS={_slug(color_space)}",
        f"MEANS={use_means}",
        f"INC={_slug(included_folders or 'all')}",
        f"S={seed}",
        f"H={ids_hash}",
    ])
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{tag}.npy")


# =============================================================
# HuggingFace embedding backend
# =============================================================

@torch.inference_mode()
def _hf_embed_batch(
    tokenizer,
    model,
    texts: Sequence[str],
    layers: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    """
    Embed a single batch of texts. For each requested layer, we mean-pool over tokens,
    then average across layers. Returns float32 numpy array [B, hidden].
    """
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    outputs = model(**encoded, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple: [layer0, ..., layerN]
    # Convert negative indices as in "-1,-2" into absolute
    L = len(hidden_states)
    layer_idxs = [(l if l >= 0 else L + l) for l in layers]
    pooled_per_layer: List[torch.Tensor] = []

    # attention mask for mean pooling
    mask = encoded["attention_mask"].unsqueeze(-1)  # [B, T, 1]

    for li in layer_idxs:
        H = hidden_states[li]  # [B, T, C]
        H = H * mask  # zero-out pads
        summed = H.sum(dim=1)  # [B, C]
        lens  = mask.sum(dim=1).clamp_min(1)  # [B, 1]
        pooled = summed / lens  # [B, C]
        pooled_per_layer.append(pooled)

    if len(pooled_per_layer) == 1:
        out = pooled_per_layer[0]
    else:
        out = torch.stack(pooled_per_layer, dim=0).mean(dim=0)

    return out.detach().float().cpu().numpy()


def _parse_layers(layers: str) -> List[int]:
    """Parse a string like '-1,-2,-3,-4' → list[int]."""
    return [int(x.strip()) for x in layers.split(",") if x.strip()]


def embed_texts(
    texts: List[str],
    *,
    model_name: str = "intfloat/e5-small-v2",
    layers: str = "-1",
    batch_size: int = 64,
    device: Optional[str] = None,
    cache_key_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute text embeddings with an HF encoder. If cache_key_path exists, it is used.
    Otherwise, results are saved there after computation.

    Returns: np.ndarray [N, D] (float32)
    """
    cache_path = cache_key_path
    if cache_path and os.path.exists(cache_path):
        try:
            arr = np.load(cache_path)
            if arr.ndim == 2:
                return arr.astype(np.float32, copy=False)
        except Exception:
            pass  # fall through and recompute

    # Device
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lazy load tokenizer+model (no global state). ModernBERT tip:
    # disable compiled embeddings (Triton) for older GPUs.
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=cfg).to(dev)
    model.eval()

    layer_idxs = _parse_layers(layers)

    # Iterate batches
    all_embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = _hf_embed_batch(tokenizer, model, batch, layer_idxs, dev)
        all_embs.append(embs)

    embs_np = np.concatenate(all_embs, axis=0).astype(np.float32, copy=False)

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embs_np)
        except Exception:
            pass

    return embs_np


# =============================================================
# Convenience: end-to-end from a DataFrame + IDs
# =============================================================

def meta_embed_from_dataframe(
    df: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    template: str,
    model_name: str,
    layers: str,
    batch_size: int,
    device: Optional[str],
    cache_dir: Optional[str],
    color_space: str,
    features: str,
    included_folders: Optional[str],
    seed: int,
) -> np.ndarray:
    """
    Render meta texts from `df` and embed them, with a collision-proof cache on disk.
    `sample_ids` must be aligned with `df` rows (same order).

    Returns: np.ndarray [N, D]
    """
    texts = render_meta_texts(df, template=template)

    cache_key_path = None
    if cache_dir:
        cache_key_path = build_meta_cache_path(
            cache_dir,
            model_name=model_name,
            layers=layers,
            template=template,
            color_space=color_space,
            features=features,
            included_folders=included_folders,
            seed=seed,
            sample_ids=sample_ids,
        )

    return embed_texts(
        texts,
        model_name=model_name,
        layers=layers,
        batch_size=batch_size,
        device=device,
        cache_key_path=cache_key_path,
    )
