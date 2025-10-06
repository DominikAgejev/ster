# src/analysis/delta_distribution.py
from __future__ import annotations
import argparse, os, sys, json, math, base64, io
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---- Project imports (keep these aligned to your tree) ----
from ..engine.datamodule import DataModule, DataModuleConfig
from ..models import build_model
from ..metrics.losses import ColorLoss, deltaE2000_loss
from ..metrics.evaluate import _apply_eval_activation

try:
    from ..analysis.summarize_sweep import sha12 as _sha12
except Exception:
    _sha12 = None

# --- tolerant coercers --------------------------------------------------------
def _intish(v, default):
    if v is None: return default
    if isinstance(v, bool): return int(v)
    if isinstance(v, (int, float)): return int(v)
    if isinstance(v, (list, tuple)):  # take the first entry if a list/tuple
        return _intish(v[0] if v else None, default)
    if isinstance(v, str):
        s = v.strip()
        if not s: return default
        try: return int(s)
        except ValueError:
            try: return int(float(s))
            except Exception: return default
    return default

def _floatish(v, default):
    if v is None: return default
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, (list, tuple)):
        return _floatish(v[0] if v else None, default)
    if isinstance(v, str):
        s = v.strip()
        if not s: return default
        try: return float(s)
        except Exception: return default
    return default

def _token_stage_from_cfg(cfg, default=-2):
    """Parse token_stage from a checkpoint config. Keep lists as lists."""
    ts = cfg.get("token_stage", default)
    if ts is None:
        return default
    if isinstance(ts, (list, tuple)):
        return [int(v) for v in ts]
    s = str(ts).strip()
    if not s:
        return default
    low = s.lower()
    if low in {"ms", "multi", "auto", "none"}:
        # Unknown exact stages; fall back to default so build_model decides (or edit to infer from ckpt)
        return default
    if s.startswith("[") and s.endswith("]"):
        try:
            import json as _json
            v = _json.loads(s)
            if isinstance(v, list):
                return [int(x) for x in v]
        except Exception:
            pass
    import re
    nums = [int(x) for x in re.findall(r"-?\d+", s)]
    if not nums:
        return default
    return nums if len(nums) > 1 else nums[0]

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _bool(v, default=False) -> bool:
    if v is None: return default
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    return s in {"1","true","yes","y"}

def _device_from_arg(arg: Optional[str]) -> str:
    if arg in ("cpu","cuda"): return arg
    return "cuda" if torch.cuda.is_available() else "cpu"

def _run_id_from_cfg(cfg: Dict) -> str:
    if _sha12 is not None:
        try:
            keys = [
                "model","backbone","features","color_space","pred_activation","activation_eps",
                "val_split","group_split","seed","hidden_classes_cnt","included_folders","excluded_folders",
                "meta_encoder","meta_model_name","meta_layers","meta_text_template",
            ]
            sub = {k: cfg.get(k) for k in keys if k in cfg}
            return _sha12(sub)
        except Exception:
            pass
    s = json.dumps(cfg, sort_keys=True, separators=(",",":"))
    return f"h{abs(hash(s))%10**10:010d}"

def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def _percentiles(arr: np.ndarray, ps=(5,25,50,75,90,95,99)) -> Dict[str, float]:
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}

def _spread_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    mean = float(np.mean(arr)) if n else float("nan")
    std  = float(np.std(arr, ddof=0)) if n else float("nan")
    med  = float(np.median(arr)) if n else float("nan")
    mad  = float(np.median(np.abs(arr - med))) if n else float("nan")
    q1, q3 = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75))) if n else (float("nan"), float("nan"))
    iqr = (q3 - q1) if n else float("nan")
    cv  = (std / mean) if (n and mean != 0.0) else float("nan")
    p = _percentiles(arr)
    # useful spread deltas
    p90_p10 = (p["p90"] - float(np.percentile(arr,10))) if n else float("nan")
    p95_p5  = (p["p95"] - p["p5"]) if n else float("nan")
    return {
        "count": int(n), "mean": mean, "std": std, "median": med, "mad": mad,
        "q1": q1, "q3": q3, "iqr": iqr, "cv": cv, **p, "p90_minus_p10": float(p90_p10), "p95_minus_p5": float(p95_p5)
    }

def _html_run_header(cfg: Dict[str, Any], ckpt_path: str) -> str:
    """
    Small HTML header with key run params and the exact checkpoint path.
    """
    def g(k, d="—"):
        v = cfg.get(k, d)
        return str(v if v is not None else d)
    rows = [
        ("Model",            g("model")),
        ("Backbone",         g("backbone")),
        ("Features",         g("features")),
        ("Included folders", g("included_folders")),
        ("Color space",      g("color_space")),
        ("Epochs",           g("epochs")),
        ("Batch size",       g("batch_size")),
        ("Seed",             g("seed")),
        ("Token stage",      g("token_stage")),
        ("Pretrained",       g("pretrained")),
        ("Optimizer",        g("optim")),
        ("LR",               g("lr")),
        ("LR schedule",      g("lr_schedule")),
        ("Weight decay",     g("weight_decay")),
    ]
    head = []
    head.append('<section id="run-header" style="font-family:system-ui,sans-serif;margin:16px 0;">')
    head.append('<div style="font-size:18px;font-weight:600;margin-bottom:6px;">Run Summary</div>')
    head.append(f'<div style="font-size:12px;color:#555;margin-bottom:10px;">Checkpoint: '
                f'<code style="font-size:12px;">{ckpt_path}</code></div>')
    head.append('<table style="border-collapse:collapse;font-size:13px;">')
    for k, v in rows:
        head.append('<tr>'
                    f'<td style="border:1px solid #ddd;padding:4px 8px;background:#fafafa;">{k}</td>'
                    f'<td style="border:1px solid #ddd;padding:4px 8px;">{v}</td>'
                    '</tr>')
    head.append('</table></section><hr style="margin:14px 0;border:none;border-top:1px solid #eee;">')
    return "".join(head)


# ---------------------------------------
# Auto-zoom + binning for histograms
# ---------------------------------------
def hist_auto_zoom_bins(arr: np.ndarray, min_bins=20, max_bins=80) -> Tuple[np.ndarray, Tuple[float,float]]:
    """
    Choose a zoomed range and bin count:
      - zoom: [Q1 - 1.5*IQR, Q3 + 1.5*IQR] clipped to [min,max]
      - bins: Freedman–Diaconis width on zoomed range
    Returns (bins, (xmin, xmax))
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.arange(1), (0.0, 1.0)
    q1, q3 = np.percentile(a, [25, 75])
    iqr = max(q3 - q1, 1e-9)
    lo  = max(a.min(), q1 - 1.5 * iqr)
    hi  = min(a.max(), q3 + 1.5 * iqr)
    zoom = a[(a >= lo) & (a <= hi)]
    if zoom.size == 0:
        zoom = a
        lo, hi = float(a.min()), float(a.max())
    # Freedman–Diaconis bin width
    iqr_zoom = max(np.percentile(zoom, 75) - np.percentile(zoom, 25), 1e-9)
    h = 2.0 * iqr_zoom * (zoom.size ** (-1/3))
    nb = int(math.ceil((hi - lo) / h)) if h > 0 else min_bins
    nb = max(min_bins, min(max_bins, nb))
    # Build explicit bin edges so both VAL/TEST line up nicely
    bins = np.linspace(lo, hi, nb + 1)
    return bins, (float(lo), float(hi))

def _save_histogram(values: np.ndarray, out_png: str, title: str, bins: Any = 50, range_: Tuple[float,float] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    if range_ is not None:
        plt.hist(values, bins=bins, range=range_)
    else:
        plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("ΔE00"); plt.ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ---------------------------------------
# Image + JSON helpers
# ---------------------------------------
_EXTS = ("jpg","jpeg","png","webp","bmp")

def _find_image(root: str, folder: str, basename: str) -> Optional[str]:
    """
    Try a few common patterns to locate an image file by (folder, basename) under root.
    """
    cands = []
    # root / folder / basename.ext
    for ext in _EXTS:
        cands.append(os.path.join(root, folder, f"{basename}.{ext}"))
    # root / basename.ext
    for ext in _EXTS:
        cands.append(os.path.join(root, f"{basename}.{ext}"))
    # root / folder / images / basename.ext
    for ext in _EXTS:
        cands.append(os.path.join(root, folder, "images", f"{basename}.{ext}"))
    # root / images / folder / basename.ext
    for ext in _EXTS:
        cands.append(os.path.join(root, "images", folder, f"{basename}.{ext}"))
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

def _find_json(json_root: str, folder: str, basename: str) -> Optional[str]:
    for cand in (
        os.path.join(json_root, folder, f"{basename}.json"),
        os.path.join(json_root, f"{basename}.json"),
    ):
        if os.path.isfile(cand):
            return cand
    return None

def _thumbnail(path: str, max_w: int = 256, max_h: int = 256) -> Optional[str]:
    """
    Make a tiny inline thumbnail (base64-encoded PNG). Returns data URL or None.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_w, max_h))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception:
        return None

def _flatten_numeric(d: Dict[str, Any], parent: str = "") -> Dict[str, float]:
    """
    Recursively flatten numeric fields (int/float/bool), return {keypath: value}.
    """
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (dict,)):
            out.update(_flatten_numeric(v, key))
        # ignore lists/strings for now
    return out

# --------------------------------------------------------------------------------------
# Core data collection
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _collect_per_sample_de(model, loader: DataLoader, device: str,
                           color_space: str, pred_activation: str, activation_eps: float):
    rows: List[Dict[str, object]] = []
    model = model.to(device).eval()
    dummy_crit = ColorLoss(input_space=color_space)

    def _to_device_meta(meta):
        if meta is None:
            return None
        if torch.is_tensor(meta):
            return meta.to(device)
        if isinstance(meta, dict):
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in meta.items()}
        if isinstance(meta, (list, tuple)):
            items = [(m.to(device) if torch.is_tensor(m) else m) for m in meta]
            return type(meta)(items) if isinstance(meta, tuple) else items
        return meta

    for batch in loader:
        img  = batch["image"].to(device)
        meta = _to_device_meta(batch.get("metadata"))
        y    = batch["label"].to(device)

        preds = model(img, meta.to(device) if torch.is_tensor(meta) else meta)
        preds = _apply_eval_activation(preds, color_space, pred_activation, activation_eps)

        de_samples = deltaE2000_loss(
            dummy_crit._to_lab(preds), dummy_crit._to_lab(y), reduction="none"
        ).detach().cpu().numpy()

        B = y.size(0)
        basenames = batch.get("basename", ["?"]*B)
        folders   = batch.get("folder",   ["?"]*B)
        for i in range(B):
            rows.append({
                "basename": basenames[i],
                "folder":   folders[i],
                "deltaE00": float(de_samples[i]),
            })
    return rows

def _build_dm(split_kind: str, cfg: Dict, images_dir: str, json_dir: str, labels_csv: str,
              batch_size: int, workers: int, seed: int,
              split_file: Optional[str], mode: str):
    dm = DataModule(DataModuleConfig(
        images_dir=images_dir, json_dir=json_dir, labels_csv=labels_csv,
        batch_size=batch_size,
        val_split = _floatish(cfg.get("val_split"), 0.15) if split_kind == "val" else 0.0,
        workers   = _intish(cfg.get("workers", workers), workers),
        seed      = _intish(cfg.get("seed", seed), seed),
        hidden_classes_cnt = _intish(cfg.get("hidden_classes_cnt"), 0),
        group_split=cfg.get("group_split", None) if split_kind=="val" else None,
        color_space=cfg.get("color_space", "lab"),
        features=cfg.get("features", "image+mean+meta"),
        pretrained=_bool(cfg.get("pretrained", True)),
        backbone=cfg.get("backbone", "smallcnn"),
        include_test=False,
        test_per_class= _intish(cfg.get("test_per_class"), 3),
        excluded_folders=cfg.get("excluded_folders", []) or [],
        included_folders=cfg.get("included_folders", []) or [],
        split_file=split_file,
        save_splits_flag=False,
        meta_encoder=cfg.get("meta_encoder", "none"),
        meta_model_name=cfg.get("meta_model_name", "jhu-clsp/ettin-encoder-17m"),
        meta_layers=cfg.get("meta_layers", "-1,-2,-3,-4"),
        meta_text_template=cfg.get("meta_text_template", "compact"),
        meta_batch_size=int(cfg.get("meta_batch_size", 64)),
    )).setup()
    loader = dm.val_dataloader() if split_kind=="val" else dm.test_dataloader()
    if loader is None:
        raise RuntimeError(f"{split_kind} loader is None (check split files).")
    return dm, loader

def _load_ckpt(ckpt_path: str) -> Tuple[Dict, Dict]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"Bad checkpoint (missing state_dict): {ckpt_path}")
    cfg = obj.get("config", {}) or {}
    return obj, cfg

def _build_model(obj: Dict,
                 dm_meta_dim: int,
                 model_name: str,
                 backbone: str,
                 device: str,
                 token_stage=None,
                 pretrained: bool = True) -> torch.nn.Module:
    """
    Build the model exactly like training, forwarding backbone-only knobs and
    (for xattn) matching the checkpoint's attention dimension to avoid shape mismatches.
    """
    # Infer attn_dim from the checkpoint for CrossAttnNet (xattn) — same trick as eval_ckpt.py
    attn_dim = None
    if str(model_name).lower() in ("xattn", "cross", "crossattn"):
        sd = obj.get("state_dict", {}) or {}
        if "attn.in_proj_weight" in sd:
            # MultiheadAttention in_proj_weight: [3d, d] → d is second dim
            attn_dim = int(sd["attn.in_proj_weight"].shape[1])
        elif "attn.out_proj.weight" in sd:
            # out_proj.weight: [d, d] → first dim is d
            attn_dim = int(sd["attn.out_proj.weight"].shape[0])
        elif "q_vec" in sd:
            # meta-free variant: q_vec: [d]
            attn_dim = int(sd["q_vec"].shape[0])
        if attn_dim is not None:
            print(f"[analysis] inferred attn_dim={attn_dim} from checkpoint.")

    # Build with the same knobs training used; token_stage/pretrained are backbone-only
    build_kwargs = dict(
        backbone=backbone,
        meta_dim=dm_meta_dim,
        token_stage=token_stage,
        pretrained=bool(pretrained),
    )
    if attn_dim is not None and str(model_name).lower() in ("xattn", "cross", "crossattn"):
        build_kwargs["attn_dim"] = attn_dim

    m = build_model(model_name, **build_kwargs)  # models.__init__ forwards backbone-only args to the bundle
    missing, unexpected = m.load_state_dict(obj["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    m.to(device).eval()
    return m

# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _write_folder_stats(df: pd.DataFrame, out_csv: str):
    rows = []
    for folder, dff in df.groupby("folder"):
        arr = dff["deltaE00"].values.astype(float)
        rec = {"folder": folder, **_spread_stats(arr)}
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(by=["mean","count"], ascending=[True, False])
    out.to_csv(out_csv, index=False)
    return out

def _write_histograms(df: pd.DataFrame, outdir: str, split: str, bins: int | None = None):
    _safe_makedirs(outdir)
    values = df["deltaE00"].values.astype(float)
    # Full-range hist (fixed bins or FD-derived)
    if bins is None:
        # basic FD bins on the full range
        _, (lo, hi) = hist_auto_zoom_bins(values)  # get zoom first for a sensible span
        rng = (float(values.min()), float(values.max()))
        nb = max(30, min(100, int((hi - lo) / (2 * (np.percentile(values,75)-np.percentile(values,25)+1e-9) * values.size ** (-1/3))) or 50))
        _save_histogram(values, os.path.join(outdir, f"{split}_hist_overall.png"),
                        f"{split.upper()} ΔE00 (overall)", bins=nb)
    else:
        _save_histogram(values, os.path.join(outdir, f"{split}_hist_overall.png"),
                        f"{split.upper()} ΔE00 (overall)", bins=bins)

    # Zoomed hist via hist_auto_zoom
    zoom_bins, (zlo, zhi) = hist_auto_zoom_bins(values)
    _save_histogram(values, os.path.join(outdir, f"{split}_hist_overall_zoom.png"),
                    f"{split.upper()} ΔE00 (zoomed)", bins=zoom_bins, range_=(zlo, zhi))

def _collect_meta_table(df_samples: pd.DataFrame, json_root: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str,float]]]:
    """
    Build a DataFrame of numeric metadata fields across samples to enable z-score checks.
    Index: (folder, basename)
    """
    recs = []
    for _, r in df_samples.iterrows():
        folder = str(r["folder"]); base = str(r["basename"])
        j = _find_json(json_root, folder, base)
        if not j: 
            recs.append({"folder": folder, "basename": base})  # keep row; will be NaNs
            continue
        try:
            with open(j, "r") as f:
                d = json.load(f)
        except Exception:
            recs.append({"folder": folder, "basename": base})
            continue
        flat = _flatten_numeric(d)
        rec = {"folder": folder, "basename": base, **flat}
        recs.append(rec)
    mdf = pd.DataFrame(recs).set_index(["folder","basename"]).sort_index()
    num_cols = [c for c in mdf.columns if mdf[c].dtype != "O"]
    stats = {}
    if len(num_cols) > 0 and len(mdf) > 0:
        mu = mdf[num_cols].mean(skipna=True)
        sd = mdf[num_cols].std(skipna=True).replace(0.0, np.nan)
        stats = {"mean": mu.to_dict(), "std": sd.to_dict()}
    return mdf, stats

def _unusual_fields_for_sample(meta_df: pd.DataFrame, stats: Dict[str, Dict[str, float]],
                               folder: str, basename: str, zthr: float = 3.0, maxn: int = 6) -> List[str]:
    if meta_df is None or not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        return []
    try:
        row = meta_df.loc[(folder, basename)]
    except KeyError:
        return []
    out = []
    for c in meta_df.columns:
        if c in ("folder","basename"): continue
        v = row.get(c)
        if pd.isna(v): continue
        m = stats["mean"].get(c); s = stats["std"].get(c)
        if m is None or s is None or not np.isfinite(s) or s == 0.0: 
            continue
        z = (float(v) - float(m)) / float(s)
        if abs(z) >= zthr:
            out.append(f"{c}={v:.4g} (z={z:+.2f})")
    # sort by |z| if we encoded it
    out = sorted(out, key=lambda s: abs(float(s.split("z=")[1].rstrip(")"))), reverse=True)
    return out[:maxn]

def _write_html_report(df: pd.DataFrame, outdir: str, split: str,
                       images_dir: str, original_dir: str, json_dir: str,
                       folder_stats_csv: str,
                       header_cfg: Dict[str, Any],
                       ckpt_path: str):
    """
    Create a compact HTML report that links images & highlights outliers + metadata oddities.
    """
    _safe_makedirs(outdir)
    html = []
    html.append("<html><head><meta charset='utf-8'>")
    html.append("""<style>
    body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:24px;}
    h1{margin:0 0 8px 0;} h2{margin-top:28px;}
    table{border-collapse:collapse;width:100%;font-size:14px}
    th,td{border:1px solid #ddd; padding:6px 8px;}
    th{background:#f7f7f7; text-align:left;}
    .grid{display:grid; grid-template-columns: 1fr 1fr; gap:8px;}
    .thumb{display:flex; gap:8px; align-items:flex-start;}
    .thumb img{border:1px solid #ddd; border-radius:8px; max-width:256px; height:auto}
    .meta{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; font-size:12px;}
    .muted{color:#777}
    </style>""")
    html.append(f"<title>ΔE00 report — {split}</title></head><body>")
    html.append(f"<h1>ΔE00 {split.upper()} report</h1>")

    # --- run header with params + checkpoint path ---
    try:
        html.append(_html_run_header(header_cfg or {}, ckpt_path))
    except Exception:
        # never fail report on header issues
        pass

    # Overview stats
    arr = df["deltaE00"].values.astype(float)
    st = _spread_stats(arr)
    st_table = "".join(f"<tr><th>{k}</th><td>{v:.6g}</td></tr>" for k, v in st.items())
    html.append("<h2>Overview</h2>")
    html.append("<table>" + st_table + "</table>")

    # Histograms
    for name in ("hist_overall.png", "hist_overall_zoom.png"):
        p = os.path.join(outdir, f"{split}_{name}")
        if os.path.isfile(p):
            html.append(f"<h2>Histogram: {'zoom' if 'zoom' in name else 'overall'}</h2>")
            html.append(f"<img src='{os.path.basename(p)}' style='max-width: 100%; border:1px solid #ddd; border-radius:8px'/>")

    # Per-folder stats (CSV rendered inline light)
    html.append("<h2>Per-folder stats</h2>")
    if os.path.isfile(folder_stats_csv):
        fdf = pd.read_csv(folder_stats_csv)
        cols = ["folder","count","mean","std","median","mad","iqr","cv","min","p5","p25","p50","p75","p90","p95","p99","max","p90_minus_p10","p95_minus_p5"]
        cols = [c for c in cols if c in fdf.columns]
        html.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        for _, r in fdf.iterrows():
            html.append("<tr>" + "".join(f"<td>{r[c] if not isinstance(r[c], float) else f'{r[c]:.6g}'}</td>" for c in cols) + "</tr>")
        html.append("</table>")
    else:
        html.append("<p class='muted'>No per-folder stats file found.</p>")

    # Outliers gallery
    html.append("<h2>Outliers (largest ΔE00)</h2>")
    topn = min(48, len(df))
    dff = df.sort_values("deltaE00", ascending=False).head(topn).reset_index(drop=True)

    # Gather metadata table for z-score checks
    meta_df, meta_stats = _collect_meta_table(dff, json_dir)

    cards = []
    for _, r in dff.iterrows():
        folder = str(r["folder"]); base = str(r["basename"]); de = float(r["deltaE00"])
        crop = _find_image(images_dir, folder, base)
        orig = _find_image(original_dir, folder, base) if original_dir else None
        crop_thumb = _thumbnail(crop) if crop else None
        orig_thumb = _thumbnail(orig) if orig else None
        unusual = _unusual_fields_for_sample(meta_df, meta_stats, folder, base, zthr=3.0, maxn=6) if isinstance(meta_df, pd.DataFrame) else []
        imgs = []
        if crop_thumb:
            imgs.append(f"<div><div class='muted'>cropped</div><img src='{crop_thumb}'/></div>")
        elif crop:
            imgs.append(f"<div><div class='muted'>cropped</div><div class='muted'>{crop}</div></div>")
        else:
            imgs.append(f"<div><div class='muted'>cropped</div><div class='muted'>not found</div></div>")
        if orig_thumb:
            imgs.append(f"<div><div class='muted'>original</div><img src='{orig_thumb}'/></div>")
        elif orig:
            imgs.append(f"<div><div class='muted'>original</div><div class='muted'>{orig}</div></div>")
        else:
            imgs.append(f"<div><div class='muted'>original</div><div class='muted'>not found</div></div>")
        meta_html = "<br/>".join(unusual) if unusual else "<span class='muted'>no unusual numeric metadata</span>"
        cards.append(f"""
            <tr>
              <td style="white-space:nowrap">{folder}</td>
              <td>{base}</td>
              <td>{de:.4f}</td>
              <td class="thumb"><div class="grid">{''.join(imgs)}</div></td>
              <td class="meta">{meta_html}</td>
            </tr>
        """)
    html.append("<table><tr><th>folder</th><th>basename</th><th>ΔE00</th><th>preview</th><th>metadata (|z|≥3)</th></tr>")
    html.extend(cards)
    html.append("</table>")

    html.append("</body></html>")
    out_html = os.path.join(outdir, f"{split}_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[ok] Wrote HTML report → {out_html}")

# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def analyze_one(ckpt_path: str,
                images_dir: str, json_dir: str, labels_csv: str,
                mode: str, outdir_root: str,
                test_split_file: Optional[str], val_split_file: Optional[str],
                device: Optional[str], bins: Optional[int] = None) -> Dict[str, Optional[str]]:
    device = _device_from_arg(device)
    obj, cfg = _load_ckpt(ckpt_path)
    run_id = _run_id_from_cfg(cfg)

    color_space = cfg.get("color_space", "lab")
    pred_act    = cfg.get("pred_activation", "none")
    act_eps    = _floatish(cfg.get("activation_eps"), 1e-3)
    batch_size = _intish(cfg.get("batch_size"), 64)
    workers    = _intish(cfg.get("workers"), 0)
    seed       = _intish(cfg.get("seed"), 100)
    model_name  = cfg.get("model", "film")
    backbone    = cfg.get("backbone", "smallcnn")
    token_stage = _token_stage_from_cfg(cfg, default=-2)
    pretrained  = _bool(cfg.get("pretrained", True))

    outdir = os.path.join(outdir_root, run_id); _safe_makedirs(outdir)
    wrote = {"val_csv": None, "test_csv": None}

    if mode in ("val","both"):
        dm, loader = _build_dm("val", cfg, images_dir, json_dir, labels_csv,
                               batch_size, workers, seed, val_split_file, mode)
        # Use the checkpoint weights and ensure device matches
        model = _build_model(obj, getattr(dm, "meta_dim", 0),
                                model_name, backbone, device,
                                token_stage=token_stage, pretrained=pretrained)
        rows = _collect_per_sample_de(model, loader, device, color_space, pred_act, act_eps)
        df = pd.DataFrame(rows); df.insert(0, "split", "val")
        path = os.path.join(outdir, "val_per_sample.csv"); df.to_csv(path, index=False)
        _write_histograms(df, outdir, "val", bins=bins)
        folder_csv = os.path.join(outdir, "val_by_folder_stats.csv")
        _write_folder_stats(df, folder_csv)
        # HTML report
        # HTML report (prepend header with cfg + checkpoint path)
        _write_html_report(df, outdir, "val", images_dir=images_dir,
                           original_dir=json_dir, json_dir=json_dir,
                           folder_stats_csv=folder_csv,
                           header_cfg=cfg, ckpt_path=ckpt_path)
        wrote["val_csv"] = path

    if mode in ("test","both"):
        if not test_split_file:
            raise ValueError("--test_split_file is required for mode=test/both")
        dm, loader = _build_dm("test", cfg, images_dir, json_dir, labels_csv, batch_size, workers, seed, test_split_file, mode)
        model = _build_model(obj, getattr(dm, "meta_dim", 0),
                            model_name, backbone, device,
                            token_stage=token_stage, pretrained=pretrained)
        rows = _collect_per_sample_de(model, loader, device, color_space, pred_act, act_eps)
        df = pd.DataFrame(rows); df.insert(0, "split", "test")
        path = os.path.join(outdir, "test_per_sample.csv"); df.to_csv(path, index=False)
        _write_histograms(df, outdir, "test", bins=bins)
        folder_csv = os.path.join(outdir, "test_by_folder_stats.csv")
        _write_folder_stats(df, folder_csv)
        _write_html_report(df, outdir, "test", images_dir=images_dir,
                           original_dir=json_dir, json_dir=json_dir,
                           folder_stats_csv=folder_csv,
                               header_cfg=cfg, ckpt_path=ckpt_path)
        wrote["test_csv"] = path

    return wrote

def analyze_from_csv(runs_csv: str,
                     images_dir: str, json_dir: str, labels_csv: str,
                     mode: str, outdir_root: str,
                     test_split_file: Optional[str] = None, val_split_file: Optional[str] = None,
                     device: Optional[str] = None, bins: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)

    # Accept several column names
    cand_cols = [c for c in ("ckpt_path","json_path","path") if c in df.columns]
    if not cand_cols:
        raise ValueError(f"{runs_csv} needs one of: ckpt_path / json_path / path.")

    def _ckpt_from_json(json_path: str) -> Optional[str]:
        try:
            with open(json_path, "r") as f:
                obj = json.load(f)
            for k in ("ckpt","ckpt_path","checkpoint","checkpoint_path"):
                v = obj.get(k)
                if isinstance(v, str) and os.path.isfile(v):
                    return v
        except Exception:
            return None
        return None

    ckpts: List[str] = []
    for _, row in df.iterrows():
        raw = None
        for c in cand_cols:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                raw = v.strip(); break
        if not raw: 
            continue
        if raw.endswith(".json"):
            ck = _ckpt_from_json(raw)
            if ck and os.path.isfile(ck):
                ckpts.append(ck)
            else:
                print(f"[skip] could not resolve ckpt from TEST json: {raw}")
        elif os.path.isfile(raw):
            ckpts.append(raw)
        else:
            print(f"[skip] missing ckpt: {raw}")

    records = []
    for ck in ckpts:
        try:
            wrote = analyze_one(ck, images_dir, json_dir, labels_csv, mode, outdir_root,
                                test_split_file, val_split_file, device, bins=bins)
            records.append({"ckpt_path": ck, **wrote})
        except Exception as e:
            print(f"[error] analyze_one failed for {ck}: {e}")

    out = pd.DataFrame(records)
    if len(out) > 0:
        _safe_makedirs(outdir_root)
        manifest = os.path.join(outdir_root, "analysis_manifest.csv")
        out.to_csv(manifest, index=False)
        print(f"[ok] Wrote manifest: {manifest}")
    else:
        print("[warn] no checkpoints analyzed.")
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="ΔE00 distribution analysis + HTML reports (VAL/TEST).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", nargs="+", help="One or more checkpoint paths (.pt/.pth).")
    src.add_argument("--runs_csv", help="Sweep summary CSV (must include 'ckpt_path' or a json path).")

    p.add_argument("--images_dir", required=True)
    p.add_argument("--json_dir",   required=True)
    p.add_argument("--labels_csv", required=True)

    p.add_argument("--mode", choices=("val","test","both"), default="both")
    p.add_argument("--test_split_file", default=None, help="Required for mode=test/both.")
    p.add_argument("--val_split_file",  default=None, help="Optional persistent VAL split JSON.")
    p.add_argument("--outdir", default="./results/analysis", help="Output root.")

    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if omitted)")
    p.add_argument("--bins", type=int, default=None, help="Override histogram bins (else auto).")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    if args.ckpt:
        for pth in args.ckpt:
            analyze_one(pth, args.images_dir, args.json_dir, args.labels_csv,
                        args.mode, args.outdir, args.test_split_file, args.val_split_file,
                        args.device, bins=args.bins)
        return 0
    if args.runs_csv:
        analyze_from_csv(args.runs_csv, args.images_dir, args.json_dir, args.labels_csv,
                         args.mode, args.outdir, args.test_split_file, args.val_split_file,
                         args.device, bins=args.bins)
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
