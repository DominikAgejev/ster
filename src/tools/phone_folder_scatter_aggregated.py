# src/tools/phone_folder_scatter_aggregated.py
from __future__ import annotations
import os, argparse, itertools, hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Visual style
# ----------------------------
WARM_PALETTE = [
    "#f2c9a1", "#e5989b", "#d17a22", "#b56576", "#f6bd60",
    "#b55d4c", "#cd9a7b", "#f28482", "#c7a27e", "#6b705c",
]
def warm(i: int) -> str: return WARM_PALETTE[i % len(WARM_PALETTE)]

# ----------------------------
# Heuristics & defaults
# ----------------------------
# Substring matches (case-insensitive)
PSAMPLE_KEYWORDS_DEFAULT = [
    "per_sample", "per-sample", "test_per_sample", "sample_metrics", "perimage", "per_image"
]
PFOLDER_KEYWORDS_DEFAULT  = [
    "by_folder", "folder_stats", "per_folder", "by-folder", "folder-metrics", "per-folder"
]

SEARCH_COLS = ["folder", "device", "dataset", "source", "id", "path"]
BRAND_DEFAULTS = {
    "iphone":  ["iphone"],
    "pixel":   ["pixel"],
    "samsung": ["samsung"],
}
METRIC_CANDS = (
    "de00","δe00","deltae00","Δe00","delta_e00",
    "value","metric","score","mean"
)
COUNT_CANDS = (
    "count","n","num","samples","size","num_images","n_images","images","files","n_files"
)

# ----------------------------
# Utilities
# ----------------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "feature" in df.columns and "features" not in df.columns:
        df.rename(columns={"feature":"features"}, inplace=True)
    return df

def ensure_folder(df: pd.DataFrame) -> pd.DataFrame:
    if "folder" in df.columns:
        return df
    for col in ["path","filepath","file","image","img_path"]:
        if col in df.columns:
            folders = df[col].astype(str).str.replace("\\","/",regex=False).str.rsplit("/",2).str[-2]
            df = df.copy(); df["folder"] = folders
            return df
    df = df.copy(); df["folder"] = "unknown"
    return df

def find_metric_col(df: pd.DataFrame, prefer: Optional[str]=None) -> Optional[str]:
    if prefer and prefer in df.columns: return prefer
    lowers = [c.lower() for c in df.columns]
    for cand in METRIC_CANDS:
        if cand in lowers: return df.columns[lowers.index(cand)]
    for cand in METRIC_CANDS:
        for i,c in enumerate(lowers):
            if cand in c: return df.columns[i]
    return None

def find_count_col(df: pd.DataFrame) -> Optional[str]:
    lowers = [c.lower() for c in df.columns]
    for cand in COUNT_CANDS:
        if cand in lowers: return df.columns[lowers.index(cand)]
    # also accept columns that *look* like counts
    for i,c in enumerate(lowers):
        if any(k in c for k in ["count","num","n_","_n","size","images","files"]):
            return df.columns[i]
    return None

def parse_brand_map(s: Optional[str]) -> Dict[str, List[str]]:
    if not s:
        return {k:list(v) for k,v in BRAND_DEFAULTS.items()}
    out: Dict[str,List[str]] = {}
    for part in s.split(";"):
        part = part.strip()
        if not part: continue
        if "=" in part:
            k, v = part.split("=",1)
            pats = [x.strip().lower() for x in v.split(",") if x.strip()]
            if pats: out[k.strip().lower()] = pats
        else:
            out[part.lower()] = [part.lower()]
    return out or {k:list(v) for k,v in BRAND_DEFAULTS.items()}

def detect_brand_row(row: pd.Series, brand_map: Dict[str, List[str]]) -> Optional[str]:
    blob = []
    for c in SEARCH_COLS:
        if c in row.index and pd.notna(row[c]):
            blob.append(str(row[c]).lower())
    text = " | ".join(blob)
    for brand, pats in brand_map.items():
        for p in pats:
            if p in text: return brand
    return None

def run_id_from_path(path: str, root: str) -> str:
    """A stable run-id from the path (first subdir under root); fallback to hash."""
    try:
        rel = os.path.relpath(os.path.dirname(path), root)
        parts = rel.replace("\\","/").split("/")
        if len(parts) >= 1 and parts[0] not in (".",""):
            return parts[0]
    except Exception:
        pass
    return hashlib.md5(path.encode("utf-8")).hexdigest()[:10]

# ----------------------------
# Scanning
# ----------------------------
def _matches_any_keywords(name: str, keywords: List[str]) -> bool:
    name = name.lower()
    return name.endswith(".csv") and any(k.lower() in name for k in keywords)

def scan_csvs(root: str,
              psample_keywords: List[str],
              pfolder_keywords: List[str]) -> Tuple[List[str], List[str]]:
    psample, pfolder = [], []
    for d, _, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if _matches_any_keywords(fl, psample_keywords):
                psample.append(os.path.join(d,f))
            elif _matches_any_keywords(fl, pfolder_keywords):
                pfolder.append(os.path.join(d,f))
    return psample, pfolder

# ----------------------------
# Aggregation logic
# ----------------------------
def aggregate_true_means(
    root: str,
    prefer_metric: Optional[str],
    brand_map: Dict[str,List[str]],
    psample_keywords: List[str],
    pfolder_keywords: List[str],
) -> pd.DataFrame:
    """
    Returns one row per (brand, folder, features) with a *true* mean across all runs,
    using weighted aggregation (by sample count when available).
    """
    ps_paths, pf_paths = scan_csvs(root, psample_keywords, pfolder_keywords)
    print(f"[scan] per-sample CSVs: {len(ps_paths)} | by-folder CSVs: {len(pf_paths)}")

    # If a run has both per-sample and per-folder, prefer per-sample (most accurate).
    run_has_ps: Dict[str,bool] = {}
    for p in ps_paths:
        rid = run_id_from_path(p, root)
        run_has_ps[rid] = True

    contribs = []  # rows: brand, folder, features, sum_w, total_w, run_id

    # ---- Per-sample contributions (best) ----
    for path in ps_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[read] skip (per-sample) {path}: {e}")
            continue
        df = norm_cols(df); df = ensure_folder(df)
        metric = find_metric_col(df, prefer_metric)
        if not metric or "features" not in df.columns:
            print(f"[skip] no metric/features in per-sample: {path}")
            continue
        rid = run_id_from_path(path, root)
        tmp = df.copy()
        # Add an absolute path column to help brand detection if needed
        if "path" not in tmp.columns:
            tmp["path"] = path
        tmp["brand"] = tmp.apply(lambda r: detect_brand_row(r, brand_map), axis=1)
        tmp = tmp.dropna(subset=["brand"])
        g = tmp.groupby(["brand","folder","features"], as_index=False)[metric].agg(["sum","count"]).reset_index()
        g.columns = ["brand","folder","features","sum_w","total_w"]
        g["run_id"] = rid
        contribs.append(g)

    # ---- Per-folder contributions (fallback) ----
    for path in pf_paths:
        rid = run_id_from_path(path, root)
        if run_has_ps.get(rid, False):
            # skip this run's by-folder if per-sample exists
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[read] skip (by-folder) {path}: {e}")
            continue
        df = norm_cols(df); df = ensure_folder(df)
        metric = "mean" if "mean" in df.columns else find_metric_col(df, prefer_metric)
        if not metric or "features" not in df.columns:
            print(f"[skip] no metric/features in by-folder: {path}")
            continue
        cnt_col = find_count_col(df)
        cnt = df[cnt_col] if cnt_col and cnt_col in df.columns else pd.Series(1, index=df.index)

        tmp = df.copy()
        if "path" not in tmp.columns:
            tmp["path"] = path
        tmp["brand"] = tmp.apply(lambda r: detect_brand_row(r, brand_map), axis=1)
        tmp = tmp.dropna(subset=["brand"])

        tmp["sum_w"] = pd.to_numeric(tmp[metric], errors="coerce") * pd.to_numeric(cnt, errors="coerce")
        tmp["total_w"] = pd.to_numeric(cnt, errors="coerce")
        g = tmp.groupby(["brand","folder","features"], as_index=False)[["sum_w","total_w"]].sum()
        g["run_id"] = rid
        contribs.append(g)

    if not contribs:
        return pd.DataFrame(columns=["brand","folder","features","mean","total_count","runs"])

    allc = pd.concat(contribs, ignore_index=True)

    # Collapse across runs -> one true mean per (brand, folder, features)
    agg = (allc.groupby(["brand","folder","features"], as_index=False)
               .agg(sum_w=("sum_w","sum"),
                    total_w=("total_w","sum"),
                    runs=("run_id","nunique")))
    agg["mean"] = agg["sum_w"] / agg["total_w"]
    agg = agg.dropna(subset=["mean"])
    agg["total_count"] = agg["total_w"].astype(float)
    return agg[["brand","folder","features","mean","total_count","runs"]]

# ----------------------------
# Plots
# ----------------------------
def feature_pairs(feats: List[str], limit: Optional[int]) -> List[Tuple[str,str]]:
    pairs = [(a,b) for a,b in itertools.combinations(feats, 2)]
    return pairs[:limit] if limit else pairs

def plot_brand_scatters(per_folder: pd.DataFrame,
                        brand: str,
                        outdir: str,
                        point_size: int = 110,
                        figsize: Tuple[int,int]=(6,6),
                        dpi: int = 160,
                        annotate: bool = False,
                        font_scale: float = 1.1):
    bdf = per_folder[per_folder["brand"] == brand]
    if bdf.empty:
        print(f"[plot] brand '{brand}': no rows, skipping.")
        return
    pivot = bdf.pivot_table(index="folder", columns="features", values="mean", aggfunc="mean")
    feats = [str(c) for c in pivot.columns if c is not None]
    if len(feats) < 2:
        print(f"[plot] brand '{brand}': <2 features, skipping.")
        return

    pairs = feature_pairs(feats, limit=None)
    for fa, fb in pairs:
        sub = pivot[[fa, fb]].dropna()
        if sub.empty: 
            continue

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        x, y = sub[fa].values, sub[fb].values
        ax.scatter(x, y, s=point_size, color=warm(0), alpha=0.9,
                   edgecolor="#2a1f1c", linewidth=0.6)

        if annotate:
            for folder, xv, yv in zip(sub.index, x, y):
                lab = os.path.basename(str(folder)).replace("_"," ")
                ax.annotate(lab, (xv,yv), xytext=(4,2),
                            textcoords="offset points", fontsize=9*font_scale)

        both = np.concatenate([x,y])
        lo, hi = float(np.min(both)), float(np.max(both))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo,hi],[lo,hi], linestyle="--", color="#555", linewidth=1)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)

        ax.set_title(f"{brand.capitalize()} — {fa} vs {fb}", fontsize=12*font_scale)
        ax.set_xlabel(str(fa), fontsize=10*font_scale)
        ax.set_ylabel(str(fb), fontsize=10*font_scale)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.grid(True, axis="both", linestyle=":", linewidth=0.7, alpha=0.5)

        os.makedirs(outdir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{brand}_scatter_{fa}_vs_{fb}.png"))
        plt.close(fig)

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Scan a root for runs, aggregate true per-folder means across runs, and plot brand scatters."
    )
    ap.add_argument("--root", required=True, help="Root directory to scan (walks recursively).")
    ap.add_argument("--metric", default=None, help="Metric column to prefer (else auto-detect ΔE/mean/etc).")
    ap.add_argument("--brands", default=None,
                    help="Brand map, e.g. 'iphone=iphone;pixel=pixel;samsung=samsung,galaxy'.")
    ap.add_argument("--outdir", default="./viz/phone_agg_scatter",
                    help="Output directory for CSVs and plots.")
    ap.add_argument("--pairs-limit", type=int, default=None, help="Limit number of feature pairs plotted.")
    ap.add_argument("--point-size", type=int, default=110, help="Scatter point size.")
    ap.add_argument("--figsize", default="6x6", help="Figure size WxH inches, e.g. 6x6.")
    ap.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    ap.add_argument("--annotate", action="store_true", help="Label points with folder names.")
    ap.add_argument("--font-scale", type=float, default=1.1, help="Scale axis/title font sizes.")
    ap.add_argument("--ps-patterns", default=None,
                    help="Comma-separated substrings for per-sample CSV filenames (case-insensitive).")
    ap.add_argument("--pf-patterns", default=None,
                    help="Comma-separated substrings for by-folder CSV filenames (case-insensitive).")
    args = ap.parse_args()

    ps_keys = [k.strip() for k in args.ps_patterns.split(",")] if args.ps_patterns else PSAMPLE_KEYWORDS_DEFAULT
    pf_keys = [k.strip() for k in args.pf_patterns.split(",")] if args.pf_patterns else PFOLDER_KEYWORDS_DEFAULT

    brand_map = parse_brand_map(args.brands)
    agg = aggregate_true_means(args.root, args.metric, brand_map, ps_keys, pf_keys)
    if agg.empty:
        print("[agg] No data aggregated. Check --root and filename patterns via --ps-patterns/--pf-patterns.")
        return

    # Save combined and per-brand CSVs
    os.makedirs(args.outdir, exist_ok=True)
    agg.to_csv(os.path.join(args.outdir, "per_folder_means_aggregated.csv"), index=False)
    for b in sorted(agg["brand"].unique()):
        agg[agg["brand"] == b].to_csv(os.path.join(args.outdir, f"per_folder_means_aggregated.{b}.csv"), index=False)

    # Plots per brand
    W, H = (int(float(x)) for x in args.figsize.lower().split("x"))
    for i, b in enumerate(sorted(agg["brand"].unique())):
        # limit pairs by availability
        pivot = agg[agg["brand"] == b].pivot_table(index="folder", columns="features", values="mean", aggfunc="mean")
        feats = [str(c) for c in pivot.columns if c is not None]
        pairs = [(a,c) for a,c in itertools.combinations(feats, 2)]
        if args.pairs_limit is not None:
            pairs = pairs[:args.pairs_limit]
        for (fa, fb) in pairs:
            sub = pivot[[fa, fb]].dropna()
            if sub.empty: continue
            fig, ax = plt.subplots(figsize=(W, H), dpi=args.dpi)
            x, y = sub[fa].values, sub[fb].values
            ax.scatter(x, y, s=args.point_size, color=warm(i), alpha=0.9,
                       edgecolor="#2a1f1c", linewidth=0.6)
            if args.annotate:
                for folder, xv, yv in zip(sub.index, x, y):
                    lab = os.path.basename(str(folder)).replace("_"," ")
                    ax.annotate(lab, (xv,yv), xytext=(4,2), textcoords="offset points", fontsize=9*args.font_scale)
            both = np.concatenate([x,y])
            lo, hi = float(np.min(both)), float(np.max(both))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo,hi],[lo,hi], linestyle="--", color="#555", linewidth=1)
            ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
            ax.set_title(f"{b.capitalize()} — {fa} vs {fb}", fontsize=12*args.font_scale)
            ax.set_xlabel(str(fa), fontsize=10*args.font_scale)
            ax.set_ylabel(str(fb), fontsize=10*args.font_scale)
            for s in ("top","right"): ax.spines[s].set_visible(False)
            ax.grid(True, axis="both", linestyle=":", linewidth=0.7, alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"{b}_scatter_{fa}_vs_{fb}.png"))
            plt.close(fig)

    print(f"[done] Wrote CSVs + plots to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
