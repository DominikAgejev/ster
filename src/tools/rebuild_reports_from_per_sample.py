# src/tools/rebuild_reports_from_per_sample.py
from __future__ import annotations
import os, argparse, re, sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

METRIC_COL_CANDIDATES = ("de00","δe00","deltae00","Δe00","delta_e00")
DEFAULT_MATCH_COLS = ["folder","dataset","device","id","basename","path","source"]

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "feature" in df.columns and "features" not in df.columns:
        df = df.rename(columns={"feature":"features"})
    return df

def find_metric_col(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns if any(k in c.lower() for k in METRIC_COL_CANDIDATES)]
    if not cols:
        return None
    for k in METRIC_COL_CANDIDATES:
        for c in df.columns:
            if c.lower() == k:
                return c
    return cols[0]

def filter_iphone(df: pd.DataFrame, patterns: List[str], cols: List[str],
                  case_sensitive: bool=False, regex: bool=False) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.iloc[0:0]
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        s = df[c].astype(str)
        for pat in patterns:
            m = s.str.contains(pat, case=case_sensitive, regex=regex, na=False)
            mask |= m.values
    return df.loc[mask].copy()

def series_stats(vals: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(vals, errors="coerce").dropna().astype("float64").values
    if arr.size == 0:
        return {}
    q = np.nanpercentile(arr, [5,25,50,75,90,95,99])
    median = float(np.nanmedian(arr))
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=0))
    mad = float(np.nanmedian(np.abs(arr - median)))
    q1, q3 = float(q[1]), float(q[3])
    iqr = q3 - q1
    cv = float(std / mean) if mean != 0 else np.nan
    return {
        "count": int(arr.size),
        "mean": mean,
        "std": std,
        "median": median,
        "mad": mad,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "cv": cv,
        "p5": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "p90_minus_p10": float(np.nanpercentile(arr, 90) - np.nanpercentile(arr, 10)),
        "p95_minus_p5": float(np.nanpercentile(arr, 95) - np.nanpercentile(arr, 5)),
    }

def compute_run_stats(csv_path: str, patterns: List[str], cols: List[str],
                      case_sensitive: bool=False, regex: bool=False) -> Dict[str, float]:
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    df = norm_cols(df)
    metric_col = find_metric_col(df)
    if not metric_col:
        return {}
    sub = filter_iphone(df, patterns, cols, case_sensitive=case_sensitive, regex=regex)
    if sub.empty:
        return {}
    return series_stats(sub[metric_col])

def maybe_rewrite_path(p: str, src_root: Optional[str], dst_root: Optional[str]) -> str:
    if not p or not src_root or not dst_root:
        return p
    src_root_n = os.path.normpath(src_root)
    if os.path.normpath(p).startswith(src_root_n):
        rest = os.path.normpath(p)[len(src_root_n):].lstrip("/\\")
        return os.path.normpath(os.path.join(dst_root, rest))
    return p

def main():
    ap = argparse.ArgumentParser(
        description="Rebuild an aggregated reports CSV with iPhone-only (or pattern-matched) per-sample stats."
    )
    ap.add_argument("--in-reports", required=True,
                    help="Input aggregated CSV (e.g., bg2_test_reports.csv). Must contain a 'csv_path' column.")
    ap.add_argument("--out", required=True, help="Output CSV path for the rebuilt (filtered) stats.")
    ap.add_argument("--patterns", action="append", default=["iphone"],
                    help="Substring/regex patterns to match in per-sample CSV rows (repeat/comma-separate). Default: 'iphone'.")
    ap.add_argument("--match-cols", default=",".join(DEFAULT_MATCH_COLS),
                    help=f"Comma-separated columns to search in per-sample CSVs. Default: {','.join(DEFAULT_MATCH_COLS)}")
    ap.add_argument("--regex", action="store_true", help="Treat patterns as regular expressions.")
    ap.add_argument("--case-sensitive", action="store_true", help="Case-sensitive matching (default off).")
    ap.add_argument("--drop-empty", action="store_true", help="Drop runs where no per-sample rows matched.")
    ap.add_argument("--min-count", type=int, default=1,
                    help="If >0, drop runs with fewer than this many matched samples.")
    ap.add_argument("--src-root", default=None,
                    help="If 'csv_path' was written on another machine, replace this leading prefix ...")
    ap.add_argument("--dst-root", default=None,
                    help="... with this prefix before reading per-sample CSVs.")
    args = ap.parse_args()

    patterns: List[str] = []
    for p in (args.patterns or []):
        patterns.extend([x.strip() for x in str(p).split(",") if x.strip()])

    match_cols = [c.strip() for c in str(args.match_cols).split(",") if c.strip()]

    rep = pd.read_csv(args.in_reports)
    rep = norm_cols(rep)
    if "csv_path" not in rep.columns:
        print("[rebuild] ERROR: input reports CSV lacks 'csv_path' column.", file=sys.stderr)
        sys.exit(2)

    rows = []
    for _, row in rep.iterrows():
        meta = row.to_dict()
        csv_path = str(meta.get("csv_path",""))
        csv_path = maybe_rewrite_path(csv_path, args.src_root, args.dst_root)
        stats = compute_run_stats(csv_path, patterns, match_cols,
                                  case_sensitive=args.case_sensitive, regex=args.regex)
        if stats and (args.min_count <= 1 or stats["count"] >= args.min_count):
            meta.update(stats)
        else:
            if args.drop_empty:
                continue
            for k in ["count","mean","std","median","mad","q1","q3","iqr","cv",
                      "p5","p25","p50","p75","p90","p95","p99","p90_minus_p10","p95_minus_p5"]:
                meta[k] = np.nan

        for k in ["id","model","backbone","features"]:
            if k not in meta:
                meta[k] = ""
        rows.append(meta)

    out = pd.DataFrame(rows)
    priority = ["id","model","backbone","features","included_folders","color_space","epochs","batch_size","seed",
                "token_stage","optimizer","lr","lr_schedule","weight_decay",
                "count","mean","std","median","mad","q1","q3","iqr","cv",
                "p5","p25","p50","p75","p90","p95","p99","p90_minus_p10","p95_minus_p5",
                "ckpt_path","html_path","csv_path","hist_overall","hist_zoom","pretrained"]
    ordered = [c for c in priority if c in out.columns] + [c for c in out.columns if c not in priority]
    out = out[ordered]
    out.to_csv(args.out, index=False)
    print(f"[rebuild] Wrote: {os.path.abspath(args.out)}  (rows: {len(out)})")

if __name__ == "__main__":
    main()
