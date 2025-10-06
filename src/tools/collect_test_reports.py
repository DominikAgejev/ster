# src/tools/collect_test_reports.py
from __future__ import annotations
import os, re, csv, json, argparse
from typing import Dict, List, Optional, Tuple

# Optional but preferred for robust CSV/Stats
try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False

# Optional HTML parser
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
    _HAVE_BS4 = True
except Exception:
    _HAVE_BS4 = False


# ---------------------------
# Small helpers
# ---------------------------
def _safe_float(x, default=None):
    try:
        if x is None: return default
        s = str(x).strip().replace("—", "").replace(",", "")
        if s == "": return default
        return float(s)
    except Exception:
        return default

def _stats_from_series(vals: List[float]) -> Dict[str, float]:
    """Compute the same keys as the HTML Overview when only CSV is available."""
    if not vals:
        return {}
    if _HAVE_PANDAS:
        import numpy as np
        s = pd.Series(vals, dtype="float64")
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        mad = (s - s.median()).abs().median()
        out = {
            "count": float(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "median": float(s.median()),
            "mad": float(mad),
            "q1": float(q.loc[0.25]),
            "q3": float(q.loc[0.75]),
            "iqr": float(q.loc[0.75] - q.loc[0.25]),
            "cv": float(s.std(ddof=1) / s.mean()) if s.mean() != 0 else None,
            "p5": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]),
            "p90": float(q.loc[0.90]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
        }
        # P90 - P10 / P95 - P5: need 0.10 and 0.90 too
        p10 = float(s.quantile(0.10))
        out["p90_minus_p10"] = float(out["p90"] - p10)
        out["p95_minus_p5"]  = float(out["p95"] - out["p5"])
        return out
    else:
        # Minimal fallback if pandas is unavailable
        import statistics as st
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n
        std = (st.pstdev(vals_sorted) if n > 1 else 0.0)  # population std as fallback
        med = vals_sorted[n // 2] if n % 2 else 0.5 * (vals_sorted[n//2 - 1] + vals_sorted[n//2])
        return {"count": float(n), "mean": mean, "std": std, "median": med}

def _rel_id(run_dir: str, root: str) -> str:
    """Use path relative to search root as the run id."""
    rel = os.path.relpath(run_dir, root).replace("\\", "/")
    # Drop trailing 'final_test' if present
    parts = [p for p in rel.split("/") if p]
    if parts and parts[-1] == "final_test":
        parts = parts[:-1]
    return "/".join(parts) or os.path.basename(run_dir)

def _guess_model_bits_from_path(run_dir: str) -> Dict[str, Optional[str]]:
    """
    Try to parse 'xattn__resnet18__image+mean+meta__...' style folder names if HTML is missing.
    """
    base = os.path.basename(run_dir)
    bits = base.split("__")
    out = {"model": None, "backbone": None, "features": None}
    if len(bits) >= 3:
        out["model"], out["backbone"], out["features"] = bits[0], bits[1], bits[2]
    return out


# ---------------------------
# HTML parsing
# ---------------------------
def parse_html_report(html_path: str) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str]]:
    """
    Returns:
      header: model/backbone/features/... + 'ckpt_path' if present
      stats:  overview stats (mean, std, median, ...)
      assets: paths to histogram images
    """
    header: Dict[str, str] = {}
    stats: Dict[str, float] = {}
    assets: Dict[str, str] = {}

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    if _HAVE_BS4:
        soup = BeautifulSoup(html, "html.parser")

        # Checkpoint (optional)
        run_hdr = soup.find(id="run-header")
        if run_hdr:
            code = run_hdr.find("code")
            if code:
                header["ckpt_path"] = code.get_text(strip=True)

            # Parse the 2-col table under the header
            t = run_hdr.find("table")
            if t:
                for tr in t.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) >= 2:
                        k = tds[0].get_text(strip=True).lower().replace(" ", "_")
                        v = tds[1].get_text(strip=True)
                        header[k] = v

        # Overview block
        over_h2 = None
        for h2 in soup.find_all("h2"):
            if h2.get_text(strip=True).lower().startswith("overview"):
                over_h2 = h2
                break
        if over_h2:
            tbl = over_h2.find_next("table")
            if tbl:
                for tr in tbl.find_all("tr"):
                    th = tr.find("th")
                    td = tr.find("td")
                    if not th or not td:
                        continue
                    key = th.get_text(strip=True).lower()
                    val = _safe_float(td.get_text(strip=True))
                    if val is not None:
                        stats[key] = val

        # Histograms
        for h2 in soup.find_all("h2"):
            name = h2.get_text(strip=True).lower()
            if "histogram" in name:
                img = h2.find_next("img")
                if img and img.get("src"):
                    src = img["src"]
                    if "zoom" in src:
                        assets["hist_zoom"] = os.path.abspath(os.path.join(os.path.dirname(html_path), src))
                    else:
                        assets["hist_overall"] = os.path.abspath(os.path.join(os.path.dirname(html_path), src))
    else:
        # Regex fallback (very lenient)
        m_ck = re.search(r"Checkpoint:\s*<code[^>]*>([^<]+)</code>", html, re.I)
        if m_ck:
            header["ckpt_path"] = m_ck.group(1).strip()

        # Key rows in "Run Summary" table (Model/Backbone/Features)
        for key in ("Model", "Backbone", "Features", "Included folders", "Color space",
                    "Epochs", "Batch size", "Seed", "Token stage", "Pretrained",
                    "Optimizer", "LR", "LR schedule", "Weight decay"):
            rx = rf"{re.escape(key)}</td><td[^>]*>([^<]+)</td>"
            m = re.search(rx, html, re.I)
            if m:
                header[key.lower().replace(" ", "_")] = m.group(1).strip()

        # Overview table rows like <tr><th>mean</th><td>0.806383</td></tr>
        for k in ("count","mean","std","median","mad","q1","q3","iqr","cv",
                  "p5","p25","p50","p75","p90","p95","p99","p90_minus_p10","p95_minus_p5"):
            m = re.search(rf"<th>\s*{k}\s*</th><td>([^<]+)</td>", html, re.I)
            if m:
                v = _safe_float(m.group(1))
                if v is not None:
                    stats[k] = v

        # Images
        for tag in re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"][^>]*>", html, re.I):
            p = os.path.abspath(os.path.join(os.path.dirname(html_path), tag))
            if "zoom" in os.path.basename(tag):
                assets["hist_zoom"] = p
            elif "hist" in os.path.basename(tag):
                assets["hist_overall"] = p

    return header, stats, assets


# ---------------------------
# CSV parsing (per-sample)
# ---------------------------
def parse_per_sample_csv(csv_path: str) -> Dict[str, float]:
    """Compute stats from test_per_sample.csv if present (column names: de00 / ΔE00 / deltaE00)."""
    if _HAVE_PANDAS:
        df = pd.read_csv(csv_path)
        cand = [c for c in df.columns if c.lower() in ("de00", "δe00", "deltae00", "Δe00", "Δe00 ", "delta_e00")]
        if not cand:
            # Try a looser match
            cand = [c for c in df.columns if "e00" in c.lower()]
        if not cand:
            return {}
        vals = df[cand[0]].astype("float64").tolist()
    else:
        # stdlib fallback
        vals = []
        with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            key = None
            for row in reader:
                if key is None:
                    # pick any column containing e00
                    for c in row:
                        if "e00" in c.lower():
                            key = c
                            break
                if key and row.get(key) not in (None, ""):
                    try:
                        vals.append(float(row[key]))
                    except Exception:
                        pass
    return _stats_from_series(vals)


# ---------------------------
# Walker
# ---------------------------
def find_runs(root: str) -> List[str]:
    """Return directories that contain a test_report.html (our run marker)."""
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        if "test_report.html" in filenames:
            hits.append(dirpath)
    return sorted(hits)

def collect_one(run_dir: str, root: str) -> Dict[str, object]:
    html_path = os.path.join(run_dir, "test_report.html")
    csv_path  = os.path.join(run_dir, "test_per_sample.csv")
    row: Dict[str, object] = {}

    header, stats, assets = ({}, {}, {})
    if os.path.exists(html_path):
        header, stats, assets = parse_html_report(html_path)

    # Minimal identity
    row["id"] = _rel_id(run_dir, root)
    # Prefer header metadata; else try to guess from folder
    row["model"]    = header.get("model")    or _guess_model_bits_from_path(os.path.dirname(run_dir)).get("model")
    row["backbone"] = header.get("backbone") or _guess_model_bits_from_path(os.path.dirname(run_dir)).get("backbone")
    row["features"] = header.get("features") or _guess_model_bits_from_path(os.path.dirname(run_dir)).get("features")

    # Common knobs (optional, if present in HTML)
    for k in ("included_folders","color_space","epochs","batch_size","seed",
              "token_stage","pretrained","optimizer","lr","lr_schedule","weight_decay"):
        row[k] = header.get(k)

    # Stats (Overview)
    wanted = ["count","mean","std","median","mad","q1","q3","iqr","cv",
              "p5","p25","p50","p75","p90","p95","p99","p90_minus_p10","p95_minus_p5"]
    for k in wanted:
        if k in stats:
            row[k] = stats[k]

    # If stats missing, try computing from per-sample CSV
    if not any(k in row for k in ("mean","std","median")) and os.path.exists(csv_path):
        csv_stats = parse_per_sample_csv(csv_path)
        for k, v in csv_stats.items():
            row[k] = v

    # Assets & paths
    row["html_path"]       = os.path.abspath(html_path) if os.path.exists(html_path) else None
    row["csv_path"]        = os.path.abspath(csv_path)  if os.path.exists(csv_path)  else None
    row["hist_overall"]    = assets.get("hist_overall")
    row["hist_zoom"]       = assets.get("hist_zoom")
    row["ckpt_path"]       = header.get("ckpt_path")

    return row


def write_csv(rows: List[Dict[str, object]], out_csv: str) -> None:
    if not rows:
        # create an empty file with a minimal header so downstream code doesn't break
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id","model","backbone","features","mean","std","median","html_path","csv_path","hist_overall","hist_zoom"])
        return

    # union of keys across all rows → stable column order with important ones first
    priority = ["id","model","backbone","features",
                "included_folders","color_space","epochs","batch_size","seed","token_stage","optimizer","lr","lr_schedule","weight_decay",
                "count","mean","std","median","mad","q1","q3","iqr","cv","p5","p25","p50","p75","p90","p95","p99","p90_minus_p10","p95_minus_p5",
                "ckpt_path","html_path","csv_path","hist_overall","hist_zoom"]
    all_keys = set().union(*[r.keys() for r in rows])
    tail = [k for k in sorted(all_keys) if k not in priority]
    cols = [k for k in priority if k in all_keys] + tail

    if _HAVE_PANDAS:
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_csv, index=False)
    else:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Collect per-run TEST stats from analysis HTMLs into a single CSV."
    )
    ap.add_argument("--root", required=True, help="Directory to scan (walks recursively).")
    ap.add_argument("--out_csv", default=None, help="Output CSV path (default: <root>/collected_test_reports.csv)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_csv = os.path.abspath(args.out_csv or os.path.join(root, "collected_test_reports.csv"))

    run_dirs = find_runs(root)
    if not run_dirs:
        print(f"[collect] No test_report.html files found under: {root}")
        write_csv([], out_csv)
        print(f"[collect] Wrote empty scaffold CSV: {out_csv}")
        return

    rows = []
    for d in run_dirs:
        try:
            rows.append(collect_one(d, root))
        except Exception as e:
            print(f"[collect][warn] Failed to parse {d}: {e}")

    write_csv(rows, out_csv)
    print(f"[collect] Parsed {len(rows)} runs.")
    print(f"[collect] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
