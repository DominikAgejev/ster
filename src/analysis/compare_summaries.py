# src/analysis/compare_summaries.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Sequence, Dict
import numpy as np
import pandas as pd

# Matplotlib only; one chart per figure; no specific colors.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_CANDIDATE_GROUPS = ["model","backbone","features","included_folders","mse_weight_epochs"]
NEVER_KEYS = {"best_metric","best_epoch","ckpt_path"}

def _canon_val(x):
    if pd.isna(x): return x
    s=str(x); t=s.strip()
    if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}")):
        try:
            import json as _json
            obj=_json.loads(t.replace("'", '"'))
            return _json.dumps(obj, sort_keys=True, separators=(",", ":"))
        except Exception:
            return s
    return s

def _dedup_best(df: pd.DataFrame, keys: Sequence[str], metric: str, lower_is_better: bool) -> pd.DataFrame:
    return df.sort_values(metric, ascending=lower_is_better).drop_duplicates(subset=list(keys), keep="first").reset_index(drop=True)

def _detect_group_cols(common_cols: Sequence[str]) -> List[str]:
    return [c for c in DEFAULT_CANDIDATE_GROUPS if c in common_cols]

def _load_csv(path: Path) -> pd.DataFrame:
    df=pd.read_csv(path); df.columns=[c.strip() for c in df.columns]; return df

def _format_top_table(df: pd.DataFrame, show_cols: List[str], metric: str, k: int) -> str:
    cols=[c for c in show_cols if c in df.columns][:5]
    if metric not in cols: cols += [metric]
    return df[cols].head(k).to_markdown(index=False)

def _slugify_group(row: pd.Series, group_cols: List[str], max_len: int = 32) -> str:
    def _short(v: object) -> str:
        s = str(v)
        # strip paths and dotted prefixes
        if "/" in s: s = s.split("/")[-1]
        if "." in s: s = s.split(".")[-1]
        # collapse very long tokens
        if len(s) > 18:
            s = s[:9] + "…" + s[-8:]
        return s

    parts = [_short(row[c]) for c in group_cols if c in row.index]
    one_line = " | ".join(parts)
    if len(one_line) <= max_len:
        return one_line
    # fall back to multi-line if still too long
    return "\n".join(parts)

def _build_md_report(comp, onlyA, onlyB, meta, metric, group_cols,
                     bestA, bestB, best_group_tbl, grouped_summary,
                     top_improve, top_regress, topk_A, topk_B,
                     delta_hist, delta_quantiles, plots: Dict[str,str]) -> str:
    lines: List[str] = []
    lines.append("# Summary Comparison Report")
    lines.append("")
    lines.append(f"**Source A:** `{meta['source_A']}`  ")
    lines.append(f"**Source B:** `{meta['source_B']}`  ")
    lines.append(f"**Metric:** `{metric}`  (lower is better: `{meta['lower_is_better']}`)  ")
    lines.append(f"**Tie tolerance:** `{meta['tie_tolerance']}`  ")
    lines.append(f"**Matched on keys ({len(meta['match_keys_used'])}):** `{', '.join(meta['match_keys_used'])}`  ")
    lines.append(f"**Ignored columns:** `{', '.join(meta['ignore_cols'])}`")
    lines.append("")
    lines.append("## At a Glance")
    lines.append("")
    cnt=meta.get("counts",{})
    lines.append(f"- Rows in A: **{cnt.get('A_rows',0)}**")
    lines.append(f"- Rows in B: **{cnt.get('B_rows',0)}**")
    lines.append(f"- Unique (by match keys) in A: **{cnt.get('A_unique_by_keys',0)}**")
    lines.append(f"- Unique (by match keys) in B: **{cnt.get('B_unique_by_keys',0)}**")
    lines.append(f"- Matched: **{cnt.get('matched',0)}**")
    lines.append(f"- Only in A: **{cnt.get('only_in_A',0)}**")
    lines.append(f"- Only in B: **{cnt.get('only_in_B',0)}**")
    lines.append("")

    # Best runs
    lines.append("## Best Runs")
    lines.append("")
    if bestA is not None and bestB is not None and metric in bestA and metric in bestB:
        vA=float(bestA[metric]); vB=float(bestB[metric]); dv=vB-vA
        verdict="same"
        if meta["lower_is_better"]:
            if abs(dv) > meta["tie_tolerance"]: verdict = "better" if dv < 0 else "worse"
        else:
            if abs(dv) > meta["tie_tolerance"]: verdict = "better" if dv > 0 else "worse"
        lines.append(f"- **Global best A**: {metric} = {vA:.4f}")
        lines.append(f"- **Global best B**: {metric} = {vB:.4f}")
        lines.append(f"- **Δ(B−A)** on global bests: {dv:+.4f} → **{verdict}**")
        lines.append("")
    else:
        lines.append("_Best runs unavailable._")
        lines.append("")

    # Top-K
    if topk_A is not None or topk_B is not None:
        lines.append("## Top‑K Best Runs within Each Summary")
        lines.append("")
        if topk_A is not None:
            lines.append("### Top‑K in A")
            lines.append(_format_top_table(topk_A, ["model","backbone","features","included_folders","mse_weight_epochs"], metric, k=len(topk_A)))
            lines.append("")
        if topk_B is not None:
            lines.append("### Top‑K in B")
            lines.append(_format_top_table(topk_B, ["model","backbone","features","included_folders","mse_weight_epochs"], metric, k=len(topk_B)))
            lines.append("")

    # Matched stats
    if comp is not None and not comp.empty and "verdict" in comp.columns:
        vc=comp["verdict"].value_counts()
        lines.append("## Matched Verdicts & Δ Stats")
        lines.append("")
        lines.append(f"- **better**: {int(vc.get('better',0))}  ")
        lines.append(f"- **same**: {int(vc.get('same',0))}  ")
        lines.append(f"- **worse**: {int(vc.get('worse',0))}  ")
        lines.append("")
        if "delta_B_minus_A" in comp.columns:
            d=comp["delta_B_minus_A"]
            lines.append("### Delta Stats (B − A)")
            lines.append("")
            lines.append(f"- mean: {d.mean():.4f}")
            lines.append(f"- median: {d.median():.4f}")
            lines.append(f"- min: {d.min():.4f}")
            lines.append(f"- max: {d.max():.4f}")
            lines.append(f"- std: {d.std():.4f}")
            if delta_quantiles:
                lines.append(f"- q05/q25/q50/q75/q95: {delta_quantiles['q05']:.4f} / {delta_quantiles['q25']:.4f} / {delta_quantiles['q50']:.4f} / {delta_quantiles['q75']:.4f} / {delta_quantiles['q95']:.4f}")
            lines.append("")

    else:
        lines.append("_No matched comparisons to summarize._")
        lines.append("")

    # Grouped summary
    if group_cols and grouped_summary is not None and not grouped_summary.empty:
        lines.append("## Grouped Summary (Matched Head‑to‑Head)")
        lines.append("")
        lines.append(f"_Grouped by:_ `{', '.join(group_cols)}`")
        lines.append("")
        lines.append(grouped_summary.to_markdown(index=False))
        lines.append("")

    # Best by group minima
    if group_cols and best_group_tbl is not None and not best_group_tbl.empty:
        lines.append("## Best by Group (A vs B Minima per Group)")
        lines.append("")
        lines.append(best_group_tbl.to_markdown(index=False))
        if "verdict_best" in best_group_tbl.columns:
            vcc=best_group_tbl["verdict_best"].value_counts()
            lines.append("")
            lines.append("**Best-by-group verdicts:** " + ", ".join([f"{k}: {int(vcc.get(k,0))}" for k in ["better","same","worse"]]))
            lines.append("")

    # Unmatched (brief)
    if onlyA is not None and not onlyA.empty:
        lines.append("## Only in A (not matched in B) — count: " + str(len(onlyA)))
        lines.append("")
    if onlyB is not None and not onlyB.empty:
        lines.append("## Only in B (not matched in A) — count: " + str(len(onlyB)))
        lines.append("")

    # Plots section
    if plots:
        lines.append("## Plots")
        lines.append("")
        if plots.get("delta_histogram_png"):
            lines.append(f"![Δ(B−A) Histogram]({Path(plots['delta_histogram_png']).name})")
            lines.append("")
        if plots.get("scatter_A_vs_B_png"):
            lines.append(f"![Scatter: {metric}_A vs {metric}_B]({Path(plots['scatter_A_vs_B_png']).name})")
            lines.append("")
        if plots.get("grouped_pct_better_png"):
            lines.append(f"![% Better by Group]({Path(plots['grouped_pct_better_png']).name})")
            lines.append("")
        if plots.get("best_by_group_delta_png"):
            lines.append(f"![Best-by-Group Δ(B−A)]({Path(plots['best_by_group_delta_png']).name})")
            lines.append("")

    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description="Compare two sweep summary CSVs and classify B vs A as same/better/worse.")
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ignore", nargs="*", default=["run_id","ckpt_path","best_epoch"])
    ap.add_argument("--metric", default="best_metric")
    ap.add_argument("--tie_tol", type=float, default=0.01)
    ap.add_argument("--higher_is_better", action="store_true")
    ap.add_argument("--group_by", nargs="*", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--hist_edges", nargs="*", type=float, default=[-5,-1,-0.5,-0.2,-0.1,0.1,0.2,0.5,1,5])
    ap.add_argument("--hist_auto_zoom", action="store_true", default=True,
                help="Auto-choose narrow symmetric bounds from delta quantiles")
    ap.add_argument("--hist_bins", type=int, default=20,
                help="Number of bins when using --hist_auto_zoom")

    args=ap.parse_args()

    A_PATH=Path(args.a); B_PATH=Path(args.b); out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    LOWER=not args.higher_is_better; metric=args.metric

    dfa=_load_csv(A_PATH); dfb=_load_csv(B_PATH)
    if metric not in dfa.columns or metric not in dfb.columns:
        raise SystemExit(f"Metric '{metric}' must exist in both CSVs.")

    common=set(dfa.columns) & set(dfb.columns)
    match_keys=sorted(list(common - NEVER_KEYS - set(args.ignore) - {metric}))

    for k in match_keys:
        if dfa[k].dtype==object: dfa[k]=dfa[k].map(_canon_val)
        if dfb[k].dtype==object: dfb[k]=dfb[k].map(_canon_val)

    dfa_d=_dedup_best(dfa, match_keys, metric, LOWER)
    dfb_d=_dedup_best(dfb, match_keys, metric, LOWER)

    comp = dfa_d.merge(dfb_d, on=match_keys, how="inner", suffixes=("_A","_B"))

    meta={"source_A":str(A_PATH),"source_B":str(B_PATH),"tie_tolerance":args.tie_tol,
          "lower_is_better":LOWER,"ignore_cols":sorted(list(set(args.ignore))),"match_keys_used":match_keys,
          "counts":{"A_rows":int(len(dfa)),"B_rows":int(len(dfb)),
                    "A_unique_by_keys":int(len(dfa_d)),"B_unique_by_keys":int(len(dfb_d)),
                    "matched":int(len(comp)),"only_in_A":0,"only_in_B":0}}

    # unmatched
    a_keys = dfa_d[match_keys].assign(_src="A"); b_keys = dfb_d[match_keys].assign(_src="B")
    allk = pd.concat([a_keys,b_keys], ignore_index=True); dupes=allk.duplicated(subset=match_keys, keep=False)
    only = allk.loc[~dupes].copy()
    onlyA = only.loc[only["_src"]=="A"].drop(columns=["_src"]).merge(dfa_d, on=match_keys, how="left")
    onlyB = only.loc[only["_src"]=="B"].drop(columns=["_src"]).merge(dfb_d, on=match_keys, how="left")
    meta["counts"]["only_in_A"]=int(len(onlyA)); meta["counts"]["only_in_B"]=int(len(onlyB))

    plots: Dict[str,str] = {}

    if not comp.empty:
        comp["delta_B_minus_A"] = comp[f"{metric}_B"] - comp[f"{metric}_A"]
        comp["verdict"] = np.where(np.abs(comp["delta_B_minus_A"])<=args.tie_tol, "same",
                            np.where(comp["delta_B_minus_A"]<(0 if LOWER else np.inf), "better", "worse"))
        comp["pct_change_vs_A"]=np.where(comp[f"{metric}_A"].abs()>1e-12, 100.0*comp["delta_B_minus_A"]/comp[f"{metric}_A"].abs(), np.nan)

        # histogram & quantiles + plot
        deltas = comp["delta_B_minus_A"].values
        if args.hist_auto_zoom:
            q05, q95 = np.quantile(deltas, [0.05, 0.95])
            m = float(max(abs(q05), abs(q95)))
            left, right = -1.1*m, 1.1*m        # symmetric, slightly padded
            if left == right:                   # edge case: all deltas same
                left, right = -1.0, 1.0
            edges = np.linspace(left, right, int(max(5, args.hist_bins)) + 1)
        else:
            edges = np.array(list(sorted(set(args.hist_edges))), dtype=float)

        h, bins = np.histogram(deltas, bins=edges)

        delta_hist = pd.DataFrame({"bin_left":bins[:-1],"bin_right":bins[1:], "count":h.astype(int)})
        delta_hist.to_csv(out/"delta_histogram.csv", index=False)
        qs=np.quantile(comp["delta_B_minus_A"].values, [0.05,0.25,0.5,0.75,0.95])
        delta_quant={"q05":float(qs[0]),"q25":float(qs[1]),"q50":float(qs[2]),"q75":float(qs[3]),"q95":float(qs[4])}

        # Plot: histogram
        plt.figure()
        plt.hist(comp["delta_B_minus_A"].values, bins=edges)
        plt.title("Δ(B−A) Histogram")
        plt.xlabel("Δ(B−A)")
        plt.ylabel("Count")
        hist_png = out/"delta_histogram.png"
        plt.savefig(hist_png, bbox_inches="tight"); plt.close()
        plots["delta_histogram_png"] = str(hist_png)

        # Plot: scatter A vs B
        plt.figure()
        x = comp[f"{metric}_A"].values; y = comp[f"{metric}_B"].values
        plt.scatter(x, y, s=8)
        mn = float(min(x.min(), y.min())); mx = float(max(x.max(), y.max()))
        plt.plot([mn, mx], [mn, mx])
        plt.title(f"{metric}_A vs {metric}_B")
        plt.xlabel(f"{metric}_A"); plt.ylabel(f"{metric}_B")
        sc_png = out/"scatter_A_vs_B.png"
        plt.savefig(sc_png, bbox_inches="tight"); plt.close()
        plots["scatter_A_vs_B_png"] = str(sc_png)

    else:
        delta_hist=None; delta_quant=None

    # group cols
    if args.group_by is None:
        group_cols=_detect_group_cols(match_keys)
    elif args.group_by==["none"]:
        group_cols=[]
    else:
        group_cols=[c for c in args.group_by if not comp.empty and c in comp.columns]

    # bests
    def _idx_best(df): 
        if df.empty: return None
        return df[metric].idxmin() if LOWER else df[metric].idxmax()
    bestA_row = dfa_d.loc[_idx_best(dfa_d)] if len(dfa_d) else None
    bestB_row = dfb_d.loc[_idx_best(dfb_d)] if len(dfb_d) else None

    # topK
    topk=max(1,int(args.topk))
    topk_A = dfa_d.sort_values(metric, ascending=LOWER).head(topk).copy() if not dfa_d.empty else None
    topk_B = dfb_d.sort_values(metric, ascending=LOWER).head(topk).copy() if not dfb_d.empty else None
    if topk_A is not None: topk_A.to_csv(out/"topk_A.csv", index=False)
    if topk_B is not None: topk_B.to_csv(out/"topk_B.csv", index=False)

    # improve/regress
    top_improve=top_regress=None
    if not comp.empty:
        top_improve = comp.sort_values("delta_B_minus_A").head(topk).copy()
        top_regress = comp.sort_values("delta_B_minus_A", ascending=False).head(topk).copy()
        top_improve.to_csv(out/"top_improvements.csv", index=False)
        top_regress.to_csv(out/"top_regressions.csv", index=False)

    # grouped summaries + plots
    grouped_summary=None; best_group_tbl=None
    if group_cols and not comp.empty and "verdict" in comp.columns:
        g=comp.groupby(group_cols, dropna=False)
        grouped_summary=g.agg(matched=("verdict","size"),
                              better=("verdict", lambda s:int((s=="better").sum())),
                              same=("verdict", lambda s:int((s=="same").sum())),
                              worse=("verdict", lambda s:int((s=="worse").sum())),
                              delta_mean=("delta_B_minus_A","mean"),
                              delta_median=("delta_B_minus_A","median")).reset_index()
        grouped_summary["pct_better"]=np.where(grouped_summary["matched"]>0, 100.0*grouped_summary["better"]/grouped_summary["matched"], 0.0)
        grouped_summary=grouped_summary.sort_values(["pct_better","delta_mean"], ascending=[False,True])
        grouped_summary.to_csv(out/"grouped_summary.csv", index=False)

        # Plot: pct better by group (bar)
        try:
            labels = grouped_summary.apply(lambda r: _slugify_group(r, group_cols), axis=1)
            plt.figure()
            plt.bar(np.arange(len(labels)), grouped_summary["pct_better"].values)
            plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
            plt.ylabel("% better")
            plt.title("% Better by Group (B vs A)")
            gp_png = out/"grouped_pct_better.png"
            plt.savefig(gp_png, bbox_inches="tight"); plt.close()
            plots["grouped_pct_better_png"] = str(gp_png)
        except Exception:
            pass

    if group_cols:
        try:
            gA = dfa_d.groupby(group_cols, dropna=False)[metric].min().reset_index().rename(columns={metric:f"{metric}_A_best"})
            gB = dfb_d.groupby(group_cols, dropna=False)[metric].min().reset_index().rename(columns={metric:f"{metric}_B_best"})
            gb = pd.merge(gA,gB,on=group_cols,how="inner")
            gb["delta_best_B_minus_A"]=gb[f"{metric}_B_best"]-gb[f"{metric}_A_best"]
            gb["verdict_best"]=np.where(np.abs(gb["delta_best_B_minus_A"])<=args.tie_tol,"same",
                                 np.where(gb["delta_best_B_minus_A"]<(0 if LOWER else np.inf),"better","worse"))
            best_group_tbl=gb.copy(); best_group_tbl.to_csv(out/"best_by_group.csv", index=False)

            # Plot: best-by-group delta bars
            try:
                gb_plot = gb.copy()
                gb_plot["label"] = gb_plot.apply(lambda r: _slugify_group(r, group_cols), axis=1)
                gb_plot = gb_plot.sort_values("delta_best_B_minus_A")
                plt.figure()
                plt.bar(np.arange(len(gb_plot)), gb_plot["delta_best_B_minus_A"].values)
                plt.xticks(np.arange(len(gb_plot)), gb_plot["label"].values, rotation=45, ha="right")
                plt.ylabel("Δ_best (B−A)")
                plt.title("Best-by-Group Δ(B−A)")
                bbg_png = out/"best_by_group_delta.png"
                plt.savefig(bbg_png, bbox_inches="tight"); plt.close()
                plots["best_by_group_delta_png"] = str(bbg_png)
            except Exception:
                pass
        except Exception:
            best_group_tbl=None

    # write MD
    Path(out/"comparison_report.md").write_text(
        _build_md_report(comp, onlyA, onlyB, meta, metric, group_cols,
                         bestA_row, bestB_row, best_group_tbl, grouped_summary,
                         top_improve, top_regress, topk_A, topk_B,
                         delta_hist if not comp.empty else None,
                         delta_quant if not comp.empty else None,
                         plots)
    )

if __name__=="__main__":
    main()
