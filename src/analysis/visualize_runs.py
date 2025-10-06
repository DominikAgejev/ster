from __future__ import annotations
import os, argparse
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe  # for soft shadows

# =========================
# Global options / defaults (trimmed)
# =========================
OPTS: Dict[str, object] = {
    "show_values": False,
    "value_fmt": "{:.2f}",
    "hide_spines": True,
    "hide_yaxis": False,     # never hide y on comparisons (and default overall)
    "hide_xaxis": False,

    # Typography (base)
    "show_titles": True,
    "font_title": 12,
    "font_axis": 11,
    "font_tick": 10,
    "font_value": 10,

    # Combined (features → model clusters → backbone bars)
    "model_gap": 0.06,

    # "Small" slide mode is ON by default
    "slide_small": True,
    "slide_scale": 1.25,
    "slide_value_scale": 1.40,

    # Internal override used by bar-value labels (set per-chart when needed)
    "value_fontsize_override": None,
}

# Always use this feature order when available
FEATURE_ORDER: List[str] = [
    "image",
    "image+mean",
    "image+meta",
    "image+mean+meta",
]

# Warm, readable palette
WARM_PALETTE = [
    "#f2c9a1", "#e5989b", "#d17a22", "#b56576", "#f6bd60",
    "#b55d4c", "#cd9a7b", "#f28482", "#c7a27e", "#6b705c",
]

def warm_color(i: int) -> str:
    return WARM_PALETTE[i % len(WARM_PALETTE)]

# ===============
# Data utilities
# ===============
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "feature" in df.columns and "features" not in df.columns:
        df.rename(columns={"feature": "features"}, inplace=True)
    return df


def ensure_metric(df: pd.DataFrame, metric: str) -> str:
    m = metric.lower()
    if m in df.columns:
        return m
    if "mean" in df.columns:
        return "mean"
    raise ValueError(
        f"Metric '{metric}' not found and 'mean' not available. Columns: {df.columns.tolist()}"
    )


def agg_mean(df: pd.DataFrame, by: List[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(by, dropna=False, as_index=False)[metric].mean().rename(columns={metric: f"{metric}_mean"})
    )


def list_unique(df: pd.DataFrame, col: str) -> List[str]:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        vals = pd.unique(obj.stack().astype(str))
    else:
        vals = obj.dropna().astype(str).unique()
    out = [x for x in vals if x is not None and str(x) != "nan"]
    out.sort()
    return out


def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "features" not in df.columns:
        return df
    # keep only present ones, in canonical order; append any unknowns after
    present = [f for f in FEATURE_ORDER if f in set(df["features"].astype(str))]
    unknown = [f for f in list_unique(df, "features") if f not in present]
    ordered = present + unknown
    df["features"] = pd.Categorical(df["features"], categories=ordered, ordered=True)
    return df


# ===============
# Title helper
# ===============
def _finish(ax, title: str, xlabel: str, ylabel: str, rot: int = 0, tight: bool = True):
    # Title
    if OPTS.get("show_titles", True) and title:
        ax.set_title(title, fontsize=int(OPTS.get("font_title", 12)))
    else:
        ax.set_title("")

    # Spines (borders)
    if OPTS.get("hide_spines", True):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#7f675f")
            ax.spines[s].set_alpha(0.6)
            ax.spines[s].set_linewidth(0.8)

    # Y axis — never hidden by default (esp. for comparisons)
    ax.set_ylabel(ylabel, fontsize=int(OPTS.get("font_axis", 11)))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55, zorder=0)

    # X axis
    ax.set_xlabel(xlabel, fontsize=int(OPTS.get("font_axis", 11)))
    if rot:
        for lab in ax.get_xticklabels():
            lab.set_rotation(rot)
            lab.set_ha("right")

    # Tick styling
    ax.tick_params(axis="y", which="both", direction="out", length=4, color="#7f675f",
                   labelsize=int(OPTS.get("font_tick", 10)))
    ax.tick_params(axis="x", which="both", direction="out", length=4, color="#7f675f",
                   labelsize=int(OPTS.get("font_tick", 10)))

    if tight:
        plt.tight_layout()


def _add_bar_value_labels(ax, bar_containers):
    if not OPTS.get("show_values", False):
        return
    fmt = OPTS.get("value_fmt", "{:.2f}")
    fs = int(OPTS.get("value_fontsize_override") or OPTS.get("font_value", 10))
    for cont in bar_containers:
        patches = cont if hasattr(cont, "__iter__") and hasattr(cont, "patches") is False else cont.patches
        for rect in patches:
            h = rect.get_height()
            if h != h:
                continue
            x = rect.get_x() + rect.get_width() / 2.0
            txt = ax.annotate(
                fmt.format(h),
                xy=(x, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fs,
                color="#2a1f1c",
            )
            txt.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])


def _fmt_feature_label(s: str) -> str:
    s = str(s)
    return s.replace("+", "\n+") if "+" in s else s


def _boost_for_slide(ax):
    if not OPTS.get("slide_small", False):
        return
    scale = float(OPTS.get("slide_scale", 1.25))
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if lab is not None:
            lab.set_fontsize(max(1, int(lab.get_fontsize() * scale)))
    if ax.xaxis and ax.xaxis.label:
        ax.xaxis.label.set_fontsize(max(1, int(ax.xaxis.label.get_fontsize() * scale)))
    if ax.yaxis and ax.yaxis.label:
        ax.yaxis.label.set_fontsize(max(1, int(ax.yaxis.label.get_fontsize() * scale)))
    if ax.title:
        ax.title.set_fontsize(max(1, int(ax.title.get_fontsize() * scale)))


# =========================
# Primitive: simple grouped
# =========================
def bar_by_features_nested(
    ax,
    df: pd.DataFrame,
    primary: str,
    secondary: str,
    value_col: str,
    order_primary: Optional[List[str]] = None,
    order_secondary: Optional[List[str]] = None,
    bar_alpha: float = 1.0,
    title: str = "",
    ylabel: str = "",
    sublabels: bool = False,
):
    prim = order_primary or list_unique(df, primary)
    sec = order_secondary or list_unique(df, secondary)
    n_groups, n_bars = len(prim), len(sec)
    x = np.arange(n_groups, dtype=float)

    total_width = 0.82
    bar_width = total_width / max(1, n_bars)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width

    containers = []
    for j, s in enumerate(sec):
        y = []
        for p in prim:
            row = df[(df[primary].astype(str) == str(p)) & (df[secondary].astype(str) == str(s))]
            y.append(float(row[value_col].values[0]) if len(row) else np.nan)

        bars = ax.bar(
            x + offsets[j],
            y,
            width=bar_width,
            label=str(s),
            color=warm_color(j),
            alpha=bar_alpha,
            edgecolor="#2a1f1c",
            linewidth=0.9,
            zorder=3,
        )
        for b in bars:
            b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
        containers.append(bars)

    feat_labels = [_fmt_feature_label(p) for p in prim]
    ax.set_xticks(x, feat_labels)

    _add_bar_value_labels(ax, containers)

    if sublabels:
        ylo, yhi = ax.get_ylim()
        base = ylo + 0.02 * (yhi - ylo)
        for j, s in enumerate(sec):
            for i, _ in enumerate(prim):
                xpos = x[i] + offsets[j]
                ax.text(
                    xpos,
                    base,
                    str(s),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#5a4a44",
                    rotation=90,
                )

    _finish(ax, title=title, xlabel=primary.title(), ylabel=ylabel, rot=0)


# ==========================================
# Combined chart: features → model clusters → backbone bars
# ==========================================
def bar_by_features_model_grouped(
    ax,
    df: pd.DataFrame,
    primary: str,  # "features"
    model_col: str,  # "model"
    backbone_col: str,  # "backbone"
    value_col: str,  # aggregated mean
    title: str,
    ylabel: str,
):
    prim_vals = list(order for order in list(df[primary].cat.categories) if order in set(df[primary])) \
        if isinstance(df[primary].dtype, pd.CategoricalDtype) else list_unique(df, primary)
    model_vals = list_unique(df, model_col)
    back_vals = list_unique(df, backbone_col)

    n_groups = len(prim_vals)
    x = np.arange(n_groups, dtype=float)

    total_width = 0.88
    gap = float(OPTS.get("model_gap", 0.06))
    M = max(1, len(model_vals))
    total_gaps = gap * (M - 1)
    cluster_width = (total_width - total_gaps) / M
    bar_width = cluster_width / max(1, len(back_vals)) * 0.9

    cluster_offsets = []
    left_edge = -total_width / 2.0
    for m_idx in range(M):
        c_left = left_edge + m_idx * (cluster_width + gap)
        c_center = c_left + cluster_width / 2.0
        cluster_offsets.append(c_center)

    containers = []
    bar_positions: List[float] = []
    bar_backbone_labels: List[str] = []
    cluster_centers_by_group: List[List[float]] = []

    for i, p in enumerate(prim_vals):
        this_centers = []
        for m_idx, m in enumerate(model_vals):
            c_off = cluster_offsets[m_idx]
            group_center = x[i]
            cluster_center_x = group_center + c_off
            this_centers.append(cluster_center_x)

            for b_idx, bname in enumerate(back_vals):
                row = df[
                    (df[primary].astype(str) == str(p))
                    & (df[model_col].astype(str) == str(m))
                    & (df[backbone_col].astype(str) == str(bname))
                ]
                yval = float(row[value_col].values[0]) if len(row) else np.nan

                inner_offset = ((b_idx + 0.5) / max(1, len(back_vals)) - 0.5) * cluster_width
                bar_x = cluster_center_x + inner_offset

                bar = ax.bar(
                    [bar_x],
                    [yval],
                    width=bar_width,
                    color=warm_color(b_idx),
                    edgecolor="#2a1f1c",
                    linewidth=0.9,
                    zorder=3,
                )
                for bpatch in bar:
                    bpatch.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
                containers.append(bar)

                bar_positions.append(bar_x)
                bar_backbone_labels.append(str(bname))
        cluster_centers_by_group.append(this_centers)

    # Axis tick labels are drawn manually in stacked rows
    ax.set_xticks(x, [""] * len(x))

    _add_bar_value_labels(ax, containers)

    trans = ax.get_xaxis_transform()  # x in data, y in axes coords
    y_model = -0.02
    y_backbone = y_model - 0.03
    y_feature = y_backbone - 0.03

    # MODEL labels at cluster centers
    for centers in cluster_centers_by_group:
        for m_idx, cx in enumerate(centers):
            ax.text(cx, y_model, str(model_vals[m_idx]), transform=trans, ha="center", va="top",
                    fontsize=9, color="#2a1f1c", clip_on=False)

    # BACKBONE labels at each bar position
    for bx, lab in zip(bar_positions, bar_backbone_labels):
        ax.text(bx, y_backbone, lab, transform=trans, ha="center", va="top",
                fontsize=8, color="#5a4a44", clip_on=False)

    # FEATURE names centered at group centers
    for i, p in enumerate(prim_vals):
        ax.text(x[i], y_feature, str(p), transform=trans, ha="center", va="top",
                fontsize=9, color="#2a1f1c", clip_on=False)

    plt.gcf().subplots_adjust(bottom=0.20)

    _finish(ax, title=title, xlabel=primary.title(), ylabel=ylabel, rot=0)


# =======================
# Feature comparisons & BG/No-BG comparisons
# =======================
def metric_suffix(metric: str) -> str:
    return "" if metric == "mean" else f"_{metric}"


def plot_feature_mean_bar(df: pd.DataFrame, metric: str, outdir: str):
    m = ensure_metric(df, metric)
    df = apply_feature_order(df)
    agg = df.groupby("features", as_index=False)[m].agg(mean="mean", std="std")

    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    cap = 5 if OPTS.get("slide_small", False) else 3
    bars = ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"],
        capsize=cap,
        color=[warm_color(i) for i in range(len(agg))],
        edgecolor="#2a1f1c",
        linewidth=0.9,
        zorder=3,
    )
    for b in bars:
        b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])

    feat_labels = [_fmt_feature_label(v) for v in agg["features"]]
    ax.set_xticks(x, feat_labels)

    old_override = OPTS.get("value_fontsize_override")
    if OPTS.get("slide_small", False):
        OPTS["value_fontsize_override"] = int(
            OPTS.get("font_value", 10) * float(OPTS.get("slide_value_scale", 1.4))
        )
    _add_bar_value_labels(ax, [bars])
    OPTS["value_fontsize_override"] = old_override

    _finish(ax, title=f"Average ± std by features ({m})", xlabel="Features", ylabel=m, rot=0)
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"feature_mean_bar{metric_suffix(m)}.png"))
    plt.close(fig)


def plot_feature_pair_scatters(df: pd.DataFrame, metric: str, outdir: str, max_pairs: int = 6):
    m = ensure_metric(df, metric)
    pivot = (
        df.groupby(["model", "backbone", "features"], as_index=False)[m]
        .mean()
        .pivot(index=["model", "backbone"], columns="features", values=m)
    )
    feats = [str(c) for c in pivot.columns]
    pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            pairs.append((feats[i], feats[j]))
    pairs = pairs[: max_pairs]

    for (fa, fb) in pairs:
        sub = pivot[[fa, fb]].dropna()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        models = sorted({idx[0] for idx in sub.index})
        markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<"]
        size_base = 60
        if OPTS.get("slide_small", False):
            size_base = int(size_base * float(OPTS.get("slide_scale", 1.25)))
        for mi, model in enumerate(models):
            rows = sub.loc[(model, slice(None)), :]
            ax.scatter(rows[fa], rows[fb], s=size_base, color=warm_color(mi), marker=markers[mi % len(markers)],
                       alpha=0.9, label=str(model))
        both = pd.concat([sub[fa], sub[fb]], axis=0)
        lo, hi = float(both.min()), float(both.max())
        pad = 0.05 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _finish(ax, title=f"{fa} vs {fb} ({m})", xlabel=str(fa), ylabel=str(fb), rot=0)
        _boost_for_slide(ax)
        fig.savefig(os.path.join(outdir, f"feature_scatter_{fa}_vs_{fb}{metric_suffix(m)}.png"))
        plt.close(fig)


def align_bg_nobg(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str) -> pd.DataFrame:
    m_bg = ensure_metric(df_bg, metric)
    m_nb = ensure_metric(df_nbg, metric)
    a_bg = agg_mean(norm_cols(df_bg), ["model", "backbone", "features"], m_bg).rename(
        columns={f"{m_bg}_mean": "metric_bg"}
    )
    a_nb = agg_mean(norm_cols(df_nbg), ["model", "backbone", "features"], m_nb).rename(
        columns={f"{m_nb}_mean": "metric_nbg"}
    )
    m = pd.merge(a_bg, a_nb, on=["model", "backbone", "features"], how="inner")
    # Enforce feature ordering here too
    m = apply_feature_order(m)
    return m


def plot_bg_vs_nobg_bars(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str, outdir: str):
    m = align_bg_nobg(df_bg, df_nbg, metric)
    agg = m.groupby("features", as_index=False)[["metric_bg", "metric_nbg"]].mean()
    x = np.arange(len(agg))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars1 = ax.bar(
        x - width / 2,
        agg["metric_bg"],
        width=width,
        label="BG",
        color=warm_color(1),
        edgecolor="#2a1f1c",
        linewidth=0.9,
        zorder=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        agg["metric_nbg"],
        width=width,
        label="No-BG",
        color=warm_color(6),
        edgecolor="#2a1f1c",
        linewidth=0.9,
        zorder=3,
    )
    for cont in (bars1, bars2):
        for b in cont:
            b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])

    feat_labels = [_fmt_feature_label(v) for v in agg["features"]]
    ax.set_xticks(x, feat_labels)

    old_override = OPTS.get("value_fontsize_override")
    if OPTS.get("slide_small", False):
        OPTS["value_fontsize_override"] = int(
            OPTS.get("font_value", 10) * float(OPTS.get("slide_value_scale", 1.4))
        )
    _add_bar_value_labels(ax, [bars1, bars2])
    OPTS["value_fontsize_override"] = old_override

    _finish(ax, title=f"BG vs No-BG by features ({ensure_metric(df_bg, metric)})", xlabel="Features", ylabel=metric, rot=0)
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"bg_vs_nobg_by_features{metric_suffix(ensure_metric(df_bg, metric))}.png"))
    plt.close(fig)


def plot_bg_vs_nobg_scatter(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str, outdir: str):
    m = align_bg_nobg(df_bg, df_nbg, metric)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    feats = [f for f in FEATURE_ORDER if f in set(m["features"].astype(str))]
    for i, f in enumerate(feats):
        sub = m[m["features"].astype(str) == str(f)]
        size = 120 if OPTS.get("slide_small", False) else 90
        ax.scatter(sub["metric_bg"], sub["metric_nbg"], s=size, color=warm_color(i), alpha=0.9, label=str(f))
    allv = pd.concat([m["metric_bg"], m["metric_nbg"]], axis=0)
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.05 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _finish(ax, title=f"BG (x) vs No-BG (y) — {ensure_metric(df_bg, metric)}", xlabel="BG", ylabel="No-BG", rot=0)
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"bg_vs_nobg_scatter{metric_suffix(ensure_metric(df_bg, metric))}.png"))
    plt.close(fig)


def plot_delta_bars(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str, outdir: str):
    m = align_bg_nobg(df_bg, df_nbg, metric)
    m["delta"] = m["metric_bg"] - m["metric_nbg"]  # BG − No-BG
    agg = m.groupby("features", as_index=False)["delta"].mean()
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    colors = [warm_color(i) for i in range(len(agg))]
    bars = ax.bar(x, agg["delta"], color=colors, alpha=0.95, edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
    for b in bars:
        b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
    ax.axhline(0.0, color="#555555", linewidth=1)
    feat_labels = [_fmt_feature_label(v) for v in agg["features"]]
    ax.set_xticks(x, feat_labels)

    old_override = OPTS.get("value_fontsize_override")
    if OPTS.get("slide_small", False):
        OPTS["value_fontsize_override"] = int(
            OPTS.get("font_value", 10) * float(OPTS.get("slide_value_scale", 1.4))
        )
    _add_bar_value_labels(ax, [bars])
    OPTS["value_fontsize_override"] = old_override

    _finish(ax, title=f"delta = BG - No-BG ({ensure_metric(df_bg, metric)})", xlabel="Features", ylabel="delta", rot=0)
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"delta_bg_minus_nobg_by_features{metric_suffix(ensure_metric(df_bg, metric))}.png"))
    plt.close(fig)


# =======================
# Figure builder functions (single CSV)
# =======================
def plot_per_model_bars(df: pd.DataFrame, metric: str, outdir: str):
    df = apply_feature_order(df)
    m = ensure_metric(df, metric)
    models = list_unique(df, "model")
    for model in models:
        sub = df[df["model"].astype(str) == str(model)].copy()
        agg = agg_mean(sub, by=["features", "backbone"], metric=m)
        # keep feature order
        ordered = [f for f in FEATURE_ORDER if f in set(agg["features"].astype(str))]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        bar_by_features_nested(
            ax,
            agg,
            primary="features",
            secondary="backbone",
            value_col=f"{m}_mean",
            order_primary=ordered,
            title=f"Model {model} — {m}",
            ylabel=m,
            sublabels=False,
        )
        fig.savefig(os.path.join(outdir, f"per_model_{model}{metric_suffix(m)}.png"))
        plt.close(fig)

def plot_backbone_first_bars_per_backbone(df: pd.DataFrame, metric: str, outdir: str):
    """Render one figure per backbone (two variants for clarity)."""
    df = apply_feature_order(df)
    m = ensure_metric(df, metric)
    backbones = list_unique(df, "backbone")

    for bb in backbones:
        sub = df[df["backbone"].astype(str) == str(bb)].copy()
        # 1) feature clusters → model bars
        agg1 = agg_mean(sub, by=["features", "model"], metric=m)
        fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
        bar_by_features_nested(
            ax1,
            agg1,
            primary="features",
            secondary="model",
            value_col=f"{m}_mean",
            order_primary=[f for f in FEATURE_ORDER if f in set(agg1["features"].astype(str))],
            title=f"{bb} — features × models ({m})",
            ylabel=m,
            sublabels=False,
        )
        fig1.savefig(os.path.join(outdir, f"{bb}_features_models{metric_suffix(m)}.png"))
        plt.close(fig1)

        # 2) model clusters → feature bars
        agg2 = agg_mean(sub, by=["model", "features"], metric=m)
        models = list_unique(agg2, "model")
        x = np.arange(len(models), dtype=float)
        # Build grouped bars where within each model we draw features in canonical order
        fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
        total_width = 0.82
        feats = [f for f in FEATURE_ORDER if f in set(agg2["features"].astype(str))]
        n_bars = len(feats)
        bar_width = total_width / max(1, n_bars)
        offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width
        containers = []
        for j, f in enumerate(feats):
            y = []
            for mdl in models:
                row = agg2[(agg2["model"].astype(str) == str(mdl)) & (agg2["features"].astype(str) == str(f))]
                y.append(float(row[f"{m}_mean"].values[0]) if len(row) else np.nan)
            bars = ax2.bar(
                x + offsets[j],
                y,
                width=bar_width,
                label=str(f),
                color=warm_color(j),
                edgecolor="#2a1f1c",
                linewidth=0.9,
                zorder=3,
            )
            for b in bars:
                b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
            containers.append(bars)
        ax2.set_xticks(x, models)
        _add_bar_value_labels(ax2, containers)
        _finish(ax2, title=f"{bb} — models × features ({m})", xlabel="model", ylabel=m, rot=0)
        fig2.savefig(os.path.join(outdir, f"{bb}_models_features{metric_suffix(m)}.png"))
        plt.close(fig2)


# =========
# Main CLI (simplified)
# =========
def main():
    ap = argparse.ArgumentParser(description="Visualize collected test CSVs (single or BG vs No-BG).")
    ap.add_argument("--csv", required=True, help="Path to main CSV.")
    ap.add_argument("--csv2", default=None, help="Optional second CSV (for BG vs No-BG).")
    ap.add_argument("--metric", default="mean", help="Metric column to use (default: mean).")
    ap.add_argument("--outdir", default="./viz_out", help="Output directory root for figures.")
    ap.add_argument("--show-values", action="store_true", help="Write values on bars.")
    ap.add_argument("--value-fmt", default="{:.2f}", help="Format for bar value labels.")
    args = ap.parse_args()

    # Global figure defaults
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["figure.dpi"] = 150

    # Wire trivial options
    OPTS["show_values"] = args.show_values
    OPTS["value_fmt"] = args.value_fmt

    # Load and normalize
    df1 = norm_cols(pd.read_csv(args.csv))
    if "model" in df1.columns:
        df1["model"] = df1["model"].replace({"late": "fusion"})  # immediate rename

    if args.csv2:
        df2 = norm_cols(pd.read_csv(args.csv2))
        if "model" in df2.columns:
            df2["model"] = df2["model"].replace({"late": "fusion"})
    else:
        df2 = None

    # Output folder structure
    if df2 is not None:
        root = args.outdir
        out_csv1 = os.path.join(root, "csv1")
        out_csv2 = os.path.join(root, "csv2")
        out_cmp  = os.path.join(root, "compare")
        for p in (root, out_csv1, out_csv2, out_cmp):
            mkdir_p(p)
    else:
        out_csv1 = args.outdir
        mkdir_p(out_csv1)

    # ---- Single-CSV visuals ----
    plot_per_model_bars(df1, args.metric, out_csv1)
    plot_feature_mean_bar(df1, args.metric, out_csv1)
    plot_feature_pair_scatters(df1, args.metric, out_csv1, max_pairs=8)
    plot_backbone_first_bars_per_backbone(df1, args.metric, out_csv1)

    if df2 is not None:
        plot_per_model_bars(df2, args.metric, out_csv2)
        plot_feature_mean_bar(df2, args.metric, out_csv2)
        plot_feature_pair_scatters(df2, args.metric, out_csv2, max_pairs=8)
        plot_backbone_first_bars_per_backbone(df2, args.metric, out_csv2)

        # ---- Comparisons (never hide y, dots size 120 in slide-small) ----
        plot_bg_vs_nobg_bars(df1, df2, args.metric, out_cmp)
        plot_bg_vs_nobg_scatter(df1, df2, args.metric, out_cmp)
        plot_delta_bars(df1, df2, args.metric, out_cmp)

    print(f"[viz] Saved figures under: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
