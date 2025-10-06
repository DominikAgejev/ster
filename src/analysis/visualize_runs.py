# src/analysis/visualize_runs.py
from __future__ import annotations
import os, argparse
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe  # for soft shadows


# =========================
# Global options / defaults
# =========================
OPTS: Dict[str, object] = {
    "show_values": False,
    "value_fmt": "{:.2f}",
    "hide_spines": True,
    "bar_sublabels": False,      # used by simple grouped bars
    "hide_yaxis": False,
    "hide_xaxis": False,
    "show_legend": False,        # legend OFF by default
    "title": None,

    # Typography (base)
    "font_title": 12,
    "font_axis": 11,
    "font_tick": 10,
    "font_value": 10,            # larger default for legible numbers

    # Coloring
    "color_by": "backbone",      # backbone | features | model

    # Combined (features → model clusters → backbone bars)
    "model_gap": 0.06,           # padding between model clusters inside each features group
    "backbone_sublabels": True,  # show backbone under each bar
    "model_label_offset": -0.045,# how far below axis to draw model labels (axes coords, negative = below)
    "backbone_label_offset": None,  # axes y offset for backbone row (if None -> derived from model_label_offset)
    "feature_label_offset":  None,  # axes y offset for feature row (if None -> derived, placed below backbones)
    "combined_flat": False,      # fallback to old flat "model-backbone" bars

    # Titles
    "show_titles": True,          # allow disabling titles globally

    # Feature pair scatter sizing
    "feature_scatter_size": 60,   # marker size for feature pair scatters

    # Slide-friendly boosts (applied to comparison charts only)
    "slide_small": False,
    "slide_scale": 1.25,
    "slide_value_scale": 1.40,

    # Internal override used by bar-value labels (set per-chart when needed)
    "value_fontsize_override": None,
}

# Warm, skin-tone friendly palette with more contrast
WARM_PALETTE = [
    "#f2c9a1",  # peach
    "#e5989b",  # dusty rose
    "#d17a22",  # ochre
    "#b56576",  # mauve
    "#f6bd60",  # warm sand
    "#b55d4c",  # clay
    "#cd9a7b",  # light brown
    "#f28482",  # coral
    "#c7a27e",  # brown sugar
    "#6b705c",  # olive gray (anchor)
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
    raise ValueError(f"Metric '{metric}' not found and 'mean' not available. Columns: {df.columns.tolist()}")

def agg_mean(df: pd.DataFrame, by: List[str], metric: str) -> pd.DataFrame:
    return (df.groupby(by, dropna=False, as_index=False)[metric]
              .mean()
              .rename(columns={metric: f"{metric}_mean"}))

def list_unique(df: pd.DataFrame, col: str) -> List[str]:
    """Return sorted uniques even if duplicate column names exist."""
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

def filter_iphone_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that mention 'iphone' (case-insensitive) in any of several likely columns."""
    if df is None or df.empty:
        return df
    cols = [c for c in ["features", "id", "dataset", "device", "source"] if c in df.columns]
    if not cols:
        # Fallback: try features if present (norm_cols usually ensures 'features')
        if "features" in df.columns:
            return df[df["features"].astype(str).str.contains("iphone", case=False, na=False)].copy()
        return df  # nothing to filter on, return as-is
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        mask |= df[c].astype(str).str.contains("iphone", case=False, na=False)
    return df[mask].copy()

# ===============
# Title helper
# ===============
def _title(default: str, **kwargs) -> str:
    t = OPTS.get("title")
    if not t:
        return default
    try:
        return str(t).format(**kwargs)
    except Exception:
        return str(t)


# =========================
# Axes / layout helpers
# =========================
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

    # Y axis
    if OPTS.get("hide_yaxis", False):
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.grid(False, axis="y")
    else:
        ax.set_ylabel(ylabel, fontsize=int(OPTS.get("font_axis", 11)))
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55, zorder=0)

    # X axis
    if OPTS.get("hide_xaxis", False):
        ax.set_xlabel("")
        ax.set_xticks([])
    else:
        ax.set_xlabel(xlabel, fontsize=int(OPTS.get("font_axis", 11)))
        if rot:
            for lab in ax.get_xticklabels():
                lab.set_rotation(rot)
                lab.set_ha("right")

    # Tick styling
    if not OPTS.get("hide_yaxis", False):
        ax.tick_params(axis="y", which="both", direction="out",
                       length=4, color="#7f675f",
                       labelsize=int(OPTS.get("font_tick", 10)))
    if not OPTS.get("hide_xaxis", False):
        ax.tick_params(axis="x", which="both", direction="out",
                       length=4, color="#7f675f",
                       labelsize=int(OPTS.get("font_tick", 10)))

    if OPTS.get("show_legend", False):
        ax.legend(frameon=False)

    if tight:
        plt.tight_layout()


def _add_bar_value_labels(ax, bar_containers):
    """Write numeric values on top of bars."""
    if not OPTS.get("show_values", False):
        return
    fmt = OPTS.get("value_fmt", "{:.2f}")
    fs = int(OPTS.get("value_fontsize_override") or OPTS.get("font_value", 10))
    for cont in bar_containers:
        # Works for BarContainer and for a single returned Bar from ax.bar([...])
        patches = cont if hasattr(cont, "__iter__") and hasattr(cont, "patches") is False else cont.patches
        for rect in patches:
            h = rect.get_height()
            if h != h:  # NaN
                continue
            x = rect.get_x() + rect.get_width() / 2.0
            txt = ax.annotate(fmt.format(h),
                              xy=(x, h), xytext=(0, 3),
                              textcoords="offset points",
                              ha="center", va="bottom",
                              fontsize=fs, color="#2a1f1c")
            # white stroke to pop on colored bars
            txt.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])


def _fmt_feature_label(s: str) -> str:
    """Center-friendly feature label. Split to a new line before each '+'."""
    s = str(s)
    return s.replace('+', '\n+') if '+' in s else s

def _choose_color_index(feature_idx: Optional[int], model_idx: Optional[int], backbone_idx: Optional[int]) -> int:
    scheme = str(OPTS.get("color_by", "backbone")).lower()
    if scheme == "features" and feature_idx is not None:
        return feature_idx
    if scheme == "model" and model_idx is not None:
        return model_idx
    # default backbone
    return backbone_idx if backbone_idx is not None else 0

def _boost_for_slide(ax):
    """Scale fonts for small-slide readability (used on comparison charts)."""
    if not OPTS.get("slide_small", False):
        return
    scale = float(OPTS.get("slide_scale", 1.25))
    # Tick labels
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if lab is not None:
            lab.set_fontsize(max(1, int(lab.get_fontsize() * scale)))
    # Axis labels
    if ax.xaxis and ax.xaxis.label:
        ax.xaxis.label.set_fontsize(max(1, int(ax.xaxis.label.get_fontsize() * scale)))
    if ax.yaxis and ax.yaxis.label:
        ax.yaxis.label.set_fontsize(max(1, int(ax.yaxis.label.get_fontsize() * scale)))
    # Title
    if ax.title:
        ax.title.set_fontsize(max(1, int(ax.title.get_fontsize() * scale)))


# =========================
# Primitive: simple grouped
# =========================
def bar_by_features_nested(ax, df: pd.DataFrame, primary: str, secondary: str, value_col: str,
                           order_primary: Optional[List[str]] = None,
                           order_secondary: Optional[List[str]] = None,
                           bar_alpha: float = 1.0,
                           title: str = "", ylabel: str = "", sublabels: bool = False):
    """Grouped bars: x groups = primary (features); within-group bars = secondary."""
    prim = order_primary or list_unique(df, primary)
    sec  = order_secondary or list_unique(df, secondary)
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
            x + offsets[j], y, width=bar_width, label=str(s),
            color=warm_color(j), alpha=bar_alpha, edgecolor="#2a1f1c", linewidth=0.9, zorder=3
        )
        for b in bars:
            b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
        containers.append(bars)

    # Centered, multi-line feature labels (no rotation)
    feat_labels = [_fmt_feature_label(p) for p in prim]
    ax.set_xticks(x, feat_labels)

    _add_bar_value_labels(ax, containers)
    if sublabels:
        # sublabels for simple grouped plots use secondary labels (backbone) under the bars
        ylo, yhi = ax.get_ylim()
        base = ylo + 0.02 * (yhi - ylo)
        for j, s in enumerate(sec):
            for i, _ in enumerate(prim):
                xpos = x[i] + offsets[j]
                ax.text(xpos, base, str(s), ha="center", va="bottom",
                        fontsize=8, color="#5a4a44", rotation=90)

    _finish(ax, title=title, xlabel=primary.title(), ylabel=ylabel, rot=0)


# ==========================================
# Combined chart: features → model clusters → backbone bars
# ==========================================
def bar_by_features_model_grouped(ax,
                                  df: pd.DataFrame,
                                  primary: str,           # "features"
                                  model_col: str,         # "model"
                                  backbone_col: str,      # "backbone"
                                  value_col: str,         # aggregated mean
                                  title: str,
                                  ylabel: str):
    """
    Within each features group:
      - split into MODEL clusters with a tiny gap between them,
      - inside each model cluster, one bar per BACKBONE,
      - draw MODEL label below the horizontal axis (outside), above feature name,
      - write BACKBONE as sublabel under each bar (not rotated),
      - draw FEATURE single-line name at bottom.
    """
    prim_vals = list_unique(df, primary)
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

    # cluster centers relative to group center
    cluster_offsets = []
    left_edge = -total_width / 2.0
    for m_idx in range(M):
        c_left = left_edge + m_idx * (cluster_width + gap)
        c_center = c_left + cluster_width / 2.0
        cluster_offsets.append(c_center)

    # plotting
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
                # value
                row = df[
                    (df[primary].astype(str) == str(p)) &
                    (df[model_col].astype(str) == str(m)) &
                    (df[backbone_col].astype(str) == str(bname))
                ]
                yval = float(row[value_col].values[0]) if len(row) else np.nan

                # bar position
                inner_offset = ((b_idx + 0.5) / max(1, len(back_vals)) - 0.5) * cluster_width
                bar_x = cluster_center_x + inner_offset

                # color scheme
                color_idx = _choose_color_index(feature_idx=i, model_idx=m_idx, backbone_idx=b_idx)

                bar = ax.bar(
                    [bar_x], [yval], width=bar_width,
                    color=warm_color(color_idx), edgecolor="#2a1f1c", linewidth=0.9, zorder=3
                )
                for bpatch in bar:
                    bpatch.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
                containers.append(bar)

                bar_positions.append(bar_x)
                bar_backbone_labels.append(str(bname))
        cluster_centers_by_group.append(this_centers)

    # ----- axis annotations under the bars -----
    # We control three rows under the axis: MODEL (top), BACKBONE (middle), FEATURE (bottom).
    ax.set_xticks(x, [""] * len(x))  # blank ticks; we draw our label stack

    # write numeric labels on bars (above bars, inside axes)
    _add_bar_value_labels(ax, containers)

    trans = ax.get_xaxis_transform()  # x in data, y in axes coords
    y_model = float(OPTS.get("model_label_offset", -0.02))  # closer to axis
    # Derive backbone/feature rows if not set explicitly
    y_backbone = OPTS.get("backbone_label_offset")
    y_feature  = OPTS.get("feature_label_offset")
    if y_backbone is None:
        y_backbone = y_model - 0.03
    if y_feature is None:
        y_feature = y_backbone - 0.03

    # MODEL labels at cluster centers (top row, dark)
    for centers in cluster_centers_by_group:
        for m_idx, cx in enumerate(centers):
            ax.text(cx, y_model, str(model_vals[m_idx]),
                    transform=trans, ha="center", va="top",
                    fontsize=9, color="#2a1f1c", clip_on=False)

    # BACKBONE labels at each bar position (middle row, not rotated)
    for bx, lab in zip(bar_positions, bar_backbone_labels):
        ax.text(bx, y_backbone, lab, transform=trans, ha="center", va="top",
                fontsize=8, color="#5a4a44", clip_on=False)

    # FEATURE names centered at group centers (bottom row, single line)
    for i, p in enumerate(prim_vals):
        ax.text(x[i], y_feature, str(p), transform=trans, ha="center", va="top",
                fontsize=9, color="#2a1f1c", clip_on=False)

    # Add bottom margin so three rows fit (scale with deepest row)
    deepest = min(y_model, y_backbone, y_feature)
    plt.gcf().subplots_adjust(bottom=max(0.18, 0.12 + abs(deepest) * 2.2))

    _finish(ax, title=title, xlabel=primary.title(), ylabel=ylabel, rot=0)


# ==========================================
# Backbone-first charts (two variants)
# ==========================================
def bar_backbone_first_features_clusters_model_bars(ax,
                                                    df: pd.DataFrame,
                                                    primary: str,        # "backbone"
                                                    cluster_col: str,    # "features"
                                                    inner_col: str,      # "model"
                                                    value_col: str,
                                                    title: str, ylabel: str):
    """
    Backbone → feature clusters → model bars.
    Cluster labels (features) are below the axis; bars are colored by --color-by.
    """
    prim_vals = list_unique(df, primary)
    cl_vals = list_unique(df, cluster_col)
    in_vals = list_unique(df, inner_col)

    n_groups = len(prim_vals)
    x = np.arange(n_groups, dtype=float)

    total_width = 0.88
    gap = 0.06
    C = max(1, len(cl_vals))
    total_gaps = gap * (C - 1)
    cluster_width = (total_width - total_gaps) / C
    bar_width = cluster_width / max(1, len(in_vals)) * 0.9

    cluster_offsets = []
    left_edge = -total_width / 2.0
    for c_idx in range(C):
        c_left = left_edge + c_idx * (cluster_width + gap)
        c_center = c_left + cluster_width / 2.0
        cluster_offsets.append(c_center)

    containers = []
    cluster_centers_by_group: List[List[float]] = []

    for i, p in enumerate(prim_vals):
        this_centers = []
        for c_idx, c in enumerate(cl_vals):
            c_off = cluster_offsets[c_idx]
            group_center = x[i]
            cluster_center_x = group_center + c_off
            this_centers.append(cluster_center_x)

            for in_idx, in_name in enumerate(in_vals):
                row = df[
                    (df[primary].astype(str) == str(p)) &
                    (df[cluster_col].astype(str) == str(c)) &
                    (df[inner_col].astype(str) == str(in_name))
                ]
                yval = float(row[value_col].values[0]) if len(row) else np.nan
                inner_offset = ( (in_idx + 0.5) / max(1, len(in_vals)) - 0.5 ) * cluster_width
                bar_x = cluster_center_x + inner_offset

                color_idx = _choose_color_index(
                    feature_idx=c_idx if cluster_col == "features" else None,
                    model_idx=in_idx if inner_col == "model" else None,
                    backbone_idx=i   if primary == "backbone" else None
                )
                bar = ax.bar([bar_x], [yval], width=bar_width,
                             color=warm_color(color_idx), edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
                for bpatch in bar:
                    bpatch.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
                containers.append(bar)
        cluster_centers_by_group.append(this_centers)

    ax.set_xticks(x, prim_vals)
    _add_bar_value_labels(ax, containers)

    # cluster labels (features) under axis
    trans = ax.get_xaxis_transform()
    yoff = -0.045
    for centers in cluster_centers_by_group:
        for c_idx, cx in enumerate(centers):
            ax.text(cx, yoff, _fmt_feature_label(cl_vals[c_idx]),
                    transform=trans, ha="center", va="top",
                    fontsize=9, color="#2a1f1c", clip_on=False)
    plt.gcf().subplots_adjust(bottom=0.18)

    _finish(ax, title=title, xlabel=primary.title(), ylabel=ylabel, rot=0)


def bar_backbone_first_model_clusters_feature_bars(ax,
                                                   df: pd.DataFrame,
                                                   primary: str,        # "backbone"
                                                   cluster_col: str,    # "model"
                                                   inner_col: str,      # "features"
                                                   value_col: str,
                                                   title: str, ylabel: str):
    """
    Backbone → model clusters → feature bars.
    Cluster labels (models) under axis; bars colored by --color-by.
    """
    prim_vals = list_unique(df, primary)
    cl_vals = list_unique(df, cluster_col)
    in_vals = list_unique(df, inner_col)

    n_groups = len(prim_vals)
    x = np.arange(n_groups, dtype=float)

    total_width = 0.88
    gap = 0.06
    C = max(1, len(cl_vals))
    total_gaps = gap * (C - 1)
    cluster_width = (total_width - total_gaps) / C
    bar_width = cluster_width / max(1, len(in_vals)) * 0.9

    cluster_offsets = []
    left_edge = -total_width / 2.0
    for c_idx in range(C):
        c_left = left_edge + c_idx * (cluster_width + gap)
        c_center = c_left + cluster_width / 2.0
        cluster_offsets.append(c_center)

    containers = []
    cluster_centers_by_group: List[List[float]] = []

    for i, p in enumerate(prim_vals):
        this_centers = []
        for c_idx, c in enumerate(cl_vals):
            c_off = cluster_offsets[c_idx]
            group_center = x[i]
            cluster_center_x = group_center + c_off
            this_centers.append(cluster_center_x)

            for in_idx, in_name in enumerate(in_vals):
                row = df[
                    (df[primary].astype(str) == str(p)) &
                    (df[cluster_col].astype(str) == str(c)) &
                    (df[inner_col].astype(str) == str(in_name))
                ]
                yval = float(row[value_col].values[0]) if len(row) else np.nan
                inner_offset = ( (in_idx + 0.5) / max(1, len(in_vals)) - 0.5 ) * cluster_width
                bar_x = cluster_center_x + inner_offset

                # color choice
                color_idx = _choose_color_index(
                    feature_idx=in_idx if inner_col == "features" else None,
                    model_idx=c_idx  if cluster_col == "model" else None,
                    backbone_idx=i   if primary == "backbone" else None
                )
                bar = ax.bar([bar_x], [yval], width=bar_width,
                             color=warm_color(color_idx), edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
                for bpatch in bar:
                    bpatch.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
                containers.append(bar)
        cluster_centers_by_group.append(this_centers)

    ax.set_xticks(x, prim_vals)
    _add_bar_value_labels(ax, containers)

    # cluster labels (models) under axis
    trans = ax.get_xaxis_transform()
    yoff = -0.045
    for centers in cluster_centers_by_group:
        for c_idx, cx in enumerate(centers):
            ax.text(cx, yoff, str(cl_vals[c_idx]),
                    transform=trans, ha="center", va="top",
                    fontsize=9, color="#2a1f1c", clip_on=False)
    plt.gcf().subplots_adjust(bottom=0.18)

    _finish(ax, title=title, xlabel=primary.title(), ylabel=ylabel, rot=20)


# =======================
# Feature comparisons & BG/No-BG comparisons
# =======================
def plot_feature_mean_bar(df: pd.DataFrame, metric: str, outdir: str):
    """Overall mean (and std as errorbar) per feature across models/backbones."""
    m = ensure_metric(df, metric)
    # Named agg avoids MultiIndex columns and the 'index' column from reset_index()
    agg = df.groupby("features", as_index=False)[m].agg(mean="mean", std="std")

    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    cap = 5 if OPTS.get("slide_small", False) else 3
    bars = ax.bar(x, agg["mean"], yerr=agg["std"], capsize=cap,
                  color=[warm_color(i) for i in range(len(agg))],
                  edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
    for b in bars:
        b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])

    # Centered, multi-line labels for features (split before '+')
    feat_labels = [_fmt_feature_label(v) for v in agg["features"]]
    ax.set_xticks(x, feat_labels)

    # Slide-mode: enlarge numbers relative to bars
    old_override = OPTS.get("value_fontsize_override")
    if OPTS.get("slide_small", False):
        OPTS["value_fontsize_override"] = int(OPTS.get("font_value", 10) * float(OPTS.get("slide_value_scale", 1.4)))
    _add_bar_value_labels(ax, [bars])
    OPTS["value_fontsize_override"] = old_override

    _finish(ax, title=_title(f"Feature comparison — mean±std ({m})",
                             chart="feature_mean_bar", model="ALL", metric=m),
            xlabel="Features", ylabel=m, rot=0)

    # Boost readability if going small on slides
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"feature_mean_bar_{m}.png"))
    plt.close(fig)


def plot_feature_pair_scatters(df: pd.DataFrame, metric: str, outdir: str, max_pairs: int = 6):
    """
    Pairwise feature-vs-feature scatter (matched on model/backbone).
    Limits to 'max_pairs' pairs to avoid explosion.
    """
    m = ensure_metric(df, metric)
    # pivot rows: (model, backbone) -> columns: features
    pivot = (df.groupby(["model", "backbone", "features"], as_index=False)[m]
               .mean()
               .pivot(index=["model", "backbone"], columns="features", values=m))
    feats = [str(c) for c in pivot.columns]
    pairs = []
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            pairs.append((feats[i], feats[j]))
    pairs = pairs[:max_pairs]

    for (fa, fb) in pairs:
        sub = pivot[[fa, fb]].dropna()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        # color by model, marker by backbone (optional small variety)
        models = sorted({idx[0] for idx in sub.index})
        markers = ['o','s','^','D','P','X','v','>','<']
        size_base = int(OPTS.get("feature_scatter_size", 60))
        if OPTS.get("slide_small", False):
            size_base = int(size_base * float(OPTS.get("slide_scale", 1.25)))
        for mi, model in enumerate(models):
            rows = sub.loc[(model, slice(None)), :]
            ax.scatter(rows[fa], rows[fb], s=size_base,
                       color=warm_color(mi), marker=markers[mi % len(markers)],
                       alpha=0.9, label=str(model))
        # parity line
        both = pd.concat([sub[fa], sub[fb]], axis=0)
        lo, hi = float(both.min()), float(both.max())
        pad = 0.05 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555", linewidth=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        _finish(ax, title=_title(f"Features scatter — {fa} vs {fb} ({m})",
                                 chart="feature_pair_scatter", model="ALL", metric=m),
                xlabel=str(fa), ylabel=str(fb), rot=0)

        # slide-friendly boost
        _boost_for_slide(ax)

        fig.savefig(os.path.join(outdir, f"feature_scatter_{fa}_vs_{fb}_{m}.png"))
        plt.close(fig)


def align_bg_nobg(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str) -> pd.DataFrame:
    m_bg = ensure_metric(df_bg, metric)
    m_nb = ensure_metric(df_nbg, metric)
    a_bg = agg_mean(norm_cols(df_bg), ["model","backbone","features"], m_bg).rename(columns={f"{m_bg}_mean": "metric_bg"})
    a_nb = agg_mean(norm_cols(df_nbg), ["model","backbone","features"], m_nb).rename(columns={f"{m_nb}_mean": "metric_nbg"})
    m = pd.merge(a_bg, a_nb, on=["model","backbone","features"], how="inner")
    return m

def plot_bg_vs_nobg_bars(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str, outdir: str):
    m = align_bg_nobg(df_bg, df_nbg, metric)
    agg = m.groupby("features", as_index=False)[["metric_bg","metric_nbg"]].mean()
    x = np.arange(len(agg))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5), dpi=150)
    bars1 = ax.bar(x - width/2, agg["metric_bg"], width=width, label="BG", color=warm_color(1),
                   edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
    bars2 = ax.bar(x + width/2, agg["metric_nbg"], width=width, label="No-BG", color=warm_color(6),
                   edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
    for cont in (bars1, bars2):
        for b in cont:
            b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])

    feat_labels = [_fmt_feature_label(v) for v in agg["features"]]
    ax.set_xticks(x, feat_labels)

    # Slide-mode: enlarge numeric labels relative to bar heights
    old_override = OPTS.get("value_fontsize_override")
    if OPTS.get("slide_small", False):
        OPTS["value_fontsize_override"] = int(OPTS.get("font_value", 10) * float(OPTS.get("slide_value_scale", 1.4)))
    _add_bar_value_labels(ax, [bars1, bars2])
    OPTS["value_fontsize_override"] = old_override

    _finish(ax, title=_title(f"BG vs No-BG — grouped by features ({ensure_metric(df_bg, metric)})",
                             chart="bar_bg_vs_nobg", model="ALL", metric=ensure_metric(df_bg, metric)),
            xlabel="Features", ylabel=metric, rot=0)

    # slide-friendly boost
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"bar_bg_vs_nobg_by_features_{metric}.png"))
    plt.close(fig)

def plot_bg_vs_nobg_scatter(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str, outdir: str):
    m = align_bg_nobg(df_bg, df_nbg, metric)
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    feats = list_unique(m, "features")
    for i, f in enumerate(feats):
        sub = m[m["features"].astype(str) == str(f)]
        ax.scatter(sub["metric_bg"], sub["metric_nbg"], s=90, color=warm_color(i), alpha=0.9, label=str(f))
    allv = pd.concat([m["metric_bg"], m["metric_nbg"]], axis=0)
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.05 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _finish(ax, title=_title(f"Scatter — BG (x) vs No-BG (y)  ({ensure_metric(df_bg, metric)})",
                             chart="scatter_bg_vs_nobg", model="ALL", metric=ensure_metric(df_bg, metric)),
            xlabel="BG", ylabel="No-BG", rot=0)

    # slide-friendly boost
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"scatter_bg_vs_nobg_{metric}.png"))
    plt.close(fig)

def plot_delta_bars(df_bg: pd.DataFrame, df_nbg: pd.DataFrame, metric: str, outdir: str):
    m = align_bg_nobg(df_bg, df_nbg, metric)
    m["delta"] = m["metric_nbg"] - m["metric_bg"]
    agg = m.groupby("features", as_index=False)["delta"].mean()
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(10,5), dpi=150)
    colors = [warm_color(i) for i in range(len(agg))]
    bars = ax.bar(x, agg["delta"], color=colors, alpha=0.95, edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
    for b in bars:
        b.set_path_effects([pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.22), pe.Normal()])
    ax.axhline(0.0, color="#555555", linewidth=1)
    feat_labels = [_fmt_feature_label(v) for v in agg["features"]]
    ax.set_xticks(x, feat_labels)

    # Slide-mode: enlarge numeric labels relative to bars
    old_override = OPTS.get("value_fontsize_override")
    if OPTS.get("slide_small", False):
        OPTS["value_fontsize_override"] = int(OPTS.get("font_value", 10) * float(OPTS.get("slide_value_scale", 1.4)))
    _add_bar_value_labels(ax, [bars])
    OPTS["value_fontsize_override"] = old_override

    _finish(ax, title=_title(f"Δ (No-BG − BG) by features  ({ensure_metric(df_bg, metric)})",
                             chart="delta_bar", model="ALL", metric=ensure_metric(df_bg, metric)),
            xlabel="Features", ylabel="Δ", rot=0)

    # slide-friendly boost
    _boost_for_slide(ax)

    fig.savefig(os.path.join(outdir, f"delta_bar_by_features_{metric}.png"))
    plt.close(fig)


# =======================
# Figure builder functions (single CSV)
# =======================
def plot_per_model_bars(df: pd.DataFrame, metric: str, outdir: str):
    m = ensure_metric(df, metric)
    models = list_unique(df, "model")
    for model in models:
        sub = df[df["model"].astype(str) == str(model)].copy()
        agg = agg_mean(sub, by=["features", "backbone"], metric=m)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        bar_by_features_nested(
            ax, agg, primary="features", secondary="backbone",
            value_col=f"{m}_mean",
            title=_title(f"Bar — {model}  ({m})", chart="bar_per_model", model=model, metric=m),
            ylabel=m,
            sublabels=False
        )
        fig.savefig(os.path.join(outdir, f"bar_per-model_{model}_{m}.png"))
        plt.close(fig)

def plot_combined_bars(df: pd.DataFrame, metric: str, outdir: str):
    m = ensure_metric(df, metric)
    df = df.copy()
    if OPTS.get("combined_flat", False):
        # flat fallback
        df["model_backbone"] = df["model"].astype(str) + "-" + df["backbone"].astype(str)
        agg = agg_mean(df, by=["features", "model_backbone"], metric=m)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        bar_by_features_nested(
            ax, agg, primary="features", secondary="model_backbone", value_col=f"{m}_mean",
            title=_title(f"Bar — combined by model-backbone  ({m})", chart="bar_combined_flat", model="ALL", metric=m),
            ylabel=m,
            sublabels=OPTS.get("bar_sublabels", False)
        )
        fig.savefig(os.path.join(outdir, f"bar_combined_flat_{m}.png"))
        plt.close(fig)
    else:
        # features → model clusters → backbone bars
        agg = agg_mean(df, by=["features", "model", "backbone"], metric=m)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        bar_by_features_model_grouped(
            ax, agg, primary="features", model_col="model", backbone_col="backbone",
            value_col=f"{m}_mean",
            title=_title(f"Bar — combined (features→model clusters→backbone bars)  ({m})",
                         chart="bar_combined_grouped", model="ALL", metric=m),
            ylabel=m
        )
        fig.savefig(os.path.join(outdir, f"bar_combined_grouped_{m}.png"))
        plt.close(fig)

def plot_backbone_first_bars(df: pd.DataFrame, metric: str, outdir: str):
    """Produce two backbone-first charts."""
    m = ensure_metric(df, metric)

    # 1) Backbone → feature clusters → model bars
    agg1 = agg_mean(df, by=["backbone", "features", "model"], metric=m)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    bar_backbone_first_features_clusters_model_bars(
        ax, agg1, primary="backbone", cluster_col="features", inner_col="model",
        value_col=f"{m}_mean",
        title=_title(f"Bar — backbone→feature clusters→model bars  ({m})",
                     chart="bar_backbone_features_model", model="ALL", metric=m),
        ylabel=m
    )
    fig.savefig(os.path.join(outdir, f"bar_backbone_features_clusters_model_bars_{m}.png"))
    plt.close(fig)

    # 2) Backbone → model clusters → feature bars
    agg2 = agg_mean(df, by=["backbone", "model", "features"], metric=m)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    bar_backbone_first_model_clusters_feature_bars(
        ax, agg2, primary="backbone", cluster_col="model", inner_col="features",
        value_col=f"{m}_mean",
        title=_title(f"Bar — backbone→model clusters→feature bars  ({m})",
                     chart="bar_backbone_model_features", model="ALL", metric=m),
        ylabel=m
    )
    fig.savefig(os.path.join(outdir, f"bar_backbone_model_clusters_feature_bars_{m}.png"))
    plt.close(fig)

def plot_per_model_box(df: pd.DataFrame, metric: str, outdir: str):
    # Disabled by default; enabled via --with-boxplots
    pass

def plot_combined_box(df: pd.DataFrame, metric: str, outdir: str):
    # Disabled by default; enabled via --with-boxplots
    pass


# =========
# Main CLI
# =========
def main():
    ap = argparse.ArgumentParser(description="Visualize collected test CSVs (single or BG vs No-BG).")
    ap.add_argument("--csv", required=True, help="Path to main CSV.")
    ap.add_argument("--csv2", default=None, help="Optional second CSV (for BG vs No-BG).")
    ap.add_argument("--metric", default="mean", help="Metric column to use (default: mean).")
    ap.add_argument("--outdir", default="./viz_out", help="Output directory for figures.")
    ap.add_argument("--figsize", default="12x6", help="Figure size WxH in inches, e.g., 12x6.")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")

    # Presentation toggles
    ap.add_argument("--show-values", action="store_true", help="Write values on bars.")
    ap.add_argument("--value-fmt", default="{:.2f}", help="Format for bar value labels.")
    ap.add_argument("--legend", action="store_true", help="Show legend (off by default).")
    ap.add_argument("--no-hide-spines", action="store_true", help="Keep full chart border (don’t remove spines).")
    ap.add_argument("--hide-yaxis", action="store_true", help="Hide y-axis ticks/labels/grid.")
    ap.add_argument("--hide-xaxis", action="store_true", help="Hide x-axis ticks/labels.")

    # Titles & typography
    ap.add_argument("--title", default=None,
                    help="Override plot titles. Placeholders: {chart}, {model}, {metric}.")
    ap.add_argument("--no-titles", action="store_true", help="Disable figure titles.")
    ap.add_argument("--font-title", type=int, default=12, help="Title font size.")
    ap.add_argument("--font-axis", type=int, default=11, help="Axis label font size.")
    ap.add_argument("--font-tick", type=int, default=10, help="Tick label font size.")
    ap.add_argument("--font-value", type=int, default=10, help="Bar value label font size.")

    # Coloring (cycle by backbone | features | model)
    ap.add_argument("--color-by", choices=["backbone", "features", "model"], default="backbone",
                    help="Which category cycles the bar colors (default: backbone).")

    # Combined chart controls
    ap.add_argument("--combined-flat", action="store_true",
                    help="Use flat combined bars (model-backbone as secondary).")
    ap.add_argument("--model-gap", type=float, default=0.06,
                    help="Gap between model clusters inside each features group.")
    ap.add_argument("--model-label-offset", type=float, default=None,
                    help="Axes-relative Y offset for model labels (negative=below axis).")
    ap.add_argument("--backbone-label-offset", type=float, default=None,
                    help="Axes-relative Y offset for backbone row (defaults to model_label_offset-0.03).")
    ap.add_argument("--feature-label-offset", type=float, default=None,
                    help="Axes-relative Y offset for feature row (defaults to backbone_label_offset-0.03).")
    ap.add_argument("--no-backbone-sublabels", action="store_true",
                    help="Disable backbone sublabels under bars in combined grouped chart.")

    # Feature pair scatters
    ap.add_argument("--feature-scatter-size", type=int, default=60,
                    help="Marker size for feature pair scatter plots (default: 60).")

    # Slide-friendly boosts for comparison charts
    ap.add_argument("--slide-small", action="store_true",
                    help="Boost fonts/values for BG vs No-BG and feature comparison charts to look good small on slides.")
    ap.add_argument("--slide-scale", type=float, default=1.25,
                    help="Scale factor for labels/ticks/titles when --slide-small is on (default: 1.25).")
    ap.add_argument("--slide-value-scale", type=float, default=1.40,
                    help="Additional scale factor for numeric bar labels when --slide-small is on (default: 1.40).")

    # Boxplots
    ap.add_argument("--with-boxplots", action="store_true", help="Also render boxplots (disabled by default).")

    ap.add_argument("--iphone-only", action="store_true",
                    help="For BG vs No-BG comparisons, restrict rows to those mentioning 'iphone' (case-insensitive).")

    args = ap.parse_args()

    mkdir_p(args.outdir)
    w, h = (int(float(x)) for x in args.figsize.lower().split("x"))
    plt.rcParams["figure.figsize"] = (w, h)
    plt.rcParams["figure.dpi"] = args.dpi

    # Wire options
    OPTS["show_values"] = args.show_values
    OPTS["value_fmt"] = args.value_fmt
    OPTS["show_legend"] = args.legend
    OPTS["hide_spines"] = not args.no_hide_spines
    OPTS["hide_yaxis"] = args.hide_yaxis
    OPTS["hide_xaxis"] = args.hide_xaxis
    OPTS["title"] = args.title
    OPTS["combined_flat"] = args.combined_flat
    OPTS["model_gap"] = args.model_gap
    OPTS["backbone_sublabels"] = not args.no_backbone_sublabels
    OPTS["color_by"] = args.color_by

    # Typography wiring
    OPTS["show_titles"] = not args.no_titles
    OPTS["font_title"] = args.font_title
    OPTS["font_axis"]  = args.font_axis
    OPTS["font_tick"]  = args.font_tick
    OPTS["font_value"] = args.font_value

    # Label row offsets
    if args.model_label_offset is not None:
        OPTS["model_label_offset"] = args.model_label_offset
    OPTS["backbone_label_offset"] = args.backbone_label_offset
    OPTS["feature_label_offset"]  = args.feature_label_offset

    # Feature scatter sizing
    OPTS["feature_scatter_size"] = args.feature_scatter_size

    # Slide-friendly boosts
    OPTS["slide_small"] = args.slide_small
    OPTS["slide_scale"] = args.slide_scale
    OPTS["slide_value_scale"] = args.slide_value_scale

    # Load data
    df = norm_cols(pd.read_csv(args.csv))

    # Single-CSV charts
    plot_per_model_bars(df, args.metric, args.outdir)
    plot_combined_bars(df, args.metric, args.outdir)

    # Feature-group comparisons
    plot_feature_mean_bar(df, args.metric, args.outdir)
    plot_feature_pair_scatters(df, args.metric, args.outdir, max_pairs=8)

    # Backbone-first (two variants)
    plot_backbone_first_bars(df, args.metric, args.outdir)

    # Boxplots (disabled by default)
    if args.with_boxplots:
        plot_per_model_box(df, args.metric, args.outdir)
        plot_combined_box(df, args.metric, args.outdir)

    # BG vs No-BG comparisons
    if args.csv2:
        df2 = norm_cols(pd.read_csv(args.csv2))
        plot_bg_vs_nobg_bars(df, df2, args.metric, args.outdir)
        plot_bg_vs_nobg_scatter(df, df2, args.metric, args.outdir)
        # plot_delta_bars(df, df2, args.metric, args.outdir)

    print(f"[viz] Saved figures to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
