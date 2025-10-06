#!/usr/bin/env python3
from __future__ import annotations
from PIL import Image
import os, argparse, json
import numpy as np
from typing import Tuple, Dict, Any, List

EXTS = ('.jpg', '.jpeg', '.png', '.heic', '.bmp', '.tiff', '.webp')

def _dir_has_images(p: str) -> bool:
    try:
        return any(f.lower().endswith(EXTS) for f in os.listdir(p))
    except Exception:
        return False

# =========================
# Color utilities (sRGB -> Lab)
# =========================
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float32) / 255.0
    out = np.empty_like(c, dtype=np.float32)
    mask = c <= 0.04045
    out[mask] = c[mask] / 12.92
    out[~mask] = ((c[~mask] + 0.055) / 1.055) ** 2.4
    return out

def _rgb_to_xyz(rgb_lin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return X, Y, Z

def _f_lab(t: np.ndarray) -> np.ndarray:
    eps = 0.008856
    kappa = 903.3
    f = np.empty_like(t, dtype=np.float32)
    mask = t > eps
    f[mask] = np.cbrt(t[mask])
    f[~mask] = (kappa * t[~mask] + 16.0) / 116.0
    return f

def rgb_to_lab_np(rgb_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_lin = _srgb_to_linear(rgb_uint8)
    X, Y, Z = _rgb_to_xyz(rgb_lin)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883  # D65
    fx = _f_lab(X / Xn)
    fy = _f_lab(Y / Yn)
    fz = _f_lab(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b

def de76_median_std(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    Lm = np.median(L); am = np.median(a); bm = np.median(b)
    dE = np.sqrt((L - Lm)**2 + (a - am)**2 + (b - bm)**2)
    return float(np.median(dE)), float(np.std(dE))

# =========================
# Geometry helpers
# =========================
def centered_bbox(cx: float, cy: float, width: int, height: int, W: int, H: int):
    l = int(round(cx - width / 2))
    t = int(round(cy - height / 2))
    r = l + width
    b = t + height
    if l < 0:
        r -= l; l = 0
    if t < 0:
        b -= t; t = 0
    if r > W:
        l -= (r - W); r = W
    if b > H:
        t -= (b - H); b = H
    l = max(0, min(l, W - width))
    t = max(0, min(t, H - height))
    r = l + width; b = t + height
    return l, t, r, b

# =========================
# Filters and edges
# =========================
def sobel_mag(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gpad = np.pad(g, 1, mode="edge")
    gx = (gpad[1:-1, 2:] - gpad[1:-1, :-2]
          + 2*(gpad[2:, 2:] - gpad[2:, :-2])
          + (gpad[:-2, 2:] - gpad[:-2, :-2]))
    gy = (gpad[2:, 1:-1] - gpad[:-2, 1:-1]
          + 2*(gpad[2:, 2:] - gpad[:-2, 2:])
          + (gpad[2:, :-2] - gpad[:-2, :-2]))
    return np.sqrt(gx*gx + gy*gy)

def luminance(rgb_uint8: np.ndarray) -> np.ndarray:
    R = rgb_uint8[..., 0].astype(np.float32)
    G = rgb_uint8[..., 1].astype(np.float32)
    B = rgb_uint8[..., 2].astype(np.float32)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B

# =========================
# Masks
# =========================
def border_and_corner_masks(h: int, w: int, border_frac: float, corner_frac: float=0.25):
    t = max(1, int(border_frac * min(h, w)))
    border = np.zeros((h, w), dtype=bool)
    border[:t, :] = True; border[-t:, :] = True
    border[:, :t] = True; border[:, -t:] = True
    interior = ~border
    # corners within the border bands
    cs = max(1, int(corner_frac * t))  # small corner squares from the band thickness
    corners = np.zeros((h, w), dtype=bool)
    corners[:t, :t] = True; corners[:t, -t:] = True
    corners[-t:, :t] = True; corners[-t:, -t:] = True
    return border, interior, corners, t

# =========================
# Simple border (original near white/black) — conservative
# =========================
def simple_border_shift(
    rgb_crop: np.ndarray,
    border_frac: float,
    white_thr: int = 245,
    black_thr: int = 10
) -> Tuple[float, float, float, Dict[str, Any]]:
    h, w, _ = rgb_crop.shape
    if h == 0 or w == 0:
        return 0.0, 0.0, 0.0, {"reason": "empty"}

    border, _, _, t = border_and_corner_masks(h, w, border_frac)
    R = rgb_crop[..., 0]; G = rgb_crop[..., 1]; B = rgb_crop[..., 2]
    white_mask = (R >= white_thr) & (G >= white_thr) & (B >= white_thr)
    black_mask = (R <= black_thr) & (G <= black_thr) & (B <= black_thr)
    extreme = (white_mask | black_mask) & border
    if not np.any(extreme):
        return 0.0, 0.0, 0.0, {"reason": "no_extreme"}

    ys, xs = np.nonzero(extreme)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y_mean = ys.mean(); x_mean = xs.mean()
    vx, vy = (x_mean - cx), (y_mean - cy)
    mag = float(np.hypot(vx, vy))
    if mag == 0.0:
        return 0.0, 0.0, 0.0, {"reason": "zero_mag"}

    frac_bad = extreme.sum() / border.sum()
    dx = -(vx / mag) * frac_bad
    dy = -(vy / mag) * frac_bad
    return float(dx), float(dy), float(frac_bad), {
        "reason": "border_simple",
        "frac_bad": float(frac_bad),
        "band": t
    }

# =========================
# Adaptive border (MAD-Z + edges) — with corner emphasis & gates
# =========================
def adaptive_border_shift(
    rgb_crop: np.ndarray,
    border_frac: float,
    z_k: float,
    alpha: float,
    beta: float,
    min_border_frac: float,
    min_corner_frac: float
) -> Tuple[float, float, float, Dict[str, Any]]:
    h, w, _ = rgb_crop.shape
    if h == 0 or w == 0:
        return 0.0, 0.0, 0.0, {"reason": "empty"}

    Y = luminance(rgb_crop)
    border, interior, corners, t = border_and_corner_masks(h, w, border_frac)

    if not np.any(interior):
        return 0.0, 0.0, 0.0, {"reason": "no_interior"}

    # robust stats on interior
    Y_int = Y[interior]
    med = np.median(Y_int)
    mad = np.median(np.abs(Y_int - med)) + 1e-6
    Z = (Y - med) / mad

    # soft extreme on border
    extreme_border = np.clip(np.abs(Z) - z_k, 0, None)
    extreme_border[~border] = 0.0

    # edges (robustly normalized)
    E = sobel_mag(Y)
    e95 = np.percentile(E, 95.0) + 1e-6
    edge_norm = np.clip(E / e95, 0.0, 1.0)
    edge_norm[~border] = 0.0

    # combine, but emphasize corners 1.5x
    Wb = alpha * extreme_border + beta * edge_norm
    Wb_corner = Wb.copy()
    Wb_corner[corners] *= 1.5

    total_w = Wb_corner.sum()
    if total_w <= 1e-6:
        return 0.0, 0.0, 0.0, {"reason": "weak_border"}

    # gate 1: fraction of border activated
    frac_border = float((Wb > 0).sum()) / float(border.sum() + 1e-6)
    if frac_border < min_border_frac:
        return 0.0, 0.0, 0.0, {"reason": "under_frac", "frac_border": frac_border}

    # gate 2: corner concentration
    corner_frac = float(Wb[corners].sum()) / float(Wb.sum() + 1e-6)
    if corner_frac < min_corner_frac:
        return 0.0, 0.0, 0.0, {"reason": "corner_low", "corner_frac": corner_frac, "frac_border": frac_border}

    ys, xs = np.nonzero(Wb_corner > 0)
    wts = Wb_corner[ys, xs]
    cy_w = (ys * wts).sum() / (wts.sum() + 1e-6)
    cx_w = (xs * wts).sum() / (wts.sum() + 1e-6)
    cy = (h - 1) / 2.0; cx = (w - 1) / 2.0
    vx = cx_w - cx; vy = cy_w - cy
    mag = np.hypot(vx, vy)
    if mag == 0.0:
        return 0.0, 0.0, 0.0, {"reason": "zero_mag"}

    # strength: weighted activity density
    strength = float(total_w / (border.sum() + 1e-6))
    dx = -(vx / mag) * strength
    dy = -(vy / mag) * strength
    return float(dx), float(dy), float(strength), {
        "reason": "border_adaptive",
        "frac_border": frac_border,
        "corner_frac": corner_frac,
        "band": t,
        "strength": float(strength)
    }

# =========================
# Shadow stripe detector (PCA on dark interior points)
# =========================
def shadow_shift(
    rgb_crop: np.ndarray,
    border_frac: float,
    z_k_shadow: float,
    stripe_min_frac: float,
    elong_min: float
) -> Tuple[float, float, float, Dict[str, Any]]:
    h, w, _ = rgb_crop.shape
    Y = luminance(rgb_crop)
    border, interior, _, _ = border_and_corner_masks(h, w, border_frac)
    if not np.any(interior):
        return 0.0, 0.0, 0.0, {"reason": "no_interior"}

    Y_int = Y[interior]
    med = np.median(Y_int)
    mad = np.median(np.abs(Y_int - med)) + 1e-6
    Z = (Y - med) / mad

    dark = (Z < -z_k_shadow) & interior
    frac_dark = float(dark.sum()) / float(interior.sum() + 1e-6)
    if frac_dark < stripe_min_frac:
        return 0.0, 0.0, 0.0, {"reason": "few_dark", "frac_dark": frac_dark}

    ys, xs = np.nonzero(dark)
    if ys.size < 20:
        return 0.0, 0.0, 0.0, {"reason": "too_few_points", "frac_dark": frac_dark}

    Xc = np.column_stack([xs, ys]).astype(np.float32)
    mu = Xc.mean(axis=0, keepdims=True)
    X0 = Xc - mu
    cov = (X0.T @ X0) / (X0.shape[0] - 1 + 1e-6)
    vals, vecs = np.linalg.eigh(cov)
    v_major = vecs[:, 1]
    elong = float(vals[1] / (vals[0] + 1e-6))
    if elong < elong_min:
        return 0.0, 0.0, 0.0, {"reason": "not_elongated", "frac_dark": frac_dark, "elong": elong}

    vx, vy = float(v_major[0]), float(v_major[1])
    nmag = (vx**2 + vy**2) ** 0.5 + 1e-6
    vx /= nmag; vy /= nmag
    nx, ny = -vy, vx  # normal

    h_idx, w_idx = np.indices((h, w))
    s = (w_idx - mu[0, 0]) * nx + (h_idx - mu[0, 1]) * ny
    side_pos = (s > 0) & interior
    side_neg = (s < 0) & interior
    dens_pos = (dark & side_pos).sum() / (side_pos.sum() + 1e-6)
    dens_neg = (dark & side_neg).sum() / (side_neg.sum() + 1e-6)
    sign = -1.0 if dens_pos > dens_neg else 1.0

    asym = abs(dens_pos - dens_neg)
    strength = float(frac_dark * (0.5 + 1.5 * asym))

    dx = sign * nx * strength
    dy = sign * ny * strength
    return float(dx), float(dy), float(strength), {
        "reason": "shadow",
        "frac_dark": frac_dark,
        "elong": elong,
        "dens_pos": float(dens_pos),
        "dens_neg": float(dens_neg),
        "stripe_dir": (float(v_major[0]), float(v_major[1])),
        "normal_dir": (float(nx), float(ny)),
        "strength": strength
    }

# =========================
# Good crop sanity metrics
# =========================
def good_crop_metrics(rgb_crop: np.ndarray, border_frac: float) -> Dict[str, float]:
    Y = luminance(rgb_crop)
    border, interior, _, _ = border_and_corner_masks(rgb_crop.shape[0], rgb_crop.shape[1], border_frac)
    E = sobel_mag(Y)
    e95 = np.percentile(E, 95.0) + 1e-6
    edge_norm = np.clip(E / e95, 0.0, 1.0)
    edge_border_mean = float(edge_norm[border].mean()) if np.any(border) else 0.0
    L, a, b = rgb_to_lab_np(rgb_crop)
    dE_med, dE_std = de76_median_std(L, a, b)
    return {"edge_border_mean": edge_border_mean, "dE_med": dE_med, "dE_std": dE_std}

# =========================
# Crop driver (combines detectors)
# =========================
def crop_image_smart(
    image: Image.Image,
    image_path: str,
    width: int,
    height: int,
    # shift scaling
    border_frac: float,
    max_shift_frac: float,
    margin_frac: float,
    iters: int,
    # decision prefs
    border_mode: str,
    prefer_shadow_bias: float,
    # adaptive border knobs (tamed)
    z_k_border: float,
    alpha: float,
    beta: float,
    min_border_frac: float,
    min_corner_frac: float,
    # simple border knobs
    white_thr: int,
    black_thr: int,
    # shadow knobs
    z_k_shadow: float,
    stripe_min_frac: float,
    elong_min: float,
    # goodness gate
    de_good: float,
    edge_low: float,
    # learned thresholds (optional)
    border_score_thresh: float | None,
    shadow_score_thresh: float | None,
    # diagnostics
    diag: bool,
    verbose: bool
) -> Image.Image:

    W, H = image.size
    width = min(width, W)
    height = min(height, H)
    cx, cy = W / 2.0, H / 2.0

    sref = min(width, height)
    max_shift_px = max_shift_frac * sref
    margin_px = margin_frac * sref

    rgb = image.convert("RGB") if image.mode != "RGB" else image

    last_info: Dict[str, Any] = {}

    for _ in range(max(1, iters)):
        l, t, r, b = centered_bbox(cx, cy, width, height, W, H)
        crop_np = np.array(rgb.crop((l, t, r, b)), dtype=np.uint8)

        metrics = good_crop_metrics(crop_np, border_frac)
        dE_med = metrics["dE_med"]; edge_border_mean = metrics["edge_border_mean"]

        # Goodness gate: if very uniform & low edge on border, don't shift.
        if dE_med <= de_good and edge_border_mean <= edge_low:
            if verbose:
                print(f"[ok]  {image_path} dE_med={dE_med:.2f} edgeB={edge_border_mean:.2f} -> keep")
            break

        # --- detectors ---
        sdx, sdy, sstr, sinfo = shadow_shift(crop_np, border_frac, z_k_shadow, stripe_min_frac, elong_min)
        adx, ady, astr, ainfo = adaptive_border_shift(
            crop_np, border_frac, z_k_border, alpha, beta, min_border_frac, min_corner_frac
        )
        sdx2, sdy2, sstr2, sinfo2 = simple_border_shift(crop_np, border_frac, white_thr, black_thr)

        # Convert unit shifts to pixels (+ margin if fired)
        def to_px(dx, dy, s):
            if s <= 0: return 0.0, 0.0
            return dx * max_shift_px + np.sign(dx) * margin_px, dy * max_shift_px + np.sign(dy) * margin_px

        spx, spy = to_px(sdx, sdy, sstr)
        apx, apy = to_px(adx, ady, astr)
        spx2, spy2 = to_px(sdx2, sdy2, sstr2)

        # Optional learned thresholds gate
        if shadow_score_thresh is not None and sstr < shadow_score_thresh:
            spx = spy = 0.0; sstr = 0.0; sinfo["reason"] = "shadow_below_thresh"
        if border_score_thresh is not None:
            # For border score we use max of adaptive/simple strengths
            bscore = max(astr, sstr2)
            if bscore < border_score_thresh:
                apx = apy = 0.0; astr = 0.0; ainfo["reason"] = "border_below_thresh"
                spx2 = spy2 = 0.0; sstr2 = 0.0

        # Pick border candidate according to mode (conservative)
        if border_mode == "simple":
            bpx, bpy, bstr, binfo = spx2, spy2, sstr2, sinfo2
        elif border_mode == "adaptive":
            bpx, bpy, bstr, binfo = apx, apy, astr, ainfo
        else:  # "both" -> prefer the one with smaller magnitude if both fire, else whichever fires
            magA = (apx**2 + apy**2)**0.5
            magS = (spx2**2 + spy2**2)**0.5
            if astr > 0 and sstr2 > 0:
                # choose smaller magnitude (more conservative)
                if magA <= magS:
                    bpx, bpy, bstr, binfo = apx, apy, astr, ainfo
                else:
                    bpx, bpy, bstr, binfo = spx2, spy2, sstr2, sinfo2
            elif astr > 0:
                bpx, bpy, bstr, binfo = apx, apy, astr, ainfo
            else:
                bpx, bpy, bstr, binfo = spx2, spy2, sstr2, sinfo2

        # Final decision between SHADOW vs BORDER (bias toward shadow if stronger)
        # compute "decision score" ~ |shift_px| scaled
        bscore_px = (abs(bpx) + abs(bpy))
        sscore_px = (abs(spx) + abs(spy)) * prefer_shadow_bias

        if sscore_px > bscore_px and sstr > 0:
            dx, dy = spx, spy
            reason = "shadow"
            info = {**sinfo, "dE_med": dE_med, "edgeB": edge_border_mean}
        elif bstr > 0:
            dx, dy = bpx, bpy
            reason = binfo.get("reason", "border")
            info = {**binfo, "dE_med": dE_med, "edgeB": edge_border_mean}
        else:
            dx = dy = 0.0
            reason = None
            info = {"dE_med": dE_med, "edgeB": edge_border_mean}

        # apply
        cx = float(np.clip(cx + dx, width / 2, W - width / 2))
        cy = float(np.clip(cy + dy, height / 2, H - height / 2))
        last_info = info

        if diag and reason is not None:
            if reason.startswith("border"):
                print(f"[det] {image_path} reason={reason} dE_med={info['dE_med']:.2f} "
                      f"edgeB={info['edgeB']:.2f} shift=({dx:.1f},{dy:.1f}) "
                    #   f"meta={{{k: info[k] for k in info if k not in ('dE_med','edgeB')}}}"
                    )
            else:
                print(f"[det] {image_path} reason=shadow dE_med={info['dE_med']:.2f} "
                      f"edgeB={info['edgeB']:.2f} shift=({dx:.1f},{dy:.1f}) "
                      f"frac_dark={info.get('frac_dark',0):.3f} elong={info.get('elong',0):.2f}")

        if abs(dx) + abs(dy) < 0.25:
            break

    l, t, r, b = centered_bbox(cx, cy, width, height, W, H)
    return image.crop((l, t, r, b))

# =========================
# Learning thresholds from examples
# =========================
def walk_images(root: str) -> List[str]:
    paths = []
    for base, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(EXTS):
                paths.append(os.path.join(base, fn))
    return paths

def score_image_for_learning(path: str, size: int, border_frac: float,
                             z_k_border: float, alpha: float, beta: float,
                             z_k_shadow: float, stripe_min_frac: float, elong_min: float,
                             white_thr: int, black_thr: int) -> Dict[str, float]:
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return {}
    W, H = im.size
    s = min(size, W, H)
    l = (W - s) // 2; t = (H - s) // 2
    crop_np = np.array(im.crop((l, t, l + s, t + s)), dtype=np.uint8)

    # border adaptive/simple strengths and shadow strength (unit strengths)
    adx, ady, astr, _ = adaptive_border_shift(crop_np, border_frac, z_k_border, alpha, beta,
                                              min_border_frac=0.10, min_corner_frac=0.20)
    sdx2, sdy2, sstr2, _ = simple_border_shift(crop_np, border_frac, white_thr, black_thr)
    sdx, sdy, sstr, _ = shadow_shift(crop_np, border_frac, z_k_shadow, stripe_min_frac, elong_min)

    # Goodness metrics
    mets = good_crop_metrics(crop_np, border_frac)
    return {
        "border_adaptive": float(astr),
        "border_simple": float(sstr2),
        "shadow": float(sstr),
        "dE_med": mets["dE_med"],
        "edgeB": mets["edge_border_mean"]
    }

def learn_params(good_dir: str, border_dir: str, shadow_dir: str, size: int,
                 base_knobs: Dict[str, Any]) -> Dict[str, Any]:
    # Collect scores
    goods = [p for p in walk_images(good_dir)] if good_dir else []
    borders = [p for p in walk_images(border_dir)] if border_dir else []
    shadows = [p for p in walk_images(shadow_dir)] if shadow_dir else []
    if not (goods and (borders or shadows)):
        raise RuntimeError("Need at least some 'good' and some 'border' or 'shadow' images.")

    def batch(paths):
        out = []
        for p in paths:
            sc = score_image_for_learning(p, size=size, border_frac=base_knobs["border_frac"],
                                          z_k_border=base_knobs["z_k_border"],
                                          alpha=base_knobs["alpha"], beta=base_knobs["beta"],
                                          z_k_shadow=base_knobs["z_k_shadow"],
                                          stripe_min_frac=base_knobs["stripe_min_frac"],
                                          elong_min=base_knobs["elong_min"],
                                          white_thr=base_knobs["white_thr"],
                                          black_thr=base_knobs["black_thr"])
            if sc:
                out.append(sc)
        return out

    G = batch(goods)
    B = batch(borders) if borders else []
    S = batch(shadows) if shadows else []

    # Propose thresholds: pick 90th percentile of good, and midpoint to 50th percentile of class
    def perc(vals, q):
        if not vals: return 0.0
        return float(np.percentile(np.array(vals, dtype=np.float32), q))

    # Border score: max(adaptive,simple)
    G_border = [max(g["border_adaptive"], g["border_simple"]) for g in G]
    B_border = [max(b["border_adaptive"], b["border_simple"]) for b in B] if B else []
    # Shadow score
    G_shadow = [g["shadow"] for g in G]
    S_shadow = [s["shadow"] for s in S] if S else []

    border_good_p90 = perc(G_border, 90)
    border_pos_p50 = perc(B_border, 50) if B_border else (border_good_p90 + 0.15)
    border_thresh = (border_good_p90 + border_pos_p50) / 2.0

    shadow_good_p90 = perc(G_shadow, 90)
    shadow_pos_p50 = perc(S_shadow, 50) if S_shadow else (shadow_good_p90 + 0.15)
    shadow_thresh = (shadow_good_p90 + shadow_pos_p50) / 2.0

    # Also suggest de_good / edge_low cutoffs from good set
    dE_good_p80 = perc([g["dE_med"] for g in G], 80)
    edge_low_p80 = perc([g["edgeB"] for g in G], 80)

    learned = {
        "border_score_thresh": float(border_thresh),
        "shadow_score_thresh": float(shadow_thresh),
        "de_good": float(max(0.4, min(2.0, dE_good_p80))),
        "edge_low": float(max(0.05, min(0.6, edge_low_p80))),
    }

    print("[learn] Proposed thresholds:")
    for k, v in learned.items():
        print(f"  {k}: {v:.3f}")

    # Quick report
    def rate(vals, thr):
        return float(np.mean(np.array(vals) >= thr)) if vals else 0.0
    if B_border:
        print(f"[learn] Border recall@thr ≈ {rate(B_border, border_thresh):.2%}   (good FPR ≈ {rate(G_border, border_thresh):.2%})")
    if S_shadow:
        print(f"[learn] Shadow recall@thr ≈ {rate(S_shadow, shadow_thresh):.2%}   (good FPR ≈ {rate(G_shadow, shadow_thresh):.2%})")

    return learned

# =========================
# I/O wrappers
# =========================
def crop_folder(folder, destination_folder, width, height, **kwargs):
    for root, _, files in os.walk(folder):
        rel_path = os.path.relpath(root, folder)
        dest_dir = os.path.join(destination_folder, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        for filename in files:
            if not filename.lower().endswith(EXTS):
                continue
            image_path = os.path.join(root, filename)
            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                continue
            try:
                out = crop_image_smart(image=image, image_path=image_path, width=width, height=height, **kwargs)
                out.save(os.path.join(dest_dir, filename))
            except Exception as e:
                print(f"Failed on {image_path}: {e}")

def crop_all_albums(original_root, cropped_root, width, height, **kwargs):
    for album_name in os.listdir(original_root):
        album_path = os.path.join(original_root, album_name)
        if os.path.isdir(album_path):
            dest_album_folder = os.path.join(cropped_root, album_name)
            print(f"Processing {album_path}")
            crop_folder(album_path, dest_album_folder, width, height, **kwargs)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Smart crop with border/shadow detection; supports learning thresholds.")
    # Modes
    ap.add_argument("--learn_params", action="store_true",
                    help="Learn thresholds from labeled examples instead of cropping.")
    ap.add_argument("--learn_good", type=str, default=None, help="Folder of good crops")
    ap.add_argument("--learn_border", type=str, default=None, help="Folder of border crops")
    ap.add_argument("--learn_shadow", type=str, default=None, help="Folder of shadow crops")
    ap.add_argument("--save_params", type=str, default=None, help="Where to save learned thresholds JSON")
    ap.add_argument("--params_json", type=str, default=None, help="Load thresholds JSON (from learn step)")

    # Crop I/O
    ap.add_argument("--input_dir", type=str, default="../../data/Pantone/original/focus/", help="Input albums root")
    ap.add_argument("--output_dir", type=str, default="../../data/Pantone/cropped-256-ada/focus/", help="Output cropped root")
    ap.add_argument("--size", type=int, default=256, help="Output crop size (square)")
    ap.add_argument("--single_image", type=str, default=None,
                help="Path to one image to crop (overrides --input_dir).")
    ap.add_argument("--out_file", type=str, default=None,
                help="Output path for --single_image; default mirrors under --output_dir.")

    # General shifts
    ap.add_argument("--border_frac", type=float, default=0.12, help="Border band thickness fraction")
    ap.add_argument("--max_shift_frac", type=float, default=0.25, help="Max proportional shift of min(size)")
    ap.add_argument("--margin_frac", type=float, default=0.05, help="Base margin shift when firing")
    ap.add_argument("--iters", type=int, default=2, help="Re-evaluate & push iterations")
    ap.add_argument("--border_mode", type=str, default="both", choices=["simple", "adaptive", "both"],
                    help="Which border detector to use (or compare both)")

    # Preference
    ap.add_argument("--prefer_shadow_bias", type=float, default=1.2,
                    help="Multiply shadow score before comparing to border (bias toward shadow)")

    # Adaptive border knobs (tamed)
    ap.add_argument("--z_k_border", type=float, default=2.5, help="MAD-Z threshold for border")
    ap.add_argument("--alpha", type=float, default=0.6, help="Weight on luminance outlier")
    ap.add_argument("--beta", type=float, default=0.4, help="Weight on edge magnitude")
    ap.add_argument("--min_border_frac", type=float, default=0.15, help="Min border activation fraction to accept")
    ap.add_argument("--min_corner_frac", type=float, default=0.30, help="Min corner weight fraction to accept")

    # Simple border knobs
    ap.add_argument("--white_thr", type=int, default=245, help="Near-white threshold (per channel)")
    ap.add_argument("--black_thr", type=int, default=10, help="Near-black threshold (per channel)")

    # Shadow knobs
    ap.add_argument("--z_k_shadow", type=float, default=2.0, help="MAD-Z (darker than interior)")
    ap.add_argument("--stripe_min_frac", type=float, default=0.06, help="Min dark fraction for stripe")
    ap.add_argument("--elong_min", type=float, default=2.5, help="Min elongation (λ1/λ2) for stripe")

    # Good-crop gate
    ap.add_argument("--de_good", type=float, default=1.0, help="Max ΔE*76 median to accept as good")
    ap.add_argument("--edge_low", type=float, default=0.30, help="Max mean border edge to accept as good")

    # Thresholds (can be learned)
    ap.add_argument("--border_score_thresh", type=float, default=None,
                    help="Optional min border score to trigger (unit strength)")
    ap.add_argument("--shadow_score_thresh", type=float, default=None,
                    help="Optional min shadow score to trigger (unit strength)")

    # Diagnostics
    ap.add_argument("--diag", action="store_true", help="Print diagnostics when detectors fire")
    ap.add_argument("--verbose", action="store_true", help="More prints, including kept crops")

    args = ap.parse_args()

    # Consolidate knobs
    knobs = dict(
        border_frac=args.border_frac,
        max_shift_frac=args.max_shift_frac,
        margin_frac=args.margin_frac,
        iters=args.iters,
        border_mode=args.border_mode,
        prefer_shadow_bias=args.prefer_shadow_bias,
        z_k_border=args.z_k_border,
        alpha=args.alpha,
        beta=args.beta,
        min_border_frac=args.min_border_frac,
        min_corner_frac=args.min_corner_frac,
        white_thr=args.white_thr,
        black_thr=args.black_thr,
        z_k_shadow=args.z_k_shadow,
        stripe_min_frac=args.stripe_min_frac,
        elong_min=args.elong_min,
        de_good=args.de_good,
        edge_low=args.edge_low,
        border_score_thresh=args.border_score_thresh,
        shadow_score_thresh=args.shadow_score_thresh,
        diag=args.diag,
        verbose=args.verbose
    )

    # Load learned params if provided
    if args.params_json:
        try:
            with open(args.params_json, "r") as f:
                learned = json.load(f)
            for k in ("border_score_thresh", "shadow_score_thresh", "de_good", "edge_low"):
                if k in learned and learned[k] is not None:
                    knobs[k] = float(learned[k])
            print(f"[load] Loaded learned params from {args.params_json}: "
                  f"border_score_thresh={knobs.get('border_score_thresh')} "
                  f"shadow_score_thresh={knobs.get('shadow_score_thresh')} "
                  f"de_good={knobs.get('de_good')} edge_low={knobs.get('edge_low')}")
        except Exception as e:
            print(f"[warn] Failed to load {args.params_json}: {e}")

    if args.learn_params:
        learned = learn_params(args.learn_good, args.learn_border, args.learn_shadow, args.size, {
            "border_frac": knobs["border_frac"],
            "z_k_border": knobs["z_k_border"],
            "alpha": knobs["alpha"],
            "beta": knobs["beta"],
            "z_k_shadow": knobs["z_k_shadow"],
            "stripe_min_frac": knobs["stripe_min_frac"],
            "elong_min": knobs["elong_min"],
            "white_thr": knobs["white_thr"],
            "black_thr": knobs["black_thr"],
        })
        if args.save_params:
            with open(args.save_params, "w") as f:
                json.dump(learned, f, indent=2)
            print(f"[save] Saved learned params -> {args.save_params}")
        return

    if not args.input_dir or not args.output_dir:
        raise SystemExit("For cropping, provide --input_dir and --output_dir (or use --learn_params).")

    # Single image mode
    if args.single_image:
        im_path = args.single_image
        im = Image.open(im_path).convert("RGB")
        # Decide output path
        if args.out_file:
            out_path = args.out_file
        else:
            rel = os.path.basename(im_path)
            out_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out = crop_image_smart(image=im, image_path=im_path, width=args.size, height=args.size, **knobs)
        out.save(out_path)
        if args.verbose:
            print(f"[ok] wrote {out_path}")
        return

    if not args.input_dir or not args.output_dir:
        raise SystemExit("For cropping, provide --input_dir and --output_dir (or use --single_image / --learn_params).")

    # If input_dir is a leaf with images, treat it as a single album
    if _dir_has_images(args.input_dir):
        if args.verbose:
            print(f"Processing (leaf album) {args.input_dir}")
        crop_folder(args.input_dir, args.output_dir, args.size, args.size, **knobs)
        return

    # Otherwise expect subfolders (albums)
    crop_all_albums(args.input_dir, args.output_dir, args.size, args.size, **knobs)


    crop_all_albums(
        args.input_dir, args.output_dir, args.size, args.size,
        **knobs
    )

if __name__ == "__main__":
    main()
