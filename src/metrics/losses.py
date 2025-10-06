# src/metrics/losses.py
import torch
import torch.nn.functional as F

def deltaE2000_loss(pred_lab, target_lab, reduction="mean"):
    L1, a1, b1 = pred_lab[:,0], pred_lab[:,1], pred_lab[:,2]
    L2, a2, b2 = target_lab[:,0], target_lab[:,1], target_lab[:,2]

    # ΔL'
    dL = L1 - L2
    L_ = 0.5 * (L1 + L2)

    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    C_ = 0.5 * (C1 + C2)

    G = 0.5 * (1 - torch.sqrt((C_**7) / (C_**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = torch.sqrt(a1p**2 + b1**2)
    C2p = torch.sqrt(a2p**2 + b2**2)
    Cp_ = 0.5 * (C1p + C2p)

    h1p = torch.atan2(b1, a1p)
    h2p = torch.atan2(b2, a2p)
    h1p = torch.where(h1p < 0, h1p + 2*torch.pi, h1p)
    h2p = torch.where(h2p < 0, h2p + 2*torch.pi, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp > torch.pi, dhp - 2*torch.pi, dhp)
    dhp = torch.where(dhp < -torch.pi, dhp + 2*torch.pi, dhp)
    dhp = torch.where((C1p*C2p)==0, torch.zeros_like(dhp), dhp)
    dHp = 2*torch.sqrt(C1p*C2p) * torch.sin(dhp/2)

    Lp_ = (L1 + L2)/2
    Cp__ = (C1p + C2p)/2

    hp_ = (h1p + h2p)/2
    hp_ = torch.where(torch.abs(h1p-h2p) > torch.pi, hp_ - torch.pi, hp_)
    hp_ = torch.where(hp_ < 0, hp_ + 2*torch.pi, hp_)
    hp_ = torch.where((C1p*C2p)==0, h1p+h2p, hp_)

    T = 1 - 0.17*torch.cos(hp_-torch.pi/6) + \
            0.24*torch.cos(2*hp_) + \
            0.32*torch.cos(3*hp_+torch.pi/30) - \
            0.20*torch.cos(4*hp_-63*torch.pi/180)

    dRo = 30*torch.pi/180 * torch.exp(- ((180/torch.pi*hp_-275)/25)**2 )
    Rc = 2*torch.sqrt((Cp__**7)/(Cp__**7 + 25**7))
    Sl = 1 + (0.015*(Lp_-50)**2)/torch.sqrt(20+(Lp_-50)**2)
    Sc = 1 + 0.045*Cp__
    Sh = 1 + 0.015*Cp__*T
    Rt = -torch.sin(2*dRo)*Rc

    dE = torch.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt*(dCp/Sc)*(dHp/Sh))

    if reduction=="mean":
        return dE.mean()
    elif reduction=="sum":
        return dE.sum()
    else:
        return dE

# -------- RGB -> Lab (differentiable, D65) ----------
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    # x in [0,1]; piecewise gamma inverse
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).clamp(min=0) ** 2.4
    )

def _rgb_to_xyz(linear_rgb: torch.Tensor) -> torch.Tensor:
    # sRGB (D65) matrix
    # shape: [..., 3]
    M = linear_rgb.new_tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return linear_rgb @ M.T  # [...,3]

def _f_xyz(t: torch.Tensor) -> torch.Tensor:
    eps = 216/24389  # (6/29)^3
    kappa = 24389/27
    return torch.where(
        t > eps,
        t.pow(1/3),
        (kappa * t + 16) / 116
    )

def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: [B,3] in [0,1] (unconstrained inputs will be clamped)
    returns Lab: [B,3] with L in [0,100], a*, b* roughly [-110,110]
    """
    # ensure last dim = 3
    assert rgb.shape[-1] == 3, f"Expected [...,3], got {rgb.shape}"
    rgb01 = rgb.clamp(0.0, 1.0)
    lin = _srgb_to_linear(rgb01)
    xyz = _rgb_to_xyz(lin)

    # D65 white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    fx, fy, fz = _f_xyz(x), _f_xyz(y), _f_xyz(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)

# --- Robust ΔE utilities -----------------------------------------------------
def _huber(r: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    # r: residuals (ΔE per-sample). Returns robust penalty.
    delta = float(delta)
    abs_r = r.abs()
    quad = 0.5 * (r ** 2) / delta
    lin  = abs_r - 0.5 * delta
    return torch.where(abs_r <= delta, quad, delta * lin)

def _charbonnier(r: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # Smooth-L1-ish; strictly convex and differentiable at 0
    return torch.sqrt(r * r + eps * eps)

def _apply_robust(kind: str | None, r: torch.Tensor, huber_delta: float, charb_eps: float) -> torch.Tensor:
    if not kind or kind == "none":
        return r  # identity
    k = str(kind).lower()
    if k.startswith("hub"):
        return _huber(r, delta=huber_delta)
    if k.startswith("charb") or k.startswith("cauchy"):  # allow "charbonnier", "charb"
        return _charbonnier(r, eps=charb_eps)
    raise ValueError(f"Unknown robust kind: {kind!r}")

# ===== ColorLoss with ΔE00 + annealed MSE (RGB or Lab) =====
class ColorLoss:
    """
    Computes ΔE00 in Lab (with optional small smoothing near 0), then
    optionally robustifies per-sample ΔE via Huber/Charbonnier *before*
    reduction ('mean' or 'median'). Blends with MSE in 'rgb' or 'lab'
    using the externally-controlled `mse_weight` (your train loop can
    schedule it via loss_cfg.mse_weight_start/mse_weight_epochs).

    Parameters
    ----------
    input_space : {'lab','rgb'}
        Space of model outputs/targets; rgb will be converted to Lab internally.
    mse_space : {'rgb','lab','same'}
        Space for the MSE term when blended.
    mse_weight : float
        Blend weight for MSE (0.0 disables MSE term).
    smooth_eps : float
        Apply sqrt(ΔE^2 + eps) per-sample (stabilizes tiny residuals).
    robust : {'none','huber','charbonnier'}
        Robustifier for per-sample ΔE before reduction (default: 'none').
    huber_delta : float
        Transition point for Huber (≈1.0 is a good start).
    charb_eps : float
        Charbonnier epsilon (≈1e-3..1e-2).
    de_reduce : {'mean','median'}
        Reduction for robustified ΔE when forming the loss.
    """
    def __init__(
        self,
        input_space: str = "lab",
        mse_space: str = "rgb",
        mse_weight: float = 1.0,
        smooth_eps: float = 1e-6,
        *,
        robust: str = "none",
        huber_delta: float = 1.0,
        charb_eps: float = 1e-3,
        de_reduce: str = "mean",
    ):
        self.input_space = input_space.lower().strip()
        self.mse_space   = mse_space.lower().strip()
        self.mse_weight  = float(mse_weight)
        self.smooth_eps  = float(smooth_eps)

        self.robust      = (robust or "none").lower()
        self.huber_delta = float(huber_delta)
        self.charb_eps   = float(charb_eps)
        self.de_reduce   = de_reduce.lower().strip()

    # --- helpers ---
    def _to_lab(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_space == "rgb":
            return rgb_to_lab_torch(x)
        return x

    def _mse_rgb(self, pred: torch.Tensor, tgt: torch.Tensor):
        if self.input_space != "rgb":
            return None
        return F.mse_loss(pred.clamp(0, 1), tgt.clamp(0, 1))

    def _mse_lab(self, pred_lab: torch.Tensor, tgt_lab: torch.Tensor):
        return F.mse_loss(pred_lab, tgt_lab)

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor):
        """
        Returns: (loss_tensor, metrics_dict)
        metrics = {'de00', 'de00_median', 'mse_lab', 'mse_rgb', 'loss'}
        """
        pred_lab = self._to_lab(pred)
        tgt_lab  = self._to_lab(tgt)

        # ΔE00 per-sample (vector)
        de_vec = deltaE2000_loss(pred_lab, tgt_lab, reduction="none")  # [B]
        if self.smooth_eps > 0.0:
            de_vec = torch.sqrt(de_vec * de_vec + self.smooth_eps)

        # Robustify per-sample ΔE then reduce for the loss term
        de_rob = _apply_robust(self.robust, de_vec, self.huber_delta, self.charb_eps)
        if self.de_reduce == "median":
            de_loss = de_rob.median()
        else:
            de_loss = de_rob.mean()

        # MSEs (mean)
        mse_lab = self._mse_lab(pred_lab, tgt_lab)
        mse_rgb = self._mse_rgb(pred, tgt)  # may be None

        # Blend
        if self.mse_weight > 0.0:
            if self.mse_space in ("same", "lab"):
                mse_term = mse_lab
            elif self.mse_space == "rgb":
                mse_term = mse_rgb if mse_rgb is not None else mse_lab
            else:
                raise ValueError("mse_space must be 'rgb', 'lab', or 'same'.")
            loss = (1.0 - self.mse_weight) * de_loss + self.mse_weight * mse_term
        else:
            loss = de_loss

        # Report plain (non-robust) mean and median for visibility/consistency
        metrics = {
            "de00": float(de_vec.mean().detach().cpu()),
            "de00_median": float(de_vec.median().detach().cpu()),
            "mse_lab": float(mse_lab.detach().cpu()) if mse_lab is not None else None,
            "mse_rgb": float(mse_rgb.detach().cpu()) if mse_rgb is not None else None,
            "loss": float(loss.detach().cpu()),
        }
        return loss, metrics
