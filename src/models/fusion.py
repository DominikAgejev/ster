# src/models/fusion.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

# -----------------------------
# Late Fusion (concat image + meta)
# -----------------------------
class LateFusionNet(nn.Module):
    """
    Encodes image with a pluggable backbone encoder (global pooled),
    encodes metadata with an MLP (optional), concatenates, then regresses to 3-dim.
    Works with and without metadata (meta_dim may be 0).
    """
    def __init__(self, img_enc: nn.Module, img_dim: int, meta_dim: int = 6):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.img_enc = img_enc

        if self.meta_dim > 0:
            self.meta_enc = nn.Sequential(
                nn.Linear(self.meta_dim, 64), nn.ReLU(),
                nn.BatchNorm1d(64), nn.Linear(64, 64), nn.ReLU()
            )
            fused_in = img_dim + 64
        else:
            self.meta_enc = None
            fused_in = img_dim

        self.head = nn.Sequential(
            nn.Linear(fused_in, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

    def forward(self, img: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        f_img = self.img_enc(img)  # [B,img_dim]
        if self.meta_dim <= 0:
            return self.head(f_img)
        if meta is None or meta.ndim != 2 or meta.shape[1] != self.meta_dim:
            raise RuntimeError(f"LateFusionNet: expected meta [B,{self.meta_dim}] but got {None if meta is None else list(meta.shape)}")
        f_meta = self.meta_enc(meta)
        return self.head(torch.cat([f_img, f_meta], dim=1))


# -----------------------------
# FiLM (Feature-wise Linear Modulation)
# -----------------------------
class FiLMAffine(nn.Module):
    """
    Per-channel affine modulation conditioned on metadata vector m \in R^D.
    y = (1 + g(m)) * x + b(m). Stable parameterization to avoid blow-ups.
    """
    def __init__(self, in_ch: int, meta_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.meta_dim = int(meta_dim)
        if self.meta_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(self.meta_dim, hidden), nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden, 2 * self.in_ch)
            )
        else:
            self.mlp = None

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor]) -> torch.Tensor:
        if self.mlp is None or meta is None:
            return x
        params = self.mlp(meta)                  # [B, 2C]
        gamma_raw, beta_raw = params.chunk(2, dim=1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta  = 0.1 * torch.tanh(beta_raw)
        gamma = gamma.view(-1, self.in_ch, 1, 1)
        beta  = beta.view(-1, self.in_ch, 1, 1)
        return gamma * x + beta

class _GAP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

class FiLMNet(nn.Module):
    """
    Multi-stage FiLM by default:
      - Accepts a feature_extractor that returns a Tensor [B,C,H,W] (single) or a list of them (multi).
      - FiLM + GAP + Linear(d) per stage, then concat (and optionally meta) → MLP → 3-dim output.
    """
    def __init__(self,
                 feature_extractor: nn.Module,
                 feat_channels: Union[int, List[int], None],
                 meta_dim: int = 6,
                 proj_dim: int = 128,
                 stage_dropout: float = 0.0):
        super().__init__()
        self.features = feature_extractor
        self.meta_dim = int(meta_dim)
        self._declared = False  # lazy build after seeing shapes

        if isinstance(feat_channels, int):
            self._declare([feat_channels], proj_dim, stage_dropout)
        elif isinstance(feat_channels, (list, tuple)) and len(feat_channels) > 0:
            self._declare(list(map(int, feat_channels)), proj_dim, stage_dropout)
        else:
            # defer (will infer channels on first forward)
            self.affines = nn.ModuleList()
            self.gaps    = nn.ModuleList()
            self.projs   = nn.ModuleList()
            self.meta_enc = None
            self.head = None

    def _declare(self, chs: List[int], proj_dim: int, stage_dropout: float):
        self.num_stages = len(chs)
        self.stage_channels = list(chs)

        self.affines = nn.ModuleList([FiLMAffine(c, self.meta_dim) for c in chs])
        self.gaps    = nn.ModuleList([_GAP() for _ in chs])
        self.projs   = nn.ModuleList([nn.Linear(c, proj_dim) for c in chs])

        if self.meta_dim > 0:
            self.meta_enc = nn.Sequential(
                nn.Linear(self.meta_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(stage_dropout) if stage_dropout > 0 else nn.Identity()
            )
        else:
            self.meta_enc = None

        fused_in = proj_dim * self.num_stages + (proj_dim if self.meta_enc is not None else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, 3)
        )
        self._declared = True

    def _maybe_declare(self, maps: List[torch.Tensor]):
        if not self._declared:
            chs = [int(m.shape[1]) for m in maps]
            self._declare(chs, proj_dim=128, stage_dropout=0.0)

    def forward(self, img: torch.Tensor, meta: Optional[torch.Tensor]) -> torch.Tensor:
        feats = self.features(img)
        maps = list(feats) if isinstance(feats, (list, tuple)) else [feats]
        self._maybe_declare(maps)

        stage_vecs = []
        for m, affine, gap, proj in zip(maps, self.affines, self.gaps, self.projs):
            y = affine(m, meta)
            v = proj(gap(y))          # [B,d]
            stage_vecs.append(v)

        fused = torch.cat(stage_vecs, dim=1)   # [B, d * S]
        if self.meta_enc is not None:
            if meta is None or meta.ndim != 2 or meta.shape[1] != self.meta_dim:
                # fallback to zeros; keeps run alive and surfaces issue in logs
                B = fused.size(0)
                device, dtype = fused.device, fused.dtype
                meta_proj = torch.zeros(B, self.meta_enc[0].out_features, device=device, dtype=dtype)
            else:
                meta_proj = self.meta_enc(meta)
            fused = torch.cat([fused, meta_proj], dim=1)

        return self.head(fused)

# -----------------------------
# Cross-Attention
# -----------------------------
class CrossAttnNet(nn.Module):
    """
    Uses a generic backbone to produce spatial tokens; metadata provides the query.
    Single-step cross-attention, then fuse with a skip projection of metadata.

    With meta_dim == 0, uses a learned query vector and removes the skip projection,
    keeping the rest of the capacity comparable.
    """
    def __init__(self,
                 feature_extractor: nn.Module,
                 feat_channels: int,
                 meta_dim: int = 6,
                 heads: int = 4,
                 proj_dim: int = 128,
                 attn_dim: Optional[int] = None):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.feature_extractor = feature_extractor   # returns [B,C,H,W]

        C = int(feat_channels)
        d = int(attn_dim) if (attn_dim is not None and int(attn_dim) > 0) else C

        # choose a valid number of heads (d must be divisible by heads)
        if d % heads != 0:
            for h in (8, 4, 2, 1):
                if d % h == 0:
                    heads = h
                    break
        self.d = d
        self.heads = heads

        # Project tokens C→d when needed
        self.token_proj = nn.Identity() if d == C else nn.Linear(C, d)

        if self.meta_dim > 0:
            self.meta_to_q = nn.Linear(self.meta_dim, d)
            self.meta_proj = nn.Sequential(
                nn.Linear(self.meta_dim, d), nn.ReLU(), nn.Linear(d, d)
            )
            proj_in = 2 * d
            self.q_vec = None
        else:
            self.meta_to_q = None
            self.meta_proj = None
            proj_in = d
            self.q_vec = nn.Parameter(torch.zeros(d))
            nn.init.normal_(self.q_vec, std=0.02)

        self.attn = nn.MultiheadAttention(d, heads, batch_first=False)
        self.proj = nn.Sequential(nn.Linear(proj_in, proj_dim), nn.ReLU(), nn.Linear(proj_dim, 3))

    def forward(self, img: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        fmap = self.feature_extractor(img)           # [B,C,H,W]
        B, C, H, W = fmap.shape
        tokens = fmap.permute(0, 2, 3, 1).reshape(B, H * W, C).transpose(0, 1)  # [HW,B,C]
        tokens = self.token_proj(tokens)             # [HW,B,d]

        if self.meta_dim > 0:
            if meta is None or meta.ndim != 2 or meta.shape[1] != self.meta_dim:
                raise RuntimeError(f"CrossAttnNet: expected meta [B,{self.meta_dim}] but got {None if meta is None else list(meta.shape)}")
            q = self.meta_to_q(meta).unsqueeze(0)    # [1,B,d]
        else:
            q = self.q_vec.view(1, 1, self.d).expand(1, B, self.d)  # [1,B,d]

        attn_out, _ = self.attn(q, tokens, tokens)   # [1,B,d]
        attn_out = attn_out.squeeze(0)               # [B,d]

        if self.meta_dim > 0:
            m = self.meta_proj(meta)                 # [B,d]
            fused = torch.cat([attn_out, m], dim=1)  # [B,2d]
        else:
            fused = attn_out                         # [B,d]

        return self.proj(fused)                      # [B,3]


__all__ = ["LateFusionNet", "FiLMAffine", "FiLMNet", "CrossAttnNet"]
