# src/models/backbones.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn

try:
    import timm
    _HAVE_TIMM = True
except Exception:
    timm = None
    _HAVE_TIMM = False

ALIASES = {
    "resnet": "resnet18",
    "convnext": "convnext_tiny",
    "efficientnet": "efficientnet_b0",
    "efficientnetv2": "tf_efficientnetv2_s",
    "mobilenet": "mobilenet_v3_large"
}

TIMM_EQUIV = {
    "resnet18": "resnet18",
    "convnext_tiny": "convnext_tiny",
    "efficientnet_b0": "efficientnet_b0",
    "tf_efficientnetv2_s": "tf_efficientnetv2_s",
    "mobilenet_v3_large": "mobilenetv3_large_100"
}

def normalize_backbone_name(name: Optional[str]) -> str:
    """
    Canonicalize a user/back-compat backbone alias to the project’s canonical key.
    Examples: 'resnet'→'resnet18', 'mobilenet'→'mobilenet_v3_large'.
    """
    if not isinstance(name, str):
        return "smallcnn"
    n = name.lower().strip()
    return ALIASES.get(n, n)

def resolve_preprocess(name: Optional[str]):
    default = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
    if not isinstance(name, str) or name.lower().strip() in ("", "smallcnn", "small_cnn"):
        return default
    if not _HAVE_TIMM:
        return default
    key = normalize_backbone_name(name)
    timm_name = TIMM_EQUIV.get(key, key)
    try:
        # New: use data config resolver instead of creating the full model
        from timm.data import resolve_data_config
        cfg = resolve_data_config({}, model=timm_name)  # cheap
        mean = cfg.get("mean", default["mean"]); std = cfg.get("std", default["std"])
        return {"mean": tuple(map(float, mean)), "std": tuple(map(float, std))}
    except Exception:
        return default

# -----------------------------
# SmallCNN (kept as before)
# -----------------------------
class SmallCNN(nn.Module):
    """
    Small image encoder used by fusion models.
    Outputs a 256-dim embedding via global average pooling.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 64x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256),nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim  # keep for heads
        assert self.out_dim == 256, "SmallCNN currently fixed to 256-d output."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                 # [B,256,H',W']
        x = self.pool(x).flatten(1)      # [B,256]
        return x


class SmallCNNFeatures(nn.Module):
    """Return the spatial feature map from SmallCNN (pre-pooling)."""
    def __init__(self, base: SmallCNN):
        super().__init__()
        self.conv = base.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B,256,H',W']


# -----------------------------
# Backbone bundle & registry
# -----------------------------
@dataclass
class BackboneBundle:
    # For Late Fusion (global embedding)
    global_encoder: nn.Module
    out_dim: int
    # For FiLM / CrossAttn (spatial features)
    feature_extractor: nn.Module
    feat_channels: Union[int, List[int]]
    preprocess: Optional[Dict[str, Tuple[float, float, float]]] = None  # mean/std
    token_stride: Optional[Union[int, List[int]]] = None                 # reduction(s) per selected stage(s)
    name: str = ""


BACKBONES: Dict[str, Callable[..., BackboneBundle]] = {}


def register_backbone(name: str):
    def deco(fn: Callable[..., BackboneBundle]):
        BACKBONES[name] = fn
        return fn
    return deco


@register_backbone("smallcnn")
def _make_smallcnn(**kw) -> BackboneBundle:
    enc = SmallCNN()
    feat = SmallCNNFeatures(enc)
    return BackboneBundle(global_encoder=enc, out_dim=256,
                          feature_extractor=feat, feat_channels=256)

def _require_timm(name: str):
    if not _HAVE_TIMM:
        raise RuntimeError(
            f"Backbone '{name}' requires timm. Please `pip install timm`."
        )

def _timm_features_last(model) -> nn.Module:
    """
    Wrap a timm features_only model so forward(x) -> last feature map [B,C,H,W].
    """
    class _LastFeature(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            outs = self.m(x)
            return outs[-1] if isinstance(outs, (list, tuple)) else outs
    return _LastFeature(model)

def _timm_features_list(model) -> nn.Module:
    """
    Wrap a timm features_only model so forward(x) -> List[Tensor] (multi-stage).
    """
    class _ListFeatures(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            outs = self.m(x)
            return list(outs) if isinstance(outs, (list, tuple)) else [outs]
    return _ListFeatures(model)

def _timm_features_pick(model, picks: List[int], multi: bool) -> nn.Module:
    """
    Forward returns selected feature maps in order.
    - multi=True  -> List[Tensor]
    - multi=False -> single Tensor
    """
    class _Pick(nn.Module):
        def __init__(self, m, idx): super().__init__(); self.m = m; self.idx = list(idx)
        def forward(self, x):
            outs = self.m(x)
            outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
            sel = [outs[i] for i in self.idx]
            return sel if len(sel) > 1 and multi else sel[0]
    return _Pick(model, picks)

def _default_multistage_indices(backbone_name: str) -> Tuple[int, ...]:
    """
    Reasonable multi-stage defaults per backbone family.
    - ResNet: use layer2, layer3, layer4 (skip earliest high-res stage)
    - Others: keep single last stage by default (safe/back-compat)
    """
    name = (backbone_name or "").lower()
    if "resnet" in name:
        return (1, 2, 3)  # timm feature indices for l2, l3, l4
    return (3,)  # last stage only

@register_backbone("resnet18")
def _make_resnet18(pretrained: bool = True, token_stage: Optional[Union[int, List[int], str]] = None, **kw) -> BackboneBundle:
    return _make_timm_bundle("resnet18", token_stage=token_stage, pretrained=pretrained, **kw)

@register_backbone("convnext_tiny")
def _make_convnext_tiny(pretrained: bool = True, token_stage: Optional[Union[int, List[int], str]] = None, **kw) -> BackboneBundle:
    return _make_timm_bundle("convnext_tiny", token_stage=token_stage, pretrained=pretrained, **kw)

@register_backbone("efficientnet_b0")
def _make_efficientnet_b0(pretrained: bool = True, token_stage: Optional[Union[int, List[int], str]] = None, **kw) -> BackboneBundle:
    return _make_timm_bundle("efficientnet_b0", token_stage=token_stage, pretrained=pretrained, **kw)

@register_backbone("tf_efficientnetv2_s")
def _make_tf_efficientnetv2_s(pretrained: bool = True, token_stage: Optional[Union[int, List[int], str]] = None, **kw) -> BackboneBundle:
    return _make_timm_bundle("tf_efficientnetv2_s", token_stage=token_stage, pretrained=pretrained, **kw)

@register_backbone("mobilenet_v3_large")
def _make_mobilenet_v3_large(pretrained: bool = True, token_stage: Optional[Union[int, List[int], str]] = None, **kw) -> BackboneBundle:
    return _make_timm_bundle("mobilenetv3_large_100", token_stage=token_stage, pretrained=pretrained, **kw)


def build_backbone_bundle(name: str, **kw) -> BackboneBundle:
    name = name.lower().strip()
    name = normalize_backbone_name(name)
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {sorted(BACKBONES.keys())}")
    return BACKBONES[name](**kw)


# Back-compat helper (returns global encoder only)
def build_backbone(name: str, **kw) -> Tuple[nn.Module, int]:
    b = build_backbone_bundle(name, **kw)
    return b.global_encoder, b.out_dim

def _make_timm_bundle(model_name: str,
                      token_stage: Optional[Union[int, Tuple[int, ...], List[int], str]] = None,
                      pretrained: bool = True,
                      pool: str = "avg",
                      **kw) -> BackboneBundle:
     _require_timm(model_name)
     # Global head: pooled embedding
     enc = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool=pool)
     out_dim = enc.num_features

     # Spatial features (build with ALL default hooks; we'll pick indices ourselves)
     feats_all = timm.create_model(model_name, pretrained=pretrained, features_only=True)
     fi_all = feats_all.feature_info
     all_ch  = list(fi_all.channels())
     all_red = list(fi_all.reduction())
     N = len(all_ch)

     # Resolve desired indices:
     if isinstance(token_stage, (list, tuple)):
         desired = [int(i) for i in token_stage]
     elif token_stage in (None, "auto", "multi", "ms"):
         desired = list(_default_multistage_indices(model_name))
     else:
         desired = [int(token_stage)]

     # Normalize negatives and validate in range
     picks = []
     for i in desired:
         j = i if i >= 0 else N + i
         if j < 0 or j >= N:
             raise IndexError(f"[backbone:{model_name}] token_stage index {i} maps to {j} out of range [0,{N-1}]. "
                              f"Available stages: 0..{N-1}.")
         picks.append(j)

     # Build selector wrapper and per-stage metadata
     multi = len(picks) > 1
     feat_extractor = _timm_features_pick(feats_all, picks, multi=multi)
     ch_list  = [all_ch[j]  for j in picks]
     red_list = [all_red[j] for j in picks]
     feat_channels = ch_list if multi else ch_list[0]
     stride        = red_list if multi else red_list[0]
     prep = resolve_preprocess(model_name)
     return BackboneBundle(
         global_encoder=enc,
         out_dim=out_dim,
         feature_extractor=feat_extractor,
         feat_channels=feat_channels,
         preprocess=prep,
         token_stride=stride,
         name=model_name,
     )

__all__ = [
    "SmallCNN",
    "BackboneBundle",
    "BACKBONES",
    "build_backbone_bundle",
    "build_backbone",
    "resolve_preprocess",
    "normalize_backbone_name",
]

