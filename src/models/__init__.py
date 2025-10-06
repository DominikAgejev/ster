"""
Model registry and factory.

Usage in train.py:
    from models import build_model
    model = build_model(args.model, backbone=args.backbone, meta_dim=D_meta)
"""
from typing import Dict, Type
import torch.nn as nn

from .backbones import build_backbone, build_backbone_bundle
from .fusion import LateFusionNet, FiLMNet, CrossAttnNet


# Canonical names -> classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "late": LateFusionNet,
    "film": FiLMNet,
    "xattn": CrossAttnNet,
}

# Common aliases -> canonical names
ALIASES: Dict[str, str] = {
    "latefusion": "late",
    "concat": "late",
    "filmnet": "film",
    "filmmod": "film",
    "cross": "xattn",
    "crossattn": "xattn",
}


def _canon(name: str) -> str:
    n = name.lower().strip()
    return ALIASES.get(n, n)


from .backbones import build_backbone_bundle
from .fusion import LateFusionNet, FiLMNet, CrossAttnNet

def build_model(name: str, backbone: str = "smallcnn", meta_dim: int = 6, **kwargs):
    key = _canon(name)
    # Plumb backbone-only knobs; don't forward them to model constructors
    _bb_kwargs = {}
    for k in ("token_stage", "pretrained"):
        if k in kwargs:
            _bb_kwargs[k] = kwargs.pop(k)
    bb = build_backbone_bundle(backbone, **_bb_kwargs)

    if key == "late":
        # Uses pooled vector
        return LateFusionNet(
            img_enc=bb.global_encoder,   # nn.Module: img -> [B,D]
            img_dim=bb.out_dim,          # D
            meta_dim=meta_dim,
            **kwargs
        )
    elif key == "film":
        # Uses spatial fmap for FiLM
        return FiLMNet(
            feature_extractor=bb.feature_extractor,  # img -> [B,C,H,W]
            feat_channels=bb.feat_channels,          # C
            meta_dim=meta_dim,
            **kwargs
        )
    elif key == "xattn":
        # Uses spatial fmap as tokens
        return CrossAttnNet(
            feature_extractor=bb.feature_extractor,  # img -> [B,C,H,W]
            feat_channels=bb.feat_channels,          # C
            meta_dim=meta_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {name}")



def available_models() -> str:
    mains = ", ".join(sorted(MODEL_REGISTRY.keys()))
    aliases = ", ".join(sorted(ALIASES.keys()))
    return f"canonical: [{mains}] | aliases: [{aliases}]"


__all__ = [
    "LateFusionNet",
    "FiLMNet",
    "CrossAttnNet",
    "build_model",
    "available_models",
]
