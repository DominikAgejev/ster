# src/data/transforms.py
from __future__ import annotations
import numpy as np
import torch
from skimage.color import lab2rgb


class ToTensorTransform:
    """
    Converts:
      - image: HxWx3 float32 -> 3xHxW torch.float32
      - metadata: K -> K torch.float32
      - label: 3 -> 3 torch.float32

    Optionally applies an img_norm_fn(image_chw) -> image_chw (e.g., RGB mean/std).
    """
    def __init__(self, img_norm_fn=None):
        self.img_norm_fn = img_norm_fn

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]  # HxWx3 float32
        if isinstance(img, torch.Tensor):
            image = img
        else:
            image = torch.from_numpy(img)

        if image.ndim == 3 and image.shape[-1] == 3:
            # H,W,C -> C,H,W
            image = image.permute(2, 0, 1).contiguous()

        if self.img_norm_fn is not None:
            # Images must be RGB in [0,1] when an RGB normalizer is used.
            # (LAB path never sets img_norm_fn.)
            if (image.min().item() < -1e-3) or (image.max().item() > 1.001):
                raise ValueError(
                    f"Image tensor expected in [0,1] before normalization; "
                    f"got range [{float(image.min()):.3f}, {float(image.max()):.3f}]"
                )
            image = self.img_norm_fn(image)

        meta = sample.get("metadata", None)
        if meta is None:
            metadata = torch.zeros(0, dtype=image.dtype)
        else:
            metadata = torch.from_numpy(meta) if not isinstance(meta, torch.Tensor) else meta
            metadata = metadata.to(dtype=image.dtype)

        lab = sample["label"]
        label = torch.from_numpy(lab) if not isinstance(lab, torch.Tensor) else lab
        label = label.to(dtype=image.dtype)

        out = dict(sample)
        out["image"] = image
        out["metadata"] = metadata
        out["label"] = label
        return out


def make_rgb_normalizer(mean, std):
    """
    Returns a function img_norm_fn(x: torch.Tensor[3,H,W]) -> normalized.
    Mean/std are 3-element iterables (e.g., timm preprocess).
    Assumes image is in RGB [0,1].
    """
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)

    def _fn(x: torch.Tensor) -> torch.Tensor:
        return (x - mean_t.to(x.device)) / std_t.to(x.device)
    return _fn


@torch.no_grad()
def lab_to_rgb_tensor(lab: torch.Tensor) -> torch.Tensor:
    """
    Convert Lab tensor to RGB in [0,1].
    Accepts [3,H,W] or [B,3,H,W]; returns same rank with RGB channels.

    Note: uses skimage for correctness (not the fast approximate formula).
    """
    if lab.ndim == 3:
        c, h, w = lab.shape
        assert c == 3, "Expected channels-first Lab"
        arr = lab.permute(1, 2, 0).cpu().numpy()  # H,W,3
        rgb = lab2rgb(arr).astype(np.float32)     # H,W,3 in [0,1]
        return torch.from_numpy(rgb).permute(2, 0, 1).to(lab.device)
    elif lab.ndim == 4:
        b, c, h, w = lab.shape
        assert c == 3, "Expected channels-first Lab"
        out = []
        for i in range(b):
            arr = lab[i].permute(1, 2, 0).cpu().numpy()  # H,W,3
            rgb = lab2rgb(arr).astype(np.float32)
            out.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.stack(out, dim=0).to(lab.device)
    else:
        raise ValueError("lab tensor must be [3,H,W] or [B,3,H,W]")
