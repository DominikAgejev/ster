# src/diag/images.py
from __future__ import annotations
from typing import Optional, Dict, Any
import torch

class ImageTools:
    def __init__(self, state, log):
        self.s, self.log = state, log

    def transform_report(self, *, color_space: str, pretrained: bool, backbone: str, rgb_mean=None, rgb_std=None):
        if not self.s.enabled: return
        self.log.write("image_transform",
                       color_space=color_space, pretrained=bool(pretrained), backbone=backbone,
                       rgb_mean=(list(rgb_mean) if rgb_mean is not None else None),
                       rgb_std=(list(rgb_std) if rgb_std is not None else None))

    def sample_stats(self, dataset, n: int = 8):
        """Quick sanity on a few transformed images (min/max/mean/std). Safe and tiny."""
        if not self.s.enabled or n <= 0: return
        n = min(n, len(dataset))
        mins, maxs, means, stds = [], [], [], []
        for i in range(n):
            sample = dataset[i]
            x = sample["image"] if isinstance(sample, dict) else sample[0]
            if not torch.is_tensor(x): continue
            vals = x.float().view(-1)
            mins.append(float(torch.nan_to_num(vals).min()))
            maxs.append(float(torch.nan_to_num(vals).max()))
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
        self.log.write("image_sample_stats",
                       n=n,
                       min=(min(mins) if mins else None),
                       max=(max(maxs) if maxs else None),
                       mean=(sum(means)/len(means) if means else None),
                       std=(sum(stds)/len(stds) if stds else None))
