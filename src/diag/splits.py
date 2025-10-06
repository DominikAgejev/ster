# src/diag/splits.py
from __future__ import annotations
from typing import Sequence, Optional, Dict, Any
from collections import Counter
import os

class SplitsTools:
    def __init__(self, state, log):
        self.s, self.log = state, log

    def filters(self, excluded: Sequence[str], included: Sequence[str]):
        if not self.s.enabled: return
        self.log.write("folder_filters", excluded=list(excluded), included=list(included))

    def report_counts(self, n_train: int, n_val: int, n_test: Optional[int]):
        if not self.s.enabled: return
        self.log.write("split_counts", train=int(n_train), val=int(n_val), test=(int(n_test) if n_test is not None else None))

    def overlap_check(self, image_paths: Sequence[str], train_idx: Sequence[int], val_idx: Sequence[int], test_idx: Optional[Sequence[int]] = None, hard_assert: bool = True):
        if not self.s.enabled: return
        def _subset(idx): return {image_paths[i] for i in (idx or [])}
        tr, va = _subset(train_idx), _subset(val_idx)
        te = _subset(test_idx) if test_idx is not None else set()
        inter_tv = tr & va
        inter_tt = tr & te
        inter_vt = va & te
        if inter_tv or inter_tt or inter_vt:
            self.log.write("split_overlap",
                           train_val=len(inter_tv), train_test=len(inter_tt), val_test=len(inter_vt))
            if hard_assert:
                raise AssertionError("[split] overlap detected (train/val/test).")
        else:
            self.log.write("split_overlap_ok", note="no duplicates across splits")

    def folder_hist(self, image_paths: Sequence[str], idx: Sequence[int], name: str):
        """Count by immediate parent folder to sanity-check filters and distribution."""
        if not self.s.enabled: return
        def parent(p): 
            try:
                return os.path.basename(os.path.dirname(p))
            except Exception:
                return "<na>"
        counter = Counter(parent(image_paths[i]) for i in idx or [])
        self.log.write("folder_hist", split=name, counts=dict(counter))

    def group_overlap(self, metadf, train_idx: Sequence[int], val_idx: Sequence[int], group_key: Optional[str]):
        """If a group column is set (e.g., 'class' or device), verify no leakage."""
        if not self.s.enabled or not group_key: return
        if group_key not in metadf.columns:
            self.log.write("group_overlap_skip", reason=f"column '{group_key}' missing")
            return
        tr = set(metadf.loc[train_idx, group_key].tolist())
        va = set(metadf.loc[val_idx, group_key].tolist())
        inter = tr & va
        if inter:
            self.log.write("group_overlap", key=group_key, n=len(inter), values=sorted(list(inter))[:20])
        else:
            self.log.write("group_overlap_ok", key=group_key)
