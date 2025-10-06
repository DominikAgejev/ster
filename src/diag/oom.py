# src/diag/oom.py
from __future__ import annotations
import torch
from contextlib import contextmanager


@contextmanager
def guard(stage: str):
    """
    Catch CUDA OOM to print helpful context, then re-raise.
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"[oom] at {stage}: allocated={alloc:.1f}MB reserved={reserved:.1f}MB "
                  f"(suggest: lower batch_size / meta_max_len / d_model or use grad accumulation)")
        except Exception:
            pass
        raise
