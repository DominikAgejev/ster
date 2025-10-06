# src/diag/timing.py
from __future__ import annotations
import time
from contextlib import contextmanager


@contextmanager
def section(name: str, logger=None):
    """
    Usage:
      with D.timing.section("forward", logger=D.log):
          preds = model(...)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if logger is not None:
            logger.write("timing", section=name, ms=dt_ms)
        else:
            print(f"[timing] {name}: {dt_ms:.1f} ms")
