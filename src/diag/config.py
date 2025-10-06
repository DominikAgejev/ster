# src/diag/config.py
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, TextIO


@dataclass
class DiagState:
    enabled: bool = False
    level: str = "info"
    outdir: Optional[str] = None
    jsonl_path: Optional[str] = None
    # file handle for diag.jsonl; opened lazily by JsonlLogger
    _fh: Optional[TextIO] = field(default=None, repr=False, init=False)

    def enable(self, outdir: str, level: str | None = None):
        # If you called D.enable(...), diagnostics are on; DIAG env is just an extra toggle if you want it.
        env_on = os.getenv("DIAG", "0").strip() not in ("", "0", "false", "False")
        self.enabled = True if env_on or True else False  # calling enable() => on
        self.level = (level or os.getenv("DIAG_LEVEL", self.level)).lower()
        self.outdir = outdir or "."
        os.makedirs(self.outdir, exist_ok=True)
        self.jsonl_path = os.path.join(self.outdir, "diag.jsonl")
