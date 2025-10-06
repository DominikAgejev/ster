# src/diag/__init__.py
from __future__ import annotations

from .config import DiagState
from .logfmt import JsonlLogger
from .validate import Validator
from .timing import section
from .oom import guard
from .hooks import grad_stats
from .normalize import NormTools
from .splits import SplitsTools
from .images import ImageTools
from .runtime import maybe_step_diag

class _Facade:
    def __init__(self):
        self.state = DiagState()
        self.log = JsonlLogger(self.state)
        self.validate = Validator(self.state, self.log)
        self.norm = NormTools(self.state, self.log)
        self.splits = SplitsTools(self.state, self.log)
        self.images = ImageTools(self.state, self.log)
        self.timing = type("Timing", (), {"section": section})
        self.oom = type("OOM", (), {"guard": guard})
        self.hooks = type("Hooks", (), {"grad_stats": grad_stats})
        self.runtime = type("Runtime", (), {"maybe_step_diag": maybe_step_diag})

    def enable(self, run, outdir: str, level: str | None = None):
        self.state.enable(outdir=outdir, level=level)
        self.log.write("diag_start",
                       model=getattr(getattr(run, "model", None), "model", None),
                       backbone=getattr(getattr(run, "model", None), "backbone", None),
                       features=getattr(getattr(run, "data", None), "features", None),
                       meta_encoder=getattr(getattr(run, "data", None), "meta_encoder", None))

    def config_snapshot(self, run):
        if not self.state.enabled: return
        def _flatten_dc(dc):
            try:
                return {k: (str(v) if len(str(v)) > 400 else v) for k, v in vars(dc).items()}
            except Exception:
                return str(dc)
        payload = {}
        for name in ("data","model","loss","loop","ckpt","early","smoke"):
            obj = getattr(run, name, None)
            if obj is not None:
                payload[name] = _flatten_dc(obj)
        self.log.write("config", **payload)

D = _Facade()
