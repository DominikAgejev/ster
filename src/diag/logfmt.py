# src/diag/logfmt.py
from __future__ import annotations
import json, time, os, traceback
from typing import Any, Dict

# Events that should *not* echo to stdout by default
QUIET_EVENTS = {
    "batch_stats",
    "forward_ok",
    "seq_len",
    "norm_params",
    "meta_cols",
    "folder_hist",
    "split_counts",
    "image_transform",
    "gpu",
    "config",
}

def _safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        s = str(obj)
        return s if len(s) <= 1000 else s[:1000] + "...<trunc>"

class JsonlLogger:
    def __init__(self, state):
        self.s = state

    def _open(self):
        if not self.s.enabled:
            return None
        if self.s._fh is None:
            path = self.s.jsonl_path or os.path.join(self.s.outdir or ".", "diag.jsonl")
            self.s._fh = open(path, "a", encoding="utf-8")
        return self.s._fh

    def write(self, event: str, **fields: Dict[str, Any]):
        """
        Write JSONL always. Echo to stdout only if:
          - event NOT in QUIET_EVENTS, AND
          - fields.get("echo", True) is True.
        Callers can force silence by passing echo=False.
        """
        if not self.s.enabled:
            return
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event}
        rec.update({k: _safe(v) for k, v in fields.items() if k != "echo"})
        fh = self._open()
        if fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

        # Only echo interesting events
        echo = bool(fields.get("echo", True))
        if echo and event not in QUIET_EVENTS:
            print(f"[diag] {event}: " + ", ".join(f"{k}={_brief(v)}" for k, v in fields.items() if k != "echo"))

    def crash_dump(self, exc: BaseException, context: Dict[str, Any] | None = None):
        if not self.s.enabled:
            return
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
            "traceback": traceback.format_exc(limit=200),
            "context": {k: _safe(v) for k, v in (context or {}).items()},
        }
        path = os.path.join(self.s.outdir or ".", "diag_crash.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # crashes should always echo
        self.write("crash", where=(context or {}).get("where"), exc=payload["exc_msg"], echo=True)

def _brief(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 120 else s[:117] + "..."
