# src/analysis/tensorboard.py
from __future__ import annotations
import traceback
import os, json, shutil
from datetime import datetime
from typing import Iterable, List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from ..data.transforms import lab_to_rgb_tensor

from math import isnan

# --------------------
# Observer interfaces
# --------------------
class Observer:
    def run_start(self, config: Dict[str, Any]) -> None: ...
    def epoch_end(self, train_log: Dict[str, float], val_log: Dict[str, float],
                  epoch: int, lr: float) -> None: ...
    def log_images(self, model, val_loader, device, epoch: int) -> None: ...
    def run_end(self, final_metrics: Dict[str, float]) -> None: ...
    def interrupt(self) -> None: ...
    def prune(self) -> None: ...

# --------------------
# Utils
# --------------------
def _coerce_tb_value(v):
    """
    Make a value safe for TensorBoard hparams:
    return a plain Python {int,float,str,bool} (or a string fallback).
    """
    # Already safe
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    # NumPy scalars -> Python scalars
    if isinstance(v, np.generic):
        return v.item()
    # Torch bits
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.item() if v.ndim == 0 else str(tuple(v.shape))
        if isinstance(v, (torch.device, torch.dtype)):
            return str(v)
    except Exception:
        pass
    # Containers / everything else -> string
    try:
        json.dumps(v)
        # Even if JSON-serializable, TB hparams want scalars, so stringify containers
        if isinstance(v, (list, tuple, dict)):
            return json.dumps(v)
        return v
    except TypeError:
        return str(v)

def _to_hparam_safe_dict(d: dict) -> dict:
    return {k: _coerce_tb_value(v) for k, v in d.items() if not str(k).startswith("_")}

def _ensure_writer(run_dir: str, writer: Optional[SummaryWriter]) -> SummaryWriter:
    if writer is None:
        os.makedirs(run_dir, exist_ok=True)
        return SummaryWriter(log_dir=run_dir)
    return writer

# --------------------
# TensorBoard logger
# --------------------
class TensorBoardLogger(Observer):
    """
    Observer that logs:
      - Scalars every epoch
      - RGB grids every N epochs
      - Rich HParams at run_end (so the HParams tab is usable for comparisons)
    """
    TAGS = {
        "loss/train": "Loss/train",
        "de/train":   "ΔE00/train",
        "mse_lab/train": "MSE/Lab_train",
        "mse_rgb/train": "MSE/RGB_train",
        "de/val":     "ΔE00/val",
        "mse_lab/val":  "MSE/Lab_val",
        "mse_rgb/val":  "MSE/RGB_val",
        "de/test":      "ΔE00/test",
        "mse_lab/test": "MSE/Lab_test",
        "mse_rgb/test": "MSE/RGB_test",
        "de/val_best":  "ΔE00/val_best",
        "lr":         "opt/LR",
        "images":     "val/input_gt_pred",
        "config":     "run/config_json",
        "meta":       "meta/columns",
        "hparams":    "hparams",
    }

    def __init__(self, logdir: str, model_name: str, meta_columns: Iterable[str],
                 min_log_epoch: int = 5):
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._run_dir = os.path.join(logdir, run_name)
        self._writer: Optional[SummaryWriter] = None
        self._hparams: Optional[Dict[str, Any]] = None
        self._activated_epoch: Optional[int] = None
        self._last_epoch_seen = 0
        self.min_log_epoch = int(min_log_epoch)
        self.meta_columns = list(meta_columns)
        self._wrote_config_once = False

        # config bits we use in image logging
        self._color_space: str = "lab"
        self._pred_activation: str = "none"
        self._activation_eps: float = 1e-3

    def _activate_if_needed(self, epoch: int):
        self._last_epoch_seen = epoch
        if self._writer is None and epoch >= self.min_log_epoch:
            self._writer = _ensure_writer(self._run_dir, self._writer)

    def _w(self) -> Optional[SummaryWriter]:
        return self._writer

    # ---- Observer API ----
    def run_start(self, config: Dict[str, Any]) -> None:
        # Rich HParams: include backbone, features, color space, activations, etc.
        # This is what populates the HParams "hyperparameters" columns.
        keep = {
            "model", "backbone", "features", "color_space",
            "pred_activation", "activation_eps",
            "mse_space", "mse_weight_start", "mse_weight_epochs",
            "lr", "weight_decay", "batch_size", "epochs",
            "val_split", "grad_clip", "workers", "seed",
            "pretrained", "token_stage", "group_split",
            "train_size", "val_size", "device", "excluded_folders", "included_folders",
            "meta_encoder", "meta_model_name", "meta_layers", "meta_text_template",
        }
        cfg = {k: config.get(k) for k in keep if k in config}
        self._hparams = _to_hparam_safe_dict(cfg)

        # needed later for image logging behavior
        self._color_space = str(config.get("color_space", "lab")).lower().strip()
        self._pred_activation = str(config.get("pred_activation", "none")).lower().strip()
        self._activation_eps = float(config.get("activation_eps", 1e-3))

    def _add_scalar_safe(self, tag: str, value, step: int) -> None:
        """Write a scalar only if value is a real number."""
        w = self._w()
        if w is None:
            return
        if value is None:
            return
        try:
            # convert numpy/tensor scalars to python floats
            if hasattr(value, "item"):
                value = value.item()
            # filter NaN
            if isinstance(value, float) and isnan(value):
                return
            w.add_scalar(tag, value, step)
        except Exception:
            # don't let TB issues crash the run
            return

    def epoch_end(self, train_log: Dict[str, float], val_log: Dict[str, float],
                  epoch: int, lr: float) -> None:
        self._activate_if_needed(epoch)
        if not self._w():
            return

        # Train metrics
        self._add_scalar_safe(self.TAGS["loss/train"], train_log.get("loss"), epoch)
        self._add_scalar_safe(self.TAGS["de/train"],   train_log.get("de00"), epoch)
        self._add_scalar_safe(self.TAGS["lr"],         lr,                    epoch)

        # Val metrics (may be missing when val_size==0)
        self._add_scalar_safe(self.TAGS["de/val"], val_log.get("de00") if val_log else None, epoch)

        # richer breakdown
        for split, log in (("train", train_log), ("val", val_log or {})):
            for key, tag in (("mse_lab", f"mse_lab/{split}"), ("mse_rgb", f"mse_rgb/{split}")):
                self._add_scalar_safe(self.TAGS[tag], log.get(key), epoch)

    def log_images(self, model, val_loader, device, epoch: int,
                   max_batches: int = 1, max_images: int = 8) -> None:
        self._activate_if_needed(epoch)
        if not self._w(): return
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_batches: break
                imgs = batch["image"][:max_images].to(device)
                meta = batch["metadata"][:max_images].to(device)
                labels = batch["label"][:max_images].to(device)
                preds = model(imgs, meta)  # [B,3]

                # mirror activation from training when color_space=rgb
                if self._color_space == "rgb":
                    if self._pred_activation == "sigmoid":
                        preds = torch.sigmoid(preds).clamp(self._activation_eps, 1 - self._activation_eps)
                    elif self._pred_activation == "sigmoid_eps":
                        s = torch.sigmoid(preds)
                        preds = self._activation_eps + (1 - 2 * self._activation_eps) * s
                    elif self._pred_activation == "tanh01":
                        y = 0.5 * (torch.tanh(preds) + 1.0)
                        preds = y.clamp(self._activation_eps, 1 - self._activation_eps)

                H, W = imgs.shape[2], imgs.shape[3]
                if self._color_space == "lab":
                    rgb_in   = lab_to_rgb_tensor(imgs).clamp(0, 1)
                    rgb_gt   = lab_to_rgb_tensor(labels.view(-1, 3, 1, 1)).expand(-1, -1, H, W).clamp(0, 1)
                    rgb_pred = lab_to_rgb_tensor(preds.view(-1, 3, 1, 1)).expand(-1, -1, H, W).clamp(0, 1)
                else:
                    rgb_in   = imgs.clamp(0, 1)
                    rgb_gt   = labels.view(-1, 3, 1, 1).expand(-1, -1, H, W).clamp(0, 1)
                    rgb_pred = preds.view(-1, 3, 1, 1).expand(-1, -1, H, W).clamp(0, 1)

                triplets = torch.stack([rgb_in, rgb_gt, rgb_pred], dim=1).flatten(0, 1)
                grid = make_grid(triplets, nrow=3, normalize=False, value_range=(0, 1), pad_value=1.0)
                w = self._writer
                if w is not None:
                    w.add_image(self.TAGS["images"], grid, epoch)
                break

    def run_end(self, final_metrics: Dict[str, float]) -> None:
        # Ensure we *do* create a writer even if min_log_epoch was not reached,
        # so HParams and end-of-run scalars always get written.
        self._writer = _ensure_writer(self._run_dir, self._writer)
        # Sanitize metrics for HParams table
        # Cast to plain floats to avoid NumPy types
        metrics = {k: float(v) for k, v in final_metrics.items() if v is not None}        
        if self._hparams is None:
            self._hparams = {}
        # Write config/meta ONCE here (never at activation), for lean logs
        if not self._wrote_config_once:
            txt = "```\n" + json.dumps(self._hparams, indent=2) + "\n```"
            self._writer.add_text(self.TAGS["config"], txt, 0)
            self._writer.add_text(self.TAGS["meta"], ", ".join(self.meta_columns), 0)
            self._wrote_config_once = True

        # ---- One-off end-of-run scalars (use the last epoch index if known) ----
        step = int(self._last_epoch_seen or 0)
        # Best validation ΔE00 achieved during training
        self._add_scalar_safe(self.TAGS["de/val_best"], final_metrics.get("val_de00_best"), step)
        # Final test metrics (written by train.py into final_metrics)
        self._add_scalar_safe(self.TAGS["de/test"],       final_metrics.get("test_de00"), step)
        self._add_scalar_safe(self.TAGS["mse_lab/test"],  final_metrics.get("test_mse_lab"), step)
        self._add_scalar_safe(self.TAGS["mse_rgb/test"],  final_metrics.get("test_mse_rgb"), step)

        if metrics:
            # Log hparams ONCE, with already-safe dict prepared at run_start
            self._writer.add_hparams(self._hparams, metrics, run_name=self.TAGS["hparams"])
        self.close()

    def interrupt(self) -> None:
        pass

    def prune(self) -> None:
        self.close()
        if os.path.isdir(self._run_dir):
            shutil.rmtree(self._run_dir)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

# --------------------
# Minimal checkpoint observer (unchanged API)
# --------------------
class ModelCheckpoint(Observer):
    """
    Checkpointing with verbosity control.
      - monitor: key in val_log (e.g., 'de00')
      - mode: 'min'/'max'
      - save_top_k, save_last
      - min_epoch
      - verbose: 0 (silent), 1 (normal), 2 (print saves/removals)
    """
    def __init__(self,
                 outdir: str,
                 filename: str | None = None,
                 monitor: str = "de00",
                 mode: str = "min",
                 save_top_k: int = 1,
                 save_last: bool = True,
                 min_epoch: int = 0,
                 verbose: int = 1):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.filename = filename
        self.monitor = monitor
        self.mode = mode.lower().strip()
        assert self.mode in ("min", "max")
        self.save_top_k = int(save_top_k)
        self.save_last = bool(save_last)
        self.min_epoch = int(min_epoch)
        self.verbose = int(verbose)

        self._saved: List[Tuple[float, str]] = []
        self._last_path: Optional[str] = None
        self._best: Optional[float] = None
        self._cfg: Dict[str, Any] = {}

    def _is_better(self, a: float, b: float) -> bool:
        return (a < b) if self.mode == "min" else (a > b)

    def _suffix(self, metric: float) -> str:
        return f"{self.monitor}{metric:.4f}"

    def _make_name(self, epoch: int, metric: float) -> str:
        base = self.filename
        if base is None:
            model = self._cfg.get("model", "model")
            backbone = self._cfg.get("backbone", "backbone")
            base = f"{model}_{backbone}_e{{epoch:03d}}_{self._suffix(metric)}.pt"
        return base.format(epoch=epoch)

    def run_start(self, config: Dict[str, Any]) -> None:
        self._cfg = dict(config)

    def epoch_end(self, train_log: Dict[str, float], val_log: Dict[str, float],
                  epoch: int, lr: float) -> None:
        if epoch < self.min_epoch:
            # Still optionally save rolling last even before min_epoch? Keep original behavior (skip).
            return

        metric = (val_log or {}).get(self.monitor, None)

        # Always write rolling last.pt, even if metric is None (use NaN placeholder)
        if self.save_last:
            path_last = os.path.join(self.outdir, f"last.pt")
            self._save(path_last, val_metric=(float(metric) if metric is not None else float("nan")), epoch=epoch)
            self._last_path = path_last

        # Top-K best checkpoints only when a real metric exists
        if (self.save_top_k > 0) and (metric is not None):
            to_save = True
            if len(self._saved) >= self.save_top_k:
                worst_idx = (min if self.mode == "max" else max)(
                    range(len(self._saved)), key=lambda i: self._saved[i][0]
                )
                worst_metric, worst_path = self._saved[worst_idx]
                if not self._is_better(metric, worst_metric):
                    to_save = False
            if to_save:
                fname = self._make_name(epoch, float(metric))
                path = os.path.join(self.outdir, fname)
                self._save(path, val_metric=float(metric), epoch=epoch)
                if self.verbose >= 2:
                    print(f"[ckpt] saved {self.monitor}={float(metric):.4f} -> {os.path.basename(path)}")
                self._saved.append((float(metric), path))
                if len(self._saved) > self.save_top_k:
                    worst_idx = (min if self.mode == "max" else max)(
                        range(len(self._saved)), key=lambda i: self._saved[i][0]
                    )
                    _, rm_path = self._saved.pop(worst_idx)
                    try:
                        os.remove(rm_path)
                        if self.verbose >= 2:
                            print(f"[ckpt] removed {os.path.basename(rm_path)} (exceeded top-{self.save_top_k})")
                    except FileNotFoundError:
                        pass

        # Track best metric only if it exists
        if metric is not None and ((self._best is None) or self._is_better(metric, self._best)):
            self._best = float(metric)


    def _save(self, path: str, *, val_metric: float, epoch: int) -> None:
        model = self._cfg.get("_model_ref")
        meta_dim = self._cfg.get("meta_dim")
        state = {
            "model": self._cfg.get("model"),
            "backbone": self._cfg.get("backbone"),
            "state_dict": model.state_dict() if model is not None else None,
            "meta_dim": meta_dim,
            f"val_{self.monitor}": float(val_metric),
            "epoch": int(epoch),
            "config": {k: v for k, v in self._cfg.items() if k != "_model_ref"},
        }
        torch.save(state, path)

    def run_end(self, final_metrics: Dict[str, float]) -> None:
        pass

    def prune(self) -> None:
        for _, p in self._saved:
            try: os.remove(p)
            except FileNotFoundError: pass
        if self._last_path and os.path.isfile(self._last_path):
            try: os.remove(self._last_path)
            except FileNotFoundError: pass
        self._saved.clear()
        self._last_path = None


TrainLogger = TensorBoardLogger

class EarlyStopping(Observer):
    """
    Stop training when the monitored validation metric has stopped improving.
    - monitor: key from val_log (e.g., 'de00', 'mse', 'mse_lab', ...)
    - mode: 'min' or 'max'
    - patience: epochs to wait after last improvement (min_delta-aware)
    - min_delta: required improvement magnitude
    - min_epoch: don't consider stopping before this epoch
    - restore_best: load best state_dict() on stop (kept on CPU to save VRAM)
    """
    def __init__(self,
                 monitor: str = "de00",
                 mode: str = "min",
                 patience: int = 10,
                 min_delta: float = 0.0,
                 min_epoch: int = 0,
                 restore_best: bool = True,
                 verbose: int = 1):
        self.monitor = monitor
        self.mode = mode.lower().strip()
        assert self.mode in ("min", "max")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epoch = int(min_epoch)
        self._restore = bool(restore_best)
        self.verbose = int(verbose)

        self._model = None
        self._best = None
        self._best_epoch = None
        self._bad = 0
        self._stop = False
        self._best_state = None

    def run_start(self, config: Dict[str, Any]) -> None:
        self._model = config.get("_model_ref", None)

    def _is_better(self, val: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return val < (self._best - self.min_delta)
        else:
            return val > (self._best + self.min_delta)

    def epoch_end(self, train_log: Dict[str, float], val_log: Dict[str, float],
                  epoch: int, lr: float) -> None:
        if self._stop:
            return
        cur = val_log.get(self.monitor, None)
        if cur is None:
            return
        if self._is_better(cur):
            self._best = float(cur)
            self._best_epoch = int(epoch)
            self._bad = 0
            if self._restore and self._model is not None:
                # keep a CPU copy to free VRAM
                state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                self._best_state = state
        else:
            if epoch >= self.min_epoch:
                self._bad += 1
                if self._bad >= self.patience:
                    self._stop = True
                    if self.verbose >= 1:
                        print(f"[early] stop at epoch {epoch} "
                              f"(best {self.monitor}={self._best:.4f} @ {self._best_epoch})")

    # Queried by the training loop
    def should_stop(self) -> bool:
        return self._stop

    # Called by the training loop when stopping
    def restore_best(self) -> None:
        if self._restore and self._best_state is not None and self._model is not None:
            self._model.load_state_dict(self._best_state)

class ObserverList(Observer):
    def __init__(self, observers: Iterable[Observer]):
        self._obs: List[Observer] = list(observers)

    def _safe(self, fn: str, *a, **k):
        for ob in self._obs:
            m = getattr(ob, fn, None)
            if callable(m):
                try:
                    m(*a, **k)
                except Exception as e:
                    print(f"[ObserverList] {fn} error in {ob.__class__.__name__}: {e}")
                    traceback.print_exc()

    # fan-out
    def run_start(self, config): self._safe("run_start", config)
    def epoch_end(self, train_log, val_log, epoch, lr): self._safe("epoch_end", train_log, val_log, epoch, lr)
    def log_images(self, model, val_loader, device, epoch): self._safe("log_images", model, val_loader, device, epoch)
    def run_end(self, final_metrics): self._safe("run_end", final_metrics)
    def interrupt(self): self._safe("interrupt")
    def prune(self): self._safe("prune")

    # NEW: queried by train loop
    def should_stop(self) -> bool:
        for ob in self._obs:
            fn = getattr(ob, "should_stop", None)
            if callable(fn) and fn():
                return True
        return False

    # NEW: called by train loop on stop to restore best model (if any observer supports it)
    def restore_best(self) -> None:
        for ob in self._obs:
            fn = getattr(ob, "restore_best", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    print(f"[ObserverList] restore_best error in {ob.__class__.__name__}: {e}")

Callback = Observer
CallbackList = ObserverList

__all__ = ["Observer", "ObserverList", "TensorBoardLogger", "ModelCheckpoint",
           "EarlyStopping", "Callback", "CallbackList", "TrainLogger"]

