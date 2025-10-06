# src/engine/train.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional, Union
import math
import torch, torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau, SequentialLR, LinearLR

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except Exception:
    tqdm = None
    _HAVE_TQDM = False

from ..analysis.smoke_utils import write_summary_and_check
from .datamodule import DataModule, DataModuleConfig
from ..models import build_model
from ..models.backbones import BACKBONES
from ..data.dataset import to_batch
from ..metrics.losses import ColorLoss
from ..metrics.evaluate import evaluate
from ..analysis.tensorboard import CallbackList, TensorBoardLogger, ModelCheckpoint, EarlyStopping
from .utils import set_seed, get_device, device_str
from .config import RunConfig

from src.diag import D

def intish(x: str) -> int:
    return int(float(x))

def _cosine_anneal_w(t: int, T: int, floor: float = 0.0) -> float:
    if T <= 0: return floor
    t = min(max(t, 0), T)
    w = 0.5 * (1.0 + math.cos(math.pi * t / T))
    return floor + (1.0 - floor) * w


def _apply_pred_activation(x: torch.Tensor, kind: str, eps: float) -> torch.Tensor:
    if kind == "none": return x
    if kind == "sigmoid": return torch.sigmoid(x).clamp(eps, 1 - eps)
    if kind == "sigmoid_eps":
        s = torch.sigmoid(x); return eps + (1 - 2 * eps) * s
    if kind == "tanh01":
        y = 0.5 * (torch.tanh(x) + 1.0); return y.clamp(eps, 1 - eps)
    raise ValueError(f"Unknown pred_activation: {kind}")

def _ns_to_run(ns) -> RunConfig:
    """Back-compat: convert argparse.Namespace or legacy TrainConfig into RunConfig."""
    if isinstance(ns, RunConfig):
        return ns
    # argparse.Namespace path: reuse RunConfig.from_namespace
    from .config import RunConfig as _RC
    return _RC.from_namespace(ns)

def fit(cfg_like: Union[RunConfig, Any],
        dm: Optional[DataModule] = None,
        test_loader=None,
        eval_test_after: bool = False) -> Dict[str, float]:
    run: RunConfig = _ns_to_run(cfg_like)

    D.enable(run, outdir=run.ckpt.outdir)
    D.config_snapshot(run)
    D.validate.meta_config(run.data)
    D.validate.gpu_support(getattr(run.data, "meta_attn_impl", "eager"))

    # Guardrails
    if run.data.color_space == "lab" and run.model.backbone != "smallcnn" and run.data.pretrained:
        print("[info] LAB + pretrained timm backbone requested; forcing pretrained=False.")
        run.data.pretrained = False

    set_seed(run.data.seed)
    device = get_device()
    if run.loop.verbose > 1:
        print(f"[device] {device_str(device)}")

    # Data (build if not provided)
    if dm is None:
        dm = DataModule(DataModuleConfig(
            images_dir=run.data.images_dir,
            json_dir=run.data.json_dir,
            labels_csv=run.data.labels_csv,
            batch_size=run.loop.batch_size,
            val_split=run.data.val_split,
            workers=run.data.workers,
            seed=run.data.seed,
            hidden_classes_cnt=run.data.hidden_classes_cnt,
            group_split=run.data.group_split,
            color_space=run.data.color_space,
            features=run.data.features,
            pretrained=run.data.pretrained,
            backbone=run.data.backbone or run.model.backbone,
            include_test=run.data.include_test,
            test_per_class=run.data.test_per_class,
            excluded_folders=run.data.excluded_folders,
            included_folders=run.data.included_folders,
            split_file=run.data.split_file,
            save_splits_flag=run.data.save_splits_flag,
        )).setup()

    # Model & optim
    model = build_model(
        run.model.model,
        backbone=run.model.backbone,
        meta_dim=dm.meta_dim,
        token_stage=getattr(run.model, "token_stage", -2),
        pretrained=bool(run.data.pretrained),
    ).to(device)
    if run.loop.optim == "sgd":
        base_lr = (0.1 * run.loop.batch_size / 256.0) if run.loop.lr_auto else run.loop.lr
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=run.loop.momentum,
            weight_decay=run.loop.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=run.loop.lr,
            weight_decay=run.loop.weight_decay,
            betas=(0.9, 0.999),
        )

    if run.loop.lr_schedule == "paper_resnet":
        # Step down by 10x at 1/3 and 2/3 of total epochs (classic ImageNet recipe).
        m1 = max(1, int(run.loop.epochs * (1.0/3.0)))
        m2 = max(m1+1, int(run.loop.epochs * (2.0/3.0)))
        scheduler = MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.1)
    elif run.loop.lr_schedule == "plateau":
        # Monitor validation ΔE00; reduce by 10x if it plateaus.
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4, threshold=0.001)
    else:
        # cosine (with a tiny warmup for stability)
        warm = max(1, min(5, run.loop.epochs // 10))
        warmup = LinearLR(optimizer, start_factor=0.2, total_iters=warm)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, run.loop.epochs - warm))
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warm])

    criterion = ColorLoss(
        input_space=run.model.color_space,
        mse_space=run.loss.mse_space,
        mse_weight=run.loss.mse_weight_start,
        smooth_eps=run.loss.de_smooth_eps,
        robust=run.loss.robust,
        huber_delta=run.loss.huber_delta,
        charb_eps=run.loss.charb_eps,
        de_reduce=run.loss.de_reduce,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(run.loop.amp))
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # ---- No-val safeguard (train-full) ----
    has_val = True
    try:
        has_val = (val_loader is not None) and (getattr(dm, "n_val", 0) > 0) and (len(val_loader) > 0)
    except Exception:
        has_val = (val_loader is not None) and (getattr(dm, "n_val", 0) > 0)
    if not has_val:
        # No validation: disable early stopping so runs don't stop at epoch 1
        run.early.early_stop_patience = 0

    callbacks = CallbackList([
        TensorBoardLogger(run.log.logdir, f"{run.model.model}-{run.model.backbone}", list(dm.meta_cols),
                          min_log_epoch=run.log.min_log_epoch),
        ModelCheckpoint(outdir=run.ckpt.outdir,
                        filename=run.ckpt.ckpt_filename if run.ckpt.ckpt_filename else None,
                        monitor=run.ckpt.ckpt_monitor,
                        mode=run.ckpt.ckpt_mode,
                        save_top_k=run.ckpt.save_top_k,
                        save_last=run.ckpt.save_last,
                        min_epoch=run.ckpt.min_ckpt_epoch,
                        verbose=run.loop.verbose),
    ])
    if run.early.early_stop_patience > 0:
        callbacks._obs.append(EarlyStopping(
            monitor=run.early.early_stop_monitor,
            mode=run.early.early_stop_mode,
            patience=run.early.early_stop_patience,
            min_delta=run.early.early_stop_min_delta,
            min_epoch=run.early.early_stop_min_epoch,
            restore_best=run.early.restore_best_on_stop,
            verbose=run.loop.verbose,
        ))
    callbacks.run_start({
        "model": run.model.model, "backbone": run.model.backbone, "features": run.data.features,
        "epochs": run.loop.epochs, "batch_size": run.loop.batch_size, "lr": run.loop.lr, "weight_decay": run.loop.weight_decay,
        "mse_weight_start": run.loss.mse_weight_start, "mse_weight_epochs": run.loss.mse_weight_epochs,
        "mse_space_switch": run.loss.mse_space_switch, "mse_space": run.loss.mse_space,
        "val_split": run.data.val_split, "grad_clip": run.loop.grad_clip, "workers": run.data.workers, "seed": run.data.seed,
        "group_split": run.data.group_split, "hidden_classes_cnt": int(run.data.hidden_classes_cnt),
        "meta_dim": dm.meta_dim, "color_space": run.model.color_space,
        "pred_activation": run.model.pred_activation, "activation_eps": run.model.activation_eps,
        "device": device_str(device), "train_size": dm.n_train, "val_size": dm.n_val,
        "_model_ref": model, "excluded_folders": run.data.excluded_folders, "included_folders": run.data.included_folders,
        "meta_encoder": run.data.meta_encoder, "meta_model_name": run.data.meta_model_name,
        "meta_layers": run.data.meta_layers, "meta_text_template": run.data.meta_text_template,
    })

    if run.loop.verbose >= 1:
        lr_show = (0.1 * run.loop.batch_size / 256.0) if (run.loop.optim=="sgd" and run.loop.lr_auto) else run.loop.lr
        print(f"[run] model={run.model.model} backbone={run.model.backbone} "
              f"train={dm.n_train} val={dm.n_val} epochs={run.loop.epochs} "
              f"optim={run.loop.optim} lr={lr_show:.5f} wd={run.loop.weight_decay:g} "
              f"momentum={run.loop.momentum if run.loop.optim=='sgd' else '—'} "
              f"schedule={run.loop.lr_schedule}")
    best_val = float("inf")
    best_epoch = 0
    epochs_run = 0
    last_train_log = last_val_log = None

    use_pbar = (run.loop.verbose == 1) and _HAVE_TQDM
    pbar = tqdm(total=run.loop.epochs, desc=f"Training ({run.model.model}/{run.model.backbone})",
                leave=True, dynamic_ncols=True) if use_pbar else None
    
    try:
        for epoch in range(1, run.loop.epochs + 1):
            model.train()
            criterion.mse_weight = run.loss.mse_weight_start * _cosine_anneal_w(
                epoch - 1, run.loss.mse_weight_epochs, floor=0.05  # keep a 5% floor
            )            
            if run.loss.mse_space == "rgb" and epoch >= run.loss.mse_space_switch:
                criterion.mse_space = "lab"

            running_loss = running_de = running_mse = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader, 1):
                b = to_batch(batch, device)

                img, meta, y = b["image"], b["metadata"], b["label"]

                with torch.cuda.amp.autocast(enabled=bool(run.loop.amp)):
                    preds = model(img, meta)
                    if run.model.color_space.lower() == "rgb":
                        preds = _apply_pred_activation(preds, run.model.pred_activation, run.model.activation_eps)
                    total_loss, batch_metrics = criterion(preds, y)

                if run.loop.accum_steps > 1:
                    scaler.scale(total_loss / run.loop.accum_steps).backward()
                    if step % run.loop.accum_steps == 0:
                        if run.loop.grad_clip and run.loop.grad_clip > 0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), run.loop.grad_clip)
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                else:
                    scaler.scale(total_loss).backward()
                    if run.loop.grad_clip and run.loop.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), run.loop.grad_clip)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

                running_loss += float(batch_metrics["loss"])
                running_de   += float(batch_metrics["de00"])
                if criterion.mse_space == "rgb" and batch_metrics["mse_rgb"] is not None:
                    running_mse += float(batch_metrics["mse_rgb"])
                elif criterion.mse_space != "rgb" and batch_metrics["mse_lab"] is not None:
                    running_mse += float(batch_metrics["mse_lab"])
                n_batches += 1

            train_log = {"loss": running_loss/max(1,n_batches),
                         "de00": running_de/max(1,n_batches),
                         "mse": (running_mse/max(1,n_batches)) if (criterion.mse_weight > 0) else None}
            if has_val:
                val_log = evaluate(model, val_loader, device=device, criterion=criterion,
                                   color_space=run.model.color_space,
                                   pred_activation=run.model.pred_activation,
                                   activation_eps=run.model.activation_eps)
            else:
                # No validation this run
                val_log = {"de00": None, "mse_lab": None, "mse_rgb": None}

            if run.loop.lr_schedule == "plateau":
                # Only step on plateau when a real validation metric exists
                if val_log["de00"] is not None:
                    scheduler.step(val_log["de00"])
                else:
                    scheduler.step()  # harmless step to keep schedule moving

            last_train_log, last_val_log = train_log, val_log
            epochs_run += 1

            callbacks.epoch_end(train_log, val_log, epoch, scheduler.get_last_lr()[0])
            if has_val and run.log.log_images_every > 0 and epoch % run.log.log_images_every == 0:    
                callbacks.log_images(model, val_loader, device, epoch)

            if callbacks.should_stop():
                if use_pbar and pbar is not None:
                    val_txt = f"{val_log['de00']:.3f}" if (val_log and val_log.get('de00') is not None) else "—"
                    loss_txt = (f"{(train_log or {}).get('loss', float('nan')):.4f}"
                                if (train_log and 'loss' in train_log) else "—")
                    pbar.set_postfix({"train loss": loss_txt, "val ΔE00": val_txt, "early_stop": "yes"})
                stopped_early = True
                break

            if use_pbar and pbar is not None:
                pbar.update(1)
                if val_log and (val_log.get('de00') is not None):
                    if val_log['de00'] < best_val:
                        best_val = val_log['de00']

                loss_txt = (f"{(train_log or {}).get('loss', float('nan')):.4f}"
                            if (train_log and 'loss' in train_log) else "—")
                val_txt = f"{val_log['de00']:.3f}" if (val_log and val_log.get('de00') is not None) else "—"
                best_txt = f"{best_val:.3f}" if best_val != float('inf') else "—"
                pbar.set_postfix({"train loss": loss_txt, "val ΔE00": val_txt, "best": best_txt})

            if val_log["de00"] is not None and val_log["de00"] < best_val:
                best_val = val_log["de00"]
                best_epoch = epoch

    except KeyboardInterrupt:
        print("[interrupt] Training interrupted by user.")
        callbacks.interrupt()
    finally:
        if use_pbar and pbar is not None: pbar.close()

        if run.ckpt.prune_short_runs and epochs_run < run.ckpt.min_keep_epochs:
            if run.loop.verbose >= 1:
                print(f"[prune] epochs_run={epochs_run} < {run.ckpt.min_keep_epochs}; deleting run + ckpt.")
            callbacks.prune()
            return {"val_de00_best": None}

        # Always restore best-val weights (if tracked) before any final eval,
        # not only when early stopping actually fired.
        if has_val and run.early.restore_best_on_stop:
            callbacks.restore_best()

        final = {
            "val_de00_best": best_val if best_val != float("inf") else None,
            "val_de00_last": last_val_log["de00"] if last_val_log else None,
            "val_mse_lab_last":  last_val_log.get("mse_lab") if last_val_log else None,
            "val_mse_rgb_last":  last_val_log.get("mse_rgb") if last_val_log else None,
            "train_de00_last": last_train_log["de00"] if last_train_log else None,
        }

        # Optional: evaluate on test **after** restoring best
        if eval_test_after and test_loader is not None:
            if run.loop.verbose >= 1:
                try:
                    n_test = getattr(getattr(test_loader, 'dataset', None), '__len__', lambda: None)() or "?"
                except Exception:
                    n_test = "?"
                print(f"[eval] Evaluating on TEST after training (n={n_test})...")
            test_log = evaluate(
                model, test_loader, device=device, criterion=criterion,
                color_space=run.model.color_space,
                pred_activation=run.model.pred_activation,
                activation_eps=run.model.activation_eps,
                )
            # Keep top-level, short keys for summaries
            final["test_de00"] = test_log.get("de00")
            if "mse_lab" in test_log: final["test_mse_lab"] = test_log["mse_lab"]
            if "mse_rgb" in test_log: final["test_mse_rgb"] = test_log["mse_rgb"]

        callbacks.run_end(final)
        if run.loop.verbose >= 1 and final["val_de00_best"] is not None:
            print(f"[done] best ΔE00={final['val_de00_best']:.3f} over {epochs_run} epochs")

        # ---- Persist final metrics for summarizer (prefer TEST, fallback to VAL) ----
        try:
            import json, os
            # Prefer checkpoint outdir; then ModelCheckpoint.dirpath; then logdir
            outdir = getattr(getattr(run, "ckpt", None), "outdir", None)
            if not outdir:
                # only then try callbacks
                for cb in (getattr(callbacks, "_obs", []) or []):
                    if cb.__class__.__name__ == "ModelCheckpoint":
                        outdir = getattr(cb, "dirpath", None) or outdir
            if not outdir:
                outdir = getattr(getattr(run, "log", None), "logdir", None)

            # Decide which epoch to record: prefer best val epoch if meaningful, else last epoch run
            if has_val and run.early.restore_best_on_stop and (best_val != float("inf")) and (best_epoch > 0):
                epoch_to_write = int(best_epoch)
            else:
                epoch_to_write = int(epochs_run)

            if outdir:
                cfg_for_json = {
                    "model": run.model.model,
                    "backbone": run.model.backbone,
                    "token_stage": getattr(run.model, "token_stage", -2),
                    "pretrained": bool(run.data.pretrained),
                    "features": run.data.features,
                    "color_space": run.model.color_space,
                    "epochs": run.loop.epochs,
                    "batch_size": run.loop.batch_size,
                    "val_split": run.data.val_split,
                    "group_split": run.data.group_split,
                    "included_folders": run.data.included_folders,
                    "excluded_folders": run.data.excluded_folders,
                    "meta_encoder": run.data.meta_encoder,
                    "meta_model_name": run.data.meta_model_name,
                    "meta_layers": run.data.meta_layers,
                    "meta_text_template": run.data.meta_text_template,
                    "optim": run.loop.optim,
                    "lr": run.loop.lr,
                    "weight_decay": run.loop.weight_decay,
                    "lr_schedule": run.loop.lr_schedule,
                    "lr_auto": run.loop.lr_auto,
                    "mse_weight_epochs": run.loss.mse_weight_epochs,
                    "mse_space_switch": run.loss.mse_space_switch,
                }
                payload = {"epoch": epoch_to_write, "config": cfg_for_json}
                # Prefer TEST if available, otherwise VAL
                if final.get("test_de00") is not None:
                    payload["metric_kind"] = "test"
                    payload["test_de00"] = float(final["test_de00"])
                    if final.get("test_mse_lab") is not None: payload["test_mse_lab"] = float(final["test_mse_lab"])
                    if final.get("test_mse_rgb") is not None: payload["test_mse_rgb"] = float(final["test_mse_rgb"])
                else:
                    payload["metric_kind"] = "val"
                    payload["val_de00"] = float(final["val_de00_best"]) if final.get("val_de00_best") is not None else None
                
                out_p = os.path.join(outdir, "final_summary.json")
                with open(out_p, "w") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)
                kind = payload.get("metric_kind")
                print(f"[eval] Wrote final summary → {out_p} (kind={kind})")  
        except Exception as e:
            print(f"[eval][warn] could not write final_summary.json: {e}")

        # ---- SMOKE (optional; no-op unless enabled in config) ----
        # Guard for older configs that don't define 'smoke'
        if getattr(run, "smoke", None) and getattr(run.smoke, "smoke_enabled", False):
            # Sizes (test may be 0 if not included)
            n_train = getattr(dm, "n_train", None) or 0
            n_val   = getattr(dm, "n_val", None) or 0
            n_test  = getattr(dm, "n_test", 0) or (0 if test_loader is None else 0)

            # Compact config snapshot used to key the baseline
            run_cfg = {
                "model": run.model.model,
                "backbone": run.model.backbone,
                "features": run.data.features,
                "color_space": run.model.color_space,
                "epochs": run.loop.epochs,
                "batch_size": run.loop.batch_size,
                "val_split": run.data.val_split,
                "group_split": run.data.group_split,
                "excluded_folders": run.data.excluded_folders,
                "included_folders": run.data.included_folders,
                "mse_space": run.loss.mse_space,
                "mse_weight_start": run.loss.mse_weight_start,
                "mse_weight_epochs": run.loss.mse_weight_epochs,
                "mse_space_switch": run.loss.mse_space_switch,
            }
            sizes = {"train": n_train, "val": n_val, "test": n_test}
            test_metrics = None
            if "test_de00" in final and final["test_de00"] is not None:
                test_metrics = {k: v for k, v in final.items() if k.startswith("test_")}

            # Write JSON summary and compare to (or update) baseline
            write_summary_and_check(
                run_cfg=run_cfg,
                sizes=sizes,
                best_val_de00=float(final["val_de00_best"]) if final["val_de00_best"] is not None else float("inf"),
                best_epoch=int(best_epoch),
                test_metrics=test_metrics,
                smoke_cfg=run.smoke
            )

        return final

def parse_token_stage(s):
    """
    Accepts:
      - int / int-ish float:   -3, "-3", "-3.0"
      - JSON list:             "[1,2,-3]"
      - CSV:                   "1,2,-3"
      - aliases:               "ms", "multi", "auto", "", "none"  -> "ms"
    Returns: int, list[int], or "ms".
    """
    if s is None:
        return "ms"

    # already a list/tuple -> coerce each element
    if isinstance(s, (list, tuple)):
        return [intish(x) for x in s]

    # plain numbers
    if isinstance(s, (int, float)):
        # only allow integral floats
        if isinstance(s, float) and not float(s).is_integer():
            raise ValueError(f"token_stage must be integral; got {s}")
        return int(s)

    # strings
    s = str(s).strip().lower()
    if s in ("", "ms", "multi", "auto", "none"):
        return "ms"

    # JSON list
    if s.startswith("[") and s.endswith("]"):
        import json
        v = json.loads(s)
        if isinstance(v, list):
            return [intish(x) for x in v]
        raise ValueError(f"token_stage JSON must be a list, got: {type(v)}")

    # CSV
    if "," in s:
        return [intish(x) for x in s.split(",") if x.strip() != ""]

    # fallback: single int-ish value (handles "-3" and "-3.0")
    return intish(s)


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Training flags (used by experiment/kfold).")
    # Required data
    p.add_argument("--images_dir", required=True)
    p.add_argument("--json_dir",   required=True)
    p.add_argument("--labels_csv", required=True)
    
    # Model & data-related knobs (kept names)
    p.add_argument("--model", choices=["late","film","xattn"], default="film")
    p.add_argument("--backbone", default="smallcnn", choices=sorted(list(BACKBONES.keys())))
    p.add_argument(
        "--token_stage",
        type=parse_token_stage,
        default="ms",
        help="Stage(s) for FiLM/XAttn. int (e.g. -2), csv ('1,2,3'), JSON list ('[1,2,3]'), or 'ms'/'auto'."
    )
    p.add_argument("--color_space", choices=["lab","rgb"], default="rgb")
    p.add_argument("--features", choices=["image","image+mean","image+meta","image+mean+meta"], default="image+mean+meta")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--workers", type=intish, default=0)
    p.add_argument("--seed", type=intish, default=100)
    p.add_argument("--hidden_classes_cnt", type=intish, default=0)
    p.add_argument("--group_split", type=str, choices=["class", "folder"], default=None,
        help="Group split by 'class', 'folder', or 'none' for no grouping"
    )    
    p.add_argument("--excluded_folders", nargs="+", type=str, default=None,
        help="List of folder names to exclude from the dataset (e.g. --excluded_folders folder1 folder2)."
    )
    p.add_argument("--included_folders", nargs="+", type=str, default=None,
        help="List of folder names to include in the dataset (e.g. --included_folders folder1 folder2)."
    )
    p.add_argument("--include_test", action="store_true", default=False)
    p.add_argument("--test_per_class", type=intish, default=3)
    try:
        from argparse import BooleanOptionalAction
        p.add_argument("--pretrained", action=BooleanOptionalAction, default=True)
    except Exception:
        p.add_argument("--pretrained", action="store_true", default=True)
    
    # Loop/opt
    p.add_argument("--epochs", type=intish, default=30)
    p.add_argument("--batch_size", type=intish, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optim", type=str, default="adamw",
                    choices=["adamw", "sgd"],
                    help="Use 'sgd' to mimic original ResNet training.")
    p.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (paper uses 0.9).")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (paper uses 1e-4).")
    p.add_argument("--lr_auto", action="store_true",
                        help="If set with SGD, use linear scaling: lr=0.1*(batch/256).")
    p.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "paper_resnet", "plateau"],
                        help="paper_resnet = MultiStep @ 1/3, 2/3 epochs with gamma=0.1.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--accum_steps", type=intish, default=1)
    p.add_argument("--verbose", type=intish, default=1, choices=[0,1,2])

    # Loss
    p.add_argument("--pred_activation", choices=["none","sigmoid","sigmoid_eps","tanh01"], default="sigmoid_eps")
    p.add_argument("--activation_eps", type=float, default=1e-3)
    p.add_argument("--mse_space", choices=["same","lab","rgb"], default="rgb")
    p.add_argument("--mse_weight_start", type=float, default=1.0)
    p.add_argument("--mse_weight_epochs", type=intish, default=75)
    p.add_argument("--mse_space_switch", type=intish, default=75)
    p.add_argument("--de_smooth_eps", type=float, default=1e-6)
    p.add_argument("--robust", default="none",
               choices=["none","huber","charbonnier"],
               help="Robustify per-sample ΔE00 when computing the loss.")
    p.add_argument("--huber_delta", type=float, default=1.0,
                help="Huber transition point (≈1.0 is reasonable).")
    p.add_argument("--charb_eps", type=float, default=1e-3,
                help="Charbonnier epsilon (≈1e-3..1e-2).")
    p.add_argument("--de_reduce", default="mean",
                choices=["mean","median"],
               help="Reduction for robustified ΔE term in the loss.")
    
    # Logging & ckpt
    p.add_argument("--logdir", type=str, default="./runs")
    p.add_argument("--log_images_every", type=intish, default=5)
    p.add_argument("--min_log_epoch", type=intish, default=0)
    p.add_argument("--outdir", type=str, default="./checkpoints")
    p.add_argument("--ckpt_monitor", type=str, default="de00")
    p.add_argument("--ckpt_mode", choices=["min","max"], default="min")
    p.add_argument("--save_top_k", type=intish, default=1)
    p.add_argument("--save_last", action="store_true", default=False)
    p.add_argument("--ckpt_filename", type=str, default=None)
    p.add_argument("--min_ckpt_epoch", type=intish, default=0)
    p.add_argument("--prune_short_runs", action="store_true", default=False)
    p.add_argument("--min_keep_epochs", type=intish, default=25)
    p.add_argument("--log_best_val", action="store_true", default=True)
    
    # Early stopping
    p.add_argument("--early_stop_patience", type=intish, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--early_stop_monitor", type=str, default="de00")
    p.add_argument("--early_stop_mode", choices=["min","max"], default="min")
    p.add_argument("--early_stop_min_epoch", type=intish, default=0)
    try:
        import argparse as _argp
        p.add_argument("--restore_best_on_stop", action=_argp.BooleanOptionalAction, default=True)
    except Exception:
        p.add_argument("--restore_best_on_stop", action="store_true", default=True)
    
    # Split persistence
    p.add_argument("--split_file", type=str, default=None)
    p.add_argument("--save_splits", action="store_true", default=False)

    # --- Metadata as text ---
    p.add_argument("--meta_encoder", choices=["none","flair"], default="none",
                   help="Replace numeric metadata with a text embedding vector.")
    p.add_argument("--meta_model_name", type=str, default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--meta_layers", type=str, default="-1,-2,-3,-4",
                   help='Flair layers string, e.g. "-1" or "-1,-2,-3,-4" or "all".')
    p.add_argument("--meta_text_template", choices=["compact","kv","json"], default="compact")
    p.add_argument("--meta_batch_size", type=intish, default=64)

    # ---- SMOKE (optional) ----
    p.add_argument("--smoke_enabled", action="store_true", default=False,
                help="Enable smoke summary + baseline check")
    p.add_argument("--abs_tol", type=float, default=0.10,
                help="Absolute ΔE00 tolerance for smoke regression check")
    p.add_argument("--rel_tol", type=float, default=0.05,
                help="Relative tolerance (fraction of baseline) for smoke")
    p.add_argument("--baseline_dir", type=str, default="./runs/smoke",
                help="Directory to store smoke_baseline.json & smoke_result.json")
    p.add_argument("--result_filename", type=str, default="smoke_result.json",
                help="Filename for the smoke run summary JSON")
    p.add_argument("--update_baseline", action="store_true", default=False,
                help="Accept current result as new smoke baseline")

    return p
