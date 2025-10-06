# src/engine/config.py
from __future__ import annotations
from dataclasses import dataclass, fields, MISSING, field
from typing import Any, Dict, Optional

# ---- Data (what the DataModule needs) ----
@dataclass
class DataConfig:
    images_dir: str
    json_dir: str
    labels_csv: str
    val_split: float = 0.15
    workers: int = 0
    seed: int = 100
    hidden_classes_cnt: int = 0
    group_split: str = None
    color_space: str = "rgb"                 # rgb | lab
    features: str = "image+mean+meta"        # image | image+mean | image+meta | image+mean+meta
    pretrained: bool = True
    backbone: str = "smallcnn"          # used for RGB normalization lookup
    include_test: bool = False
    test_per_class: int = 3
    excluded_folders: list[str] = field(default_factory=list)
    included_folders: list[str] = field(default_factory=list)
    split_file: Optional[str] = None
    save_splits_flag: bool = False
    # --- metadata-as-text embedding options ---
    meta_encoder: str = "none"          # none | flair
    meta_model_name: str = "jhu-clsp/ettin-encoder-17m"
    meta_layers: str = "-2"
    meta_text_template: str = "compact"  # compact | kv | json
    meta_batch_size: int = 64

# ---- Model choice (what to build) ----
@dataclass
class ModelConfig:
    model: str = "film"                      # late | film | xattn
    backbone: str = "smallcnn"
    token_stage: int = -2
    pred_activation: str = "sigmoid_eps"     # none | sigmoid | sigmoid_eps | tanh01
    activation_eps: float = 1e-3
    color_space: str = "rgb"                 # repeated here for clarity inside the loop

# ---- Optimization & loss scheduling ----
@dataclass
class LossConfig:
    mse_space: str = "rgb"                   # rgb | lab | same
    mse_weight_start: float = 1.0
    mse_weight_epochs: int = 75
    mse_space_switch: int = 75
    de_smooth_eps: float = 1e-6
    robust: str = "none"                     # none | huber | charbonnier
    huber_delta: float = 1.0
    charb_eps: float = 1e-3
    de_reduce: str = "mean"                  # mean | median

@dataclass
class LoopConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # --- SGD / schedule knobs (CLI maps here through RunConfig.from_namespace) ---
    optim: str = "adamw"           # ["adamw", "sgd"]
    momentum: float = 0.9          # used when optim == "sgd"
    lr_auto: bool = False          # if True with SGD: lr = 0.1*(batch/256)
    lr_schedule: str = "cosine"    # ["cosine", "paper_resnet", "plateau"]
    # -------------------------------------------------------------
    grad_clip: float = 1.0
    amp: bool = False
    accum_steps: int = 1
    verbose: int = 1

# ---- Logging & checkpointing ----
@dataclass
class LoggingConfig:
    logdir: str = "./runs"
    log_images_every: int = 5
    min_log_epoch: int = 0
    log_best_val: bool = True

@dataclass
class CheckpointConfig:
    outdir: str = "./checkpoints"
    ckpt_monitor: str = "de00"
    ckpt_mode: str = "min"
    save_top_k: int = 1
    save_last: bool = False
    ckpt_filename: Optional[str] = None
    prune_short_runs: bool = True
    min_keep_epochs: int = 25
    min_ckpt_epoch: int = 0

@dataclass
class EarlyStoppingConfig:
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    early_stop_monitor: str = "de00"
    early_stop_mode: str = "min"
    early_stop_min_epoch: int = 0
    restore_best_on_stop: bool = True

@dataclass
class SmokeConfig:
    smoke_enabled: bool = False              # turn on smoke assertions & summary
    abs_tol: float = 0.10              # absolute ΔE00 tolerance
    rel_tol: float = 0.05              # relative tolerance (fraction of baseline)
    baseline_dir: str = "./runs/smoke" # where baseline/result jsons live
    result_filename: str = "smoke_result.json"
    update_baseline: bool = False      # if true, write/refresh baseline

# ---- Top-level run config ----
@dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    loop: LoopConfig
    log: LoggingConfig
    ckpt: CheckpointConfig
    early: EarlyStoppingConfig
    smoke: SmokeConfig

    @classmethod
    def _from_namespace_to(cls, ns, target_cls):
        d = vars(ns)
        vals = {}
        for f in fields(target_cls):
            name = f.name
            if name in d:
                vals[name] = d[name]
            elif f.default is not MISSING:
                vals[name] = f.default
            elif getattr(f, "default_factory", MISSING) is not MISSING:
                vals[name] = f.default_factory()  # type: ignore
            else:
                raise TypeError(f"Missing required config key for {target_cls.__name__}: {name}")
        return target_cls(**vals)

    @classmethod
    def from_namespace(cls, ns) -> "RunConfig":
        data = cls._from_namespace_to(ns, DataConfig)
        # model.color_space should mirror data.color_space unless explicitly overridden
        model = cls._from_namespace_to(ns, ModelConfig)
        if "color_space" not in vars(ns):
            model.color_space = data.color_space
        loss = cls._from_namespace_to(ns, LossConfig)
        loop = cls._from_namespace_to(ns, LoopConfig)
        log  = cls._from_namespace_to(ns, LoggingConfig)
        ckpt = cls._from_namespace_to(ns, CheckpointConfig)
        early= cls._from_namespace_to(ns, EarlyStoppingConfig)
        smoke= cls._from_namespace_to(ns, SmokeConfig)
        return RunConfig(data=data, model=model, loss=loss, loop=loop, log=log, ckpt=ckpt, early=early, smoke=smoke)


