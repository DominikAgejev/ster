# src/engine/experiment.py
#!/usr/bin/env python3
from __future__ import annotations

from .datamodule import DataModule, DataModuleConfig
from .train import fit, build_argparser as build_train_argparser
from .config import RunConfig, DataConfig
from ..models.backbones import normalize_backbone_name
from ..data.splitters import load_splits, train_val_split, save_splits, normalize_folder_filters

def build_argparser():
    p = build_train_argparser()  # reuse all existing flags for back-compat
    # Keys to include in run_tag (provided by sweep CLI from the YAML grid)
    if not any("--tag_keys" in a.option_strings for a in p._actions):
        p.add_argument("--tag_keys", nargs="+", default=None,
                       help="Config keys to include in the run_tag (passed from sweep grid).")
    # Split persistence knobs for sweeps (kept)
    if not any("--split_file" in a.option_strings for a in p._actions):
        p.add_argument("--split_file", type=str, default=None)
    if not any("--save_splits" in a.option_strings for a in p._actions):
        p.add_argument("--save_splits", dest="save_splits", action="store_true", default=False)
    # Optional automatic test evaluation
    p.add_argument("--eval_test_after", action="store_true", default=False,
                   help="After training, evaluate on test split and include it in the result.")
    p.add_argument("--test_split_file", type=str, default=None,
                   help="JSON with 'test' indices. If set, training uses the remaining samples.")
    p.add_argument("--fallback_test_per_class", type=int, default=3,
                   help="If no test_split_file, carve test per class (persisted).")
    return p

def main():
    args = build_argparser().parse_args()
    run = RunConfig.from_namespace(args)

    # --- Per-run subdirectory to isolate checkpoints & logs ---
    import os, re
    def _slug(x: str) -> str:
        # Preserve leading '-' (so token_stage=-2 stays -2)
        return re.sub(r"[^A-Za-z0-9_+.\-]+", "-", str(x)).strip()
    
    def _get_val_from_run(k: str):
        # look across nested config sections
        for sect in (run.data, run.model, run.loss, run.loop, run.log, run.ckpt, run.early, run.smoke):
            if hasattr(sect, k):
                return getattr(sect, k)
        # final fallback to argparse namespace (rare)
        return getattr(args, k, None)

    # Build a descriptive tag; include identity bits that matter
    _inc    = run.data.included_folders or "all"
    bb_norm = normalize_backbone_name(run.model.backbone or run.data.backbone or "smallcnn")
    _layers = getattr(run.model, "meta_layers", None) or "none"
    # Decide "ms" vs "ts<idx>" from the CLI arg itself (no backbone needed here)
    ts_val = getattr(args, "token_stage", getattr(run.model, "token_stage", None))
    def _is_ms(ts):
        return (isinstance(ts, (list, tuple)) and len(ts) > 1) or \
               (isinstance(ts, str) and ts in ("ms", "multi", "auto", ""))
    if _is_ms(ts_val):
        ts_tag = "ms"
    elif isinstance(ts_val, int):
        ts_tag = f"ts{ts_val}"
    else:
        ts_tag = "ts-1"  # conservative default label

    run_tag = "__".join([
        _slug(run.model.model),
        _slug(bb_norm),
        _slug(run.data.features),
        _slug(_inc),
        f"layers{_slug(_layers)}",
        ts_tag,
        f"s{run.data.seed}",
    ])

    # Add grid-driven keys. Format robust trio as: __<robust>__<delta>__<reduce>
    tag_keys = list(getattr(args, "tag_keys", []) or [])
    extras: list[str] = []

    # --- robust family in compact form ---
    if "robust" in tag_keys:
        rb = str(_get_val_from_run("robust") or "").lower()
        if rb and rb != "none":
            extras.append(_slug(rb))  # e.g., "huber", "charbonnier"
            if rb == "huber":
                hd = _get_val_from_run("huber_delta")
                if hd is not None:
                    extras.append(_slug(str(hd).replace(".", "_")))  # 0.5 -> 0_5
            red = _get_val_from_run("de_reduce") or _get_val_from_run("reduce")
            if red:
               extras.append(_slug(str(red)))

    # --- any remaining tag keys as k=v for readability ---
    already = {"robust", "huber_delta", "de_reduce", "reduce"}
    for k in tag_keys:
        if k in already:
            continue
        v = _get_val_from_run(k)
        if v is None:
            continue
        extras.append(f"{_slug(k)}={_slug(str(v))}")

    if extras:
        run_tag = run_tag + "__" + "__".join(extras)
    # Isolate outputs
    run.ckpt.outdir = os.path.join(run.ckpt.outdir, run_tag)
    run.log.logdir  = os.path.join(run.log.logdir,  run_tag)

    # Build DataModule from the *data* part only
    d: DataConfig = run.data
    d.backbone = bb_norm   
    # If a test split is provided, training should exclude it.
    train_split_file = d.split_file
    test_loader = None
    test_indices = None

    if args.test_split_file:
        # Build a *test-only* DataModule to align dataset & indices
        test_dm = DataModule(DataModuleConfig(
            images_dir=d.images_dir, json_dir=d.json_dir, labels_csv=d.labels_csv,
            batch_size=run.loop.batch_size, val_split=d.val_split, workers=d.workers,
            seed=d.seed, hidden_classes_cnt=d.hidden_classes_cnt, group_split=None,
            color_space=d.color_space, features=d.features, pretrained=d.pretrained,
            backbone=d.backbone or run.model.backbone, include_test=False,
            excluded_folders=d.excluded_folders, included_folders=d.included_folders,
            test_per_class=args.fallback_test_per_class, split_file=args.test_split_file,
            save_splits_flag=False,
            meta_encoder=d.meta_encoder, meta_model_name=d.meta_model_name,
            meta_layers=d.meta_layers, meta_text_template=d.meta_text_template,
            meta_batch_size=d.meta_batch_size,
        )).setup()
        test_loader = test_dm.test_dataloader()
        
        # Load the TEST indices so we can *exclude* them from train/val
        sp = load_splits(args.test_split_file)
        test_indices = sp.get("test", None) or []

    # If we have a provided TEST, compute train/val excluding it and write a temp split file
    if test_indices:
        exc = normalize_folder_filters(d.excluded_folders)
        inc = normalize_folder_filters(d.included_folders)
        full_ds = test_dm.ds  # same dataset view (paths/ordering) used to pick TEST
        tr_idx, va_idx = train_val_split(
            ds=full_ds, val_split=float(d.val_split), seed=int(d.seed),
            test_indices=test_indices, hidden_classes_cnt=int(d.hidden_classes_cnt),
            excluded_folders=exc, included_folders=inc, group_split=d.group_split,
        )
        os.makedirs(run.ckpt.outdir, exist_ok=True)
        train_split_file = os.path.join(run.ckpt.outdir, "trainval_from_test.json")
        save_splits(train_split_file, train_idx=tr_idx, val_idx=va_idx, test_idx=test_indices)


    # Build the training DataModule (now using the temp split if created)
    dm = DataModule(DataModuleConfig(
        images_dir=d.images_dir, json_dir=d.json_dir, labels_csv=d.labels_csv,
        batch_size=run.loop.batch_size, val_split=d.val_split, workers=d.workers,
        seed=d.seed, hidden_classes_cnt=d.hidden_classes_cnt, group_split=d.group_split,
        color_space=d.color_space, features=d.features, pretrained=d.pretrained,
        backbone=d.backbone or run.model.backbone, include_test=d.include_test,
        excluded_folders=d.excluded_folders, included_folders=d.included_folders,
        test_per_class=d.test_per_class, split_file=train_split_file,
        save_splits_flag=d.save_splits_flag or getattr(args, "save_splits", False),
        meta_encoder=d.meta_encoder, meta_model_name=d.meta_model_name,
        meta_layers=d.meta_layers, meta_text_template=d.meta_text_template,
        meta_batch_size=d.meta_batch_size,
    )).setup()

    # Optional test DataModule/loader (do not clobber if already built above)
    if 'test_loader' not in locals():
        test_loader = None
    if bool(args.eval_test_after) and (test_loader is None):
        test_split = args.test_split_file
        if test_split:
            test_dm = DataModule(DataModuleConfig(
                images_dir=d.images_dir,
                json_dir=d.json_dir,
                labels_csv=d.labels_csv,
                batch_size=run.loop.batch_size,
                val_split=d.val_split,
                workers=d.workers,
                seed=d.seed,
                hidden_classes_cnt=d.hidden_classes_cnt,
                group_split=None,
                color_space=d.color_space,
                features=d.features,
                pretrained=d.pretrained,
                backbone=d.backbone or run.model.backbone,
                include_test=False,
                excluded_folders=d.excluded_folders,
                included_folders=d.included_folders,
                test_per_class=args.fallback_test_per_class,
                split_file=test_split,
                save_splits_flag=False,
                # mirror text/meta encoder knobs
                meta_encoder=d.meta_encoder,
                meta_model_name=d.meta_model_name,
                meta_layers=d.meta_layers,
                meta_text_template=d.meta_text_template,
                meta_batch_size=d.meta_batch_size,
            )).setup()
            test_loader = test_dm.test_dataloader()
        elif d.include_test:
            # User opted for include_test; reuse this dm's test loader
            test_loader = dm.test_dataloader()
        else:
            print("[warn] --eval_test_after requested but no test split provided. "
                  "Enable include_test or pass --test_split_file.")

    # Train once; if test_loader is provided and eval_test_after is true,
    # the trainer will evaluate on test **after** training.
    fit(run, dm, test_loader=test_loader, eval_test_after=bool(args.eval_test_after))

if __name__ == "__main__":
    main()
