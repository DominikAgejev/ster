# src/engine/eval_ckpt.py
from __future__ import annotations
import argparse, os, json, sys
import torch

# Local imports align with the existing training code
from .datamodule import DataModule, DataModuleConfig
from ..models import build_model
from ..metrics.losses import ColorLoss
from ..metrics.evaluate import evaluate

def _bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y"): return True
    if s in ("0", "false", "no", "n"): return False
    return default

def _parse_token_stage(ts):
    # Mirror training's permissive semantics
    if ts is None:
        return "ms"
    if isinstance(ts, (list, tuple)):
        return [int(x) for x in ts]
    s = str(ts).strip().lower()
    if s in ("", "ms", "multi", "auto", "none"):
        return "ms"
    if s.startswith("[") and s.endswith("]"):
        import json
        v = json.loads(s)
        if isinstance(v, list):
            return [int(x) for x in v]
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    try:
        return int(float(s))
    except Exception:
        return "ms"

def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluate a stored checkpoint (best-val) on TEST and write final_summary.json"
    )
    # Required
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt/.pth) with saved config + state_dict")
    p.add_argument("--images_dir", required=True)
    p.add_argument("--json_dir",   required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--test_split_file", required=True, help="Split JSON containing the TEST indices")

    # Optional overrides / bookkeeping (monitor must be explicit)
    p.add_argument("--monitor", required=True, help="Name of monitored val metric (for logging only)")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if omitted)")
    p.add_argument("--outdir", default=None, help="Folder to write final_summary.json (default: alongside ckpt)")
    p.add_argument("--out_json", default=None, help="Explicit path to write JSON (overrides --outdir)")
    # NEW: keep CLI compatible with sweep/winners; optional override
    p.add_argument(
        "--token_stage",
        default=None,
        help="Optional override for token conditioning stages. "
             "Examples: -3, -2, '1,2,3', '[-2]'."
    )
    return p

def main():
    args = build_argparser().parse_args()

    # ---------- Load checkpoint (must contain state_dict + config) ----------
    obj = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        print(f"[eval] Bad checkpoint (missing state_dict): {args.ckpt}", file=sys.stderr)
        sys.exit(2)

    ck_cfg = obj.get("config", {}) or {}
    if not ck_cfg:
        print("[eval] Checkpoint missing 'config' dict.", file=sys.stderr)
        sys.exit(2)
    epoch = obj.get("epoch", None)

    token_stage = _parse_token_stage(ck_cfg.get("token_stage", -2))
    # If CLI provided, override saved value (winners.py may pass it)
    if args.token_stage is not None:
        token_stage = _parse_token_stage(args.token_stage)

    # Pull model/data knobs (fall back to sensible defaults if older ckpts)
    model_name   = ck_cfg.get("model", "film")
    backbone     = ck_cfg.get("backbone", "smallcnn")
    token_stage  = token_stage  
    color_space  = ck_cfg.get("color_space", "rgb")
    features     = ck_cfg.get("features", "image")
    pretrained   = _bool(ck_cfg.get("pretrained", True))
    hidden_cnt   = int(ck_cfg.get("hidden_classes_cnt", 0))
    workers      = int(ck_cfg.get("workers", 0))
    batch_size   = int(ck_cfg.get("batch_size", 32))
    seed         = int(ck_cfg.get("seed", 100))
    group_split  = ck_cfg.get("group_split", None)
    included     = ck_cfg.get("included_folders", None)
    excluded     = ck_cfg.get("excluded_folders", None)
    # Loss/pred activation defaults align with training defaults
    mse_space    = ck_cfg.get("mse_space", "rgb")
    mse_weight   = float(ck_cfg.get("mse_weight_start", ck_cfg.get("mse_weight", 1.0)))
    smooth_eps   = float(ck_cfg.get("de_smooth_eps", ck_cfg.get("smooth_eps", 0.0)))
    pred_act     = ck_cfg.get("pred_activation", "sigmoid_eps" if color_space == "rgb" else "none")
    act_eps      = float(ck_cfg.get("activation_eps", 1e-3))

    # Note: if 'pretrained' was missing above, we already defaulted to True.
    assumed_pretrained = None
    if "pretrained" not in ck_cfg:
        assumed_pretrained = True

    # Optional text/meta encoder fields (no opinionated defaults)
    meta_encoder       = ck_cfg.get("meta_encoder", None)
    meta_model_name    = ck_cfg.get("meta_model_name", None)
    meta_layers        = ck_cfg.get("meta_layers", None)
    meta_text_template = ck_cfg.get("meta_text_template", None)
    meta_batch_size    = int(ck_cfg.get("meta_batch_size", 64))

    # ---------- Data module (test-only) ----------
    dm = DataModule(DataModuleConfig(
        images_dir=args.images_dir,
        json_dir=args.json_dir,
        labels_csv=args.labels_csv,
        batch_size=batch_size,
        val_split=0.0,
        workers=workers,
        seed=seed,
        hidden_classes_cnt=hidden_cnt,
        group_split=None,                  # not used for test
        color_space=color_space,
        features=features,
        pretrained=pretrained,
        backbone=backbone,
        include_test=False,
        excluded_folders=excluded,
        included_folders=included,
        test_per_class=3,                  # irrelevant for fixed split
        split_file=args.test_split_file,   # must contain "test"
        save_splits_flag=False,
        # text/meta encoder passthrough
        meta_encoder=meta_encoder,
        meta_model_name=meta_model_name,
        meta_layers=meta_layers,
        meta_text_template=meta_text_template,
        meta_batch_size=meta_batch_size,
    )).setup()
    test_loader = dm.test_dataloader()
    if test_loader is None:
        print("[eval] No TEST loader from provided split file.", file=sys.stderr)
        sys.exit(3)

    # ---------- Model ----------
    meta_dim = getattr(dm, "meta_dim", None) or obj.get("meta_dim", 0) or 0

    # NEW: infer attn_dim from checkpoint (only for xattn/CrossAttnNet)
    attn_dim = None
    if str(model_name).lower() in ("xattn", "cross", "crossattn"):
        sd = obj.get("state_dict", {})
        if "attn.in_proj_weight" in sd:
            # MultiheadAttention in_proj_weight: [3d, d] → d is second dim
            attn_dim = int(sd["attn.in_proj_weight"].shape[1])
        elif "attn.out_proj.weight" in sd:
            # out_proj.weight: [d, d] → first dim is d
            attn_dim = int(sd["attn.out_proj.weight"].shape[0])
        elif "q_vec" in sd:
            # meta-free variant: q_vec: [d]
            attn_dim = int(sd["q_vec"].shape[0])
        if attn_dim is not None:
            print(f"[eval] inferred attn_dim={attn_dim} from checkpoint.")

    # Build exactly like training; for xattn also pass attn_dim if inferred.
    build_kwargs = dict(
        backbone=backbone,
        meta_dim=meta_dim,
        token_stage=token_stage,
        pretrained=pretrained,
    )
    if attn_dim is not None and str(model_name).lower() in ("xattn", "cross", "crossattn"):
        build_kwargs["attn_dim"] = attn_dim

    model = build_model(model_name, **build_kwargs)

    missing, unexpected = model.load_state_dict(obj["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[eval][warn] State dict load: missing={len(missing)}, unexpected={len(unexpected)}")


    # ---------- Criterion & device ----------
    criterion = ColorLoss(
        input_space=color_space,
        mse_space=mse_space,
        mse_weight=mse_weight,
        smooth_eps=smooth_eps,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- Run TEST eval ----------
    metrics = evaluate(
        model, test_loader, device=device, criterion=criterion,
        color_space=color_space, pred_activation=pred_act, activation_eps=act_eps
    )

    # ---------- Persist summary ----------
    # Prefer caller's explicit path; else honor --outdir; else fail fast
    if args.out_json:
        out_json = args.out_json
        outdir = os.path.dirname(os.path.abspath(out_json))
    elif args.outdir:
        outdir = args.outdir
        out_json = os.path.join(outdir, "final_summary.json")
    else:
        print("[eval] Must provide --out_json or --outdir for summary output.", file=sys.stderr)
        sys.exit(2)
    os.makedirs(outdir, exist_ok=True)

    cfg_for_json = {
        "model": model_name,
        "backbone": backbone,
        "token_stage": token_stage,
        "pretrained": bool(pretrained),
        "features": features,
        "color_space": color_space,
        "batch_size": batch_size,
        "val_split": 0.0,
        "group_split": group_split,
        "included_folders": included,
        "excluded_folders": excluded,
        "meta_encoder": meta_encoder,
        "meta_model_name": meta_model_name,
        "meta_layers": meta_layers,
        "meta_text_template": meta_text_template,
        "attn_dim": int(attn_dim) if attn_dim is not None else None,
    }
    payload = {
        "epoch": int(epoch) if epoch is not None else None,
        "config": cfg_for_json,
        "metric_kind": "test",
        "test_de00": float(metrics.get("de00")) if metrics.get("de00") is not None else None,
        "pretrained": pretrained,
        # NEW: allow downstream tools (like the ΔE analyzer) to resolve the actual model file:
        "ckpt_path": str(args.ckpt),
    }
    if assumed_pretrained is not None:
        payload.setdefault("_assumptions", {})["pretrained"] = assumed_pretrained
    if metrics.get("mse_lab") is not None: payload["test_mse_lab"] = float(metrics["mse_lab"])
    if metrics.get("mse_rgb") is not None: payload["test_mse_rgb"] = float(metrics["mse_rgb"])

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"[eval][ok] Test eval complete. Wrote → {out_json}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
