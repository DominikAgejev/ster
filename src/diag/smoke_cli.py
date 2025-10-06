# src/diag/smoke_cli.py
from __future__ import annotations
import argparse, torch
from . import D

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--attn_impl", default="eager")  # eager|sdpa on 1080 Ti
    args = p.parse_args()

    class _Dummy: pass
    run = _Dummy()
    run.model = _Dummy(); run.model.model="dummy"; run.model.backbone="resnet18"
    run.data  = _Dummy(); run.data.features="image+meta"; run.data.meta_encoder="e2e_text"

    # --- REQUIRED e2e_text knobs (no silent defaults) ---
    cfg = dict(
        meta_tokenizer="word",
        meta_max_len=64,
        meta_d_model=128,
        meta_n_layers=2,
        meta_n_heads=4,
        meta_ffn_mult=4.0,
        meta_dropout=0.1,
        meta_pad_id=0,           # <-- missing before; required
        meta_pool="mean",
        meta_attn_impl=args.attn_impl,
        # vocab source: either build OR path
        meta_build_vocab=True,
        meta_vocab_size=1000,
        # meta_vocab_path="/path/to/vocab.json",  # use this if build is False
    )
    for k, v in cfg.items(): setattr(run.data, k, v)

    D.enable(run, outdir=args.outdir)
    D.config_snapshot(run)
    D.validate.meta_config(run.data)
    D.validate.gpu_support(args.attn_impl)

    b = {
        "image": torch.zeros(2,3,128,128),
        "label": torch.zeros(2,3),
        "metadata": torch.zeros(2,0),
        "meta_tokens": torch.zeros(2,64, dtype=torch.long),
        "meta_mask": torch.ones(2,64, dtype=torch.bool),
    }
    D.validate.batch(b)
    D.batch.range_nan_report(b)
    print("[diag/smoke_cli] OK")

if __name__ == "__main__":
    main()
