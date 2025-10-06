# src/diag/validate.py
from __future__ import annotations
import torch
from typing import Any


class Validator:
    def __init__(self, state, log):
        self.s, self.log = state, log

    def meta_config(self, meta_cfg: Any):
        if not self.s.enabled:
            return
        enc = getattr(meta_cfg, "meta_encoder", "none")
        if enc != "e2e_text":
            return

        # Per-field validation (0 is valid for pad_id)
        def _req_str(name): 
            v = getattr(meta_cfg, name, None)
            return (v is None) or (isinstance(v, str) and v.strip() == "")
        def _req_posint(name):
            v = getattr(meta_cfg, name, None)
            try:
                return v is None or int(v) <= 0
            except Exception:
                return True
        def _req_posfloat(name):
            v = getattr(meta_cfg, name, None)
            try:
                return v is None or float(v) <= 0.0
            except Exception:
                return True
        def _present(name):
            return getattr(meta_cfg, name, None) is not None

        missing: list[str] = []
        # tokenizer & sequence
        if _req_str("meta_tokenizer"): missing.append("meta_tokenizer")
        if _req_posint("meta_max_len"): missing.append("meta_max_len (>0)")
        # encoder dims
        if _req_posint("meta_d_model"):  missing.append("meta_d_model (>0)")
        if _req_posint("meta_n_layers"): missing.append("meta_n_layers (>0)")
        if _req_posint("meta_n_heads"):  missing.append("meta_n_heads (>0)")
        if _req_posfloat("meta_ffn_mult"): missing.append("meta_ffn_mult (>0)")
        # regularization/pooling/attn
        if _req_posfloat("meta_dropout"): missing.append("meta_dropout (>0)")
        if _req_str("meta_pool"): missing.append("meta_pool")
        if _req_str("meta_attn_impl"): missing.append("meta_attn_impl")
        # pad id: 0 is VALID
        if not _present("meta_pad_id"):
            missing.append("meta_pad_id (0 is valid)")

        # vocab source: either build OR path
        build = bool(getattr(meta_cfg, "meta_build_vocab", False))
        if build:
            if _req_posint("meta_vocab_size"):
                missing.append("meta_vocab_size (>0 when meta_build_vocab=True)")
        else:
            if _req_str("meta_vocab_path"):
                missing.append("meta_vocab_path (or set meta_build_vocab=True)")

        if missing:
            raise TypeError(f"[meta-e2e] Missing/invalid keys: {missing}")

        self.log.write("meta_cfg_ok", encoder="e2e_text")

    def gpu_support(self, attn_impl: str, require_cc=(6, 1)):
        if not self.s.enabled:
            return
        if not torch.cuda.is_available():
            self.log.write("gpu", note="cuda_not_available")
            return
        cc = torch.cuda.get_device_capability()
        self.log.write("gpu", cc=cc, device=torch.cuda.get_device_name())
        bad = attn_impl and attn_impl.lower() in ("flash", "xformers")
        if bad and cc < (8, 0):
            raise RuntimeError(
                f"[meta-e2e] attn_impl='{attn_impl}' not supported on cc={cc}. "
                "Use 'eager' or 'sdpa' on Pascal (e.g., GTX 1080 Ti)."
            )

    def batch(self, b: dict, require=("image", "label")):
        if not self.s.enabled:
            return
        for k in require:
            if k not in b:
                raise RuntimeError(f"[batch] missing key: {k}")
        img = b["image"]; lbl = b["label"]
        if not (img.ndim == 4 and img.shape[1] == 3):
            raise RuntimeError(f"[batch] image bad shape: {tuple(img.shape)} expected [B,3,H,W]")
        if not (lbl.ndim == 2 and lbl.shape[1] == 3):
            raise RuntimeError(f"[batch] label bad shape: {tuple(lbl.shape)} expected [B,3]")
        md = b.get("metadata", None)
        if md is not None and md.ndim != 2:
            raise RuntimeError(f"[batch] metadata must be 2-D; got {tuple(md.shape)}")

        has_tok = ("meta_tokens" in b) or ("meta_mask" in b)
        if has_tok:
            if "meta_tokens" not in b or "meta_mask" not in b:
                raise RuntimeError("[batch] meta_tokens/meta_mask must both be present together")
            tt, mm = b["meta_tokens"], b["meta_mask"]
            if not (tt.dtype == torch.long and mm.dtype == torch.bool):
                raise RuntimeError("[batch] expected meta_tokens=int64 and meta_mask=bool")

    def forward_io(self, img, meta_vec_or_num, tokens: dict | None, preds, expected_meta_dim: int | None):
        if not self.s.enabled:
            return
        if preds.ndim != 2 or preds.shape[1] != 3:
            raise RuntimeError(f"[forward] preds shape {tuple(preds.shape)} expected [B,3]")
        if tokens:
            tt = tokens.get("meta_tokens"); mm = tokens.get("meta_mask")
            if tt is None or mm is None:
                raise RuntimeError("[forward] encoder active but tokens/mask missing")
            if tt.ndim != 2 or mm.ndim != 2 or tt.shape != mm.shape:
                raise RuntimeError(f"[forward] token/mask bad shapes: {tuple(tt.shape)} vs {tuple(mm.shape)}")
        # meta_dim mismatch checks are enforced in the wrapper; we only do light checks here
        self.log.write("forward_ok", B=int(img.shape[0]), meta_dim=expected_meta_dim)
