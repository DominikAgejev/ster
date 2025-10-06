# src/engine/sweep/winners.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Iterable, Optional
import os, json, sys, subprocess, glob, math

from ..split_utils import ensure_auto_splits_for_run
from .utils import stable_hash, build_experiment_cmd

DEFAULT_ALLOWED_KEYS: set[str] = {
    "model", "backbone", "features", "color_space",
    "meta_encoder", "meta_layers", "meta_text_template",
    "included_folders", "excluded_folders",
    "seed", "group_split",
    "robust", "token_stage",
    "run_id", "ckpt_path"
}

def stage_outdirs(base: dict, stage_name: str) -> dict:
    """Append /<stage_name> to outdir/logdir to isolate a stage's artifacts."""
    z = dict(base or {})
    if "outdir" in z:
        z["outdir"] = os.path.join(z["outdir"], stage_name)
    if "logdir" in z:
        z["logdir"] = os.path.join(z["logdir"], stage_name)
    return z


def _rehydrate_listish(v: Any) -> Any:
    """Turn simple CSV-ish strings back into lists for CLI hparams that expect multiple values."""
    if isinstance(v, str) and ("," in v) and ("[" not in v) and ("{" not in v):
        xs = [s.strip() for s in v.split(",") if s.strip()]
        return xs
    return v


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def select_winners_from_csv(csv_path: str,
                            keep_top_k: int | None,
                            keep_top_frac: float | None,
                            mode: str = "min",
                            allowed_keys: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """
    Read the per-run summary CSV and return winner rows as sanitized dicts.
    - Drop duplicate configs (keep best) before keep_top filters.
    - Keep only whitelisted keys to avoid overriding base config with NaNs.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    asc = (mode == "min")
    df = df.sort_values("best_metric", ascending=asc)

    # Winner identity: use all provided keys (typically ALL YAML grid keys)
    if allowed_keys is None:
        allowed_keys = DEFAULT_ALLOWED_KEYS
    allowed_keys = list(dict.fromkeys(list(allowed_keys) + list(DEFAULT_ALLOWED_KEYS)))

    # ---- Canonicalize ONLY for identity (do not overwrite the column) ----
    canon_col = None
    if "token_stage" in df.columns:
        def _canon_ts_for_id(x):
            # Lists/tuples -> stable "1+2+3"; strings left as-is (keep "-2" intact)
            if isinstance(x, (list, tuple)):
                return "+".join(str(int(v)) for v in x)
            return str(x).strip()
        df["_ts_id"] = df["token_stage"].apply(_canon_ts_for_id)
        canon_col = "_ts_id"

    # IMPORTANT: do NOT dedup on ckpt_path; keep runs distinct across token_stage etc.
    id_cols = []
    for c in allowed_keys:
        if c not in df.columns:
            continue
        if c == "ckpt_path":
            continue
        if c == "token_stage" and canon_col is not None:
            id_cols.append(canon_col)  # use shadow column for identity
        else:
            id_cols.append(c)

    if id_cols:
        df = df.drop_duplicates(subset=id_cols, keep="first")

    # Apply top-k/fract filters
    if keep_top_k is not None:
        keep_top_k = max(1, int(keep_top_k))
        df = df.head(keep_top_k)
    elif keep_top_frac is not None:
        frac = max(0.0, min(1.0, float(keep_top_frac)))
        k = max(1, int(round(len(df) * frac)))
        df = df.head(k)

    if allowed_keys is None:
        allowed_keys = DEFAULT_ALLOWED_KEYS
    allowed_keys = set(allowed_keys) | DEFAULT_ALLOWED_KEYS

    winners: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for k in allowed_keys:
            if k not in row.index:
                continue
            v = row[k]
            if _is_nan(v):
                continue
            if k in ("included_folders", "excluded_folders"):
                v = _rehydrate_listish(v)
            rec[k] = v
        winners.append(rec)

    # De-duplicate identical checkpoints picked in multiple groups
    uniq, seen = [], set()
    for w in winners:
        key = (w.get("ckpt_path"), tuple((k, w.get(k)) for k in id_cols if k in w))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
    winners = uniq

    return winners


def _find_best_ckpt_fallback(stage_out_root: str, ckpt_hint: str | None) -> str | None:
    """
    Fallback only if the summary didn't carry ckpt_path.
    Prefer '*best*.pt'; otherwise pick the newest .pt under the stage root.
    """
    search_dir = os.path.dirname(ckpt_hint) if ckpt_hint else stage_out_root
    cand = glob.glob(os.path.join(search_dir, "**", "*.pt"), recursive=True)
    if not cand:
        return None
    best = [p for p in cand if "best" in os.path.basename(p).lower()]
    pool = best or cand
    pool.sort(key=lambda p: (-os.path.getmtime(p), os.path.basename(p)))
    return pool[0]


def eval_winners_on_test(
    winners: List[Dict[str, Any]],
    *,
    base_cfg: Dict[str, Any],
    stage_outdir: str,
    stage_logdir: str,
    monitor: str,
    test_split_file: str,
    train_full_split_file: str | None,
    split_root: str,
    retrain_for_test: bool = False,
    eval_subdir: str = "final_test",
    tag_keys: Iterable[str] | None = None,
) -> Tuple[int, int]:
    """
    DEFAULT path: evaluate saved best-val checkpoint for each winner on TEST.
    Legacy opt-in: retrain on train_full (non-test) and run a test eval at the end.

    Returns: (ok_count, fail_count)
    """
    def _format_token_stage_arg(ts):
        if isinstance(ts, (list, tuple)):
            return "[" + ",".join(str(int(v)) for v in ts) + "]"
        return str(ts)

    ok, fail = 0, 0
    final_root = os.path.join(stage_outdir, eval_subdir)
    os.makedirs(final_root, exist_ok=True)

    for w in winners:
        # Merge a sanitized winner over base
        run_cfg = dict(base_cfg or {})
        run_cfg.update(w)
        # Prefer explicit run_id; otherwise use GRID keys to avoid collisions
        if w.get("run_id"):
            run_tag = str(w["run_id"])
        else:
            keys = [k for k in (list(tag_keys) if tag_keys else []) if k in run_cfg]
            if not keys:
                # conservative fallback (old behavior)
                keys = [k for k in ("model","backbone","features","meta_encoder") if k in run_cfg]
            run_tag = stable_hash({k: run_cfg.get(k) for k in keys})

        # Ensure per-filter splits if requested via 'AUTO'
        if str(test_split_file).upper() == "AUTO":
            test_json, train_full_json = ensure_auto_splits_for_run(run_cfg, split_root=split_root)
        else:
            test_json, train_full_json = test_split_file, train_full_split_file

        if not test_json or not os.path.exists(test_json):
            print(f"[eval][warn] Missing test split for {run_tag}; skipping.")
            fail += 1
            continue

        outdir = os.path.join(final_root, run_tag)
        os.makedirs(outdir, exist_ok=True)

        if not retrain_for_test:
            # Evaluate BEST checkpoint on test via src.engine.eval_ckpt
            ckpt = w.get("ckpt_path")
            # Some rows (from --include_json) point at final_summary.json — resolve to the real .pt
            def _ckpt_from_json(p: str) -> str | None:
                try:
                    if not p or not p.endswith(".json") or not os.path.isfile(p):
                        return None
                    with open(p, "r") as f:
                        rec = json.load(f) or {}
                    cp = rec.get("ckpt_path")
                    return cp if (isinstance(cp, str) and cp.endswith((".pt", ".pth")) and os.path.exists(cp)) else None
                except Exception:
                    return None

            if isinstance(ckpt, str) and ckpt.endswith(".json"):
                ckpt = _ckpt_from_json(ckpt)
            if not ckpt or not os.path.exists(ckpt):
                ckpt = _find_best_ckpt_fallback(stage_outdir, ckpt)
            if not ckpt:
                print(f"[eval][warn] No checkpoint found for {run_tag}")
                fail += 1
                continue

            out_json = os.path.join(outdir, "final_summary.json")
            cmd = [
                sys.executable, "-m", "src.engine.eval_ckpt",
                "--ckpt", ckpt,
                "--images_dir", str(run_cfg["images_dir"]),
                "--json_dir",   str(run_cfg["json_dir"]),
                "--labels_csv", str(run_cfg["labels_csv"]),
                "--test_split_file", str(test_json),
                "--monitor", str(monitor),
                "--out_json", out_json,
            ]
            # Pass token_stage ONLY if the winner row set it (avoid taking base default like "-3")
            ts_row = w.get("token_stage", None)
            if ts_row is not None and str(ts_row).strip().lower() not in {"", "nan"}:
                cmd += ["--token_stage", _format_token_stage_arg(ts_row)]

            print("[eval] RUN:", " ".join(cmd))
            rc = subprocess.call(cmd)
            if rc == 0 and os.path.exists(out_json):
                ok += 1
            else:
                fail += 1
            continue

        # Legacy path: retrain on train_full, then evaluate on test at the end.
        if not train_full_json or not os.path.exists(train_full_json):
            print(f"[eval][warn] --retrain_for_test set but train_full missing for {run_tag}; skipping.")
            fail += 1
            continue

        cfg2 = dict(run_cfg)
        cfg2.update({
            "split_file": train_full_json,         # train on all non-test indices
            "include_test": False,
            "group_split": None,
            "save_splits_flag": False,
            "outdir": outdir,
            "logdir": os.path.join(stage_logdir, eval_subdir, run_tag),
            "test_split_file": str(test_json),
            "eval_test_after": True,
            "ckpt_monitor": str(monitor),
        })
        if tag_keys:
            cfg2["tag_keys"] = list(tag_keys)

        cmd = build_experiment_cmd(cfg2)
        print("[eval][retrain] RUN:", " ".join(cmd))
        rc = subprocess.call(cmd)
        ok += int(rc == 0)
        fail += int(rc != 0)

    return ok, fail
