# src/engine/sweep/utils.py
from __future__ import annotations
from typing import Any, Dict, List
import hashlib, json, os, subprocess

# ---------------------------
# Public helpers
# ---------------------------

def stable_hash(d: Dict[str, Any], n: int = 12) -> str:
    """Stable short hash for dict-like configs."""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:n]


def state_dir_for_config(config_path: str) -> str:
    """
    Where to put sweep state markers for a config file.
    Example: configs/sweep/smoke.yaml -> configs/sweep/.sweep_state_smoke
    """
    base = os.path.basename(config_path)
    name, _ = os.path.splitext(base)
    return os.path.join(os.path.dirname(config_path), f".sweep_state_{name}")


def marker_path(state_dir: str, run_id: str, kind: str) -> str:
    """
    `kind` ∈ {"skip","success","fail"}. Content is a tiny JSON blob.
    """
    os.makedirs(state_dir, exist_ok=True)
    fname = f"{run_id}__{kind}.json"
    return os.path.join(state_dir, fname)


def write_marker(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_subprocess(cmd: list[str], cwd: str | None = None) -> int:
    """Launch a command; return process return code."""
    return subprocess.run(cmd, cwd=cwd).returncode

# -----------------------
# Helpers
# -----------------------
def _coerce_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    s = str(v)
    if not s:
        return []
    # allow comma-separated or whitespace-separated
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [x for x in s.split() if x]


def _if_token(included_folders: List[str]) -> str:
    """
    Build the IF=<...> token used by AUTO split naming, e.g.
    included_folders=["focus","iphone"] -> "focus-iphone"
    included_folders=["focus/iphone"]   -> "focus-iphone"
    """
    parts: List[str] = []
    for item in included_folders:
        parts.extend(str(item).replace("/", "-").split("-"))
    parts = [p for p in (x.strip() for x in parts) if p]
    return "-".join(parts) if parts else "ALL"


def _resolve_auto_test_split(base_cfg: Dict[str, Any], split_root: str) -> str:
    """
    Deterministically re-create the TEST split filename used by --test_split_file AUTO.
    Pattern (as seen in logs):
      ./splits/sweep/IF=<IF>__EF=NONE__HC=<HC>__SEED=<SEED>/test.json
    """
    seed = int(base_cfg.get("seed", base_cfg.get("data", {}).get("seed", 0)) or 0)
    hidden = int(base_cfg.get("hidden_classes_cnt", 0) or 0)
    included = _coerce_list(base_cfg.get("included_folders", []))
    IF = _if_token(included)
    subdir = f"IF={IF}__EF=NONE__HC={hidden}__SEED={seed}"
    return os.path.join(split_root, subdir, "test.json")


def _default_results_root_for_config(cfg_path: str, label_dir: str | None = None) -> str:
    """
    Mirror /configs/... to /results/... for nicer organization.
    E.g. /repo/src/configs/reports/foo.yaml -> /repo/src/results/reports/foo[/label_dir]
    """
    cfg_name = os.path.splitext(os.path.basename(cfg_path or "config"))[0]
    abs_cfg_dir = os.path.dirname(os.path.abspath(cfg_path)) if cfg_path else os.getcwd()
    base_root = abs_cfg_dir.replace("/configs/", "/results/")
    if label_dir:
        return os.path.join(base_root, cfg_name, label_dir)
    return os.path.join(base_root, cfg_name)


# ---------------------------
# Command builder (robust boolean & "-" value handling)
# ---------------------------

# experiment.py booleans that are "store_true": presence => True, absence => False
STORE_TRUE_FLAGS: set[str] = {
    "lr_auto",
    "prune_short_runs",
    "save_last",
    "save_splits",
    "include_test",
    "amp",
    "smoke_enabled",
    "update_baseline",
    "eval_test_after",
}

# experiment.py paired booleans that have explicit negative flags
PAIRED_BOOL_FLAGS: dict[str, tuple[str, str]] = {
    # key -> (positive_flag, negative_flag)
    "pretrained": ("--pretrained", "--no-pretrained"),
    "restore_best_on_stop": ("--restore_best_on_stop", "--no-restore_best_on_stop"),
}

# args that accept multiple values (space-separated) rather than a single CSV token
MULTI_VALUE_FLAGS: set[str] = {
    "included_folders",
    "excluded_folders",
    "tag_keys",
}

def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)

def _as_listish(v: Any) -> list[str]:
    """Turn v into a list[str] for multi-value flags."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    # allow CSV strings as a convenience
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]

def _emit_kv(cmd: list[str], k: str, v: Any) -> None:
    """
    Add a key-value pair to `cmd`, safely handling values that begin with '-'.
    """
    sval = str(v)
    if sval.startswith("-"):
        # emit with equals so argparse doesn't treat value as a new flag
        cmd.append(f"--{k}={sval}")
    else:
        cmd.extend([f"--{k}", sval])

def build_experiment_cmd(cfg: Dict[str, Any]) -> list[str]:
    """
    Compose `python -m src.engine.experiment ...` from the given merged run config.
    - store_true flags: include only when truthy
    - paired booleans: emit the correct positive/negative flag
    - multi-value flags: expand into multiple args
    - values beginning with '-' are passed as --key=value to avoid argparse confusion
    """
    required = ("images_dir", "json_dir", "labels_csv")
    missing = [r for r in required if r not in cfg]
    if missing:
        raise KeyError(f"Missing required run keys: {missing}")

    cmd: list[str] = [
        "python", "-m", "src.engine.experiment",
        "--images_dir", str(cfg["images_dir"]),
        "--json_dir",   str(cfg["json_dir"]),
        "--labels_csv", str(cfg["labels_csv"]),
    ]
    if "outdir" in cfg: _emit_kv(cmd, "outdir", cfg["outdir"])
    if "logdir" in cfg: _emit_kv(cmd, "logdir", cfg["logdir"])

    # never pass these to experiment.py
    skip = {
        "images_dir", "json_dir", "labels_csv",
        "outdir", "logdir",
        # run-only metadata (not accepted by experiment.py):
        "name", "run_name", "tag", "id",
    }
    for k, v in cfg.items():
        if k in skip or v is None:
            continue

        # Paired booleans with explicit negative flag
        if k in PAIRED_BOOL_FLAGS:
            pos, neg = PAIRED_BOOL_FLAGS[k]
            cmd.append(pos if _truthy(v) else neg)
            continue

        # store_true flags
        if k in STORE_TRUE_FLAGS:
            if _truthy(v):
                cmd.append(f"--{k}")
            continue

        # multi-value flags
        if k in MULTI_VALUE_FLAGS:
            items = _as_listish(v)
            if items:
                cmd.append(f"--{k}")
                cmd.extend(items)
            continue

        # everything else is a single value
        _emit_kv(cmd, k, v)

    return cmd
