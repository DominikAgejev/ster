# src/diag/normalize.py
from __future__ import annotations
from typing import Iterable, Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd

class NormTools:
    def __init__(self, state, log):
        self.s, self.log = state, log

    def meta_columns(self, used_cols: Sequence[str], img_mean_cols: Sequence[str], meta_only_cols: Sequence[str]):
        if not self.s.enabled: return
        self.log.write("meta_cols",
                       used=len(used_cols), used_cols=list(used_cols),
                       img_mean_cols=list(img_mean_cols),
                       meta_only_cols=list(meta_only_cols))

    def check_standardization(self,
                              df: pd.DataFrame,
                              cont_cols: Sequence[str],
                              train_rows: Sequence[int],
                              mean_atol: float = 5e-3,
                              std_atol: float  = 5e-2,
                              hard_assert: bool = True) -> Dict[str, Any]:
        """
        Validate that TRAIN standardization produced ~0 mean and ~1 std on continuous columns.
        Returns a dict summary and optionally raises.
        """
        if not self.s.enabled or not cont_cols:
            return {}

        post_tr = df.loc[train_rows, list(cont_cols)].astype(float)
        m = post_tr.mean(0)
        s = post_tr.std(0)

        bad_mean = (m.abs() > mean_atol)
        bad_std  = ((s - 1.0).abs() > std_atol)

        bad_cols = []
        for c in cont_cols:
            flags = []
            mv = float(m.get(c, np.nan))
            sv = float(s.get(c, np.nan))
            if not np.isfinite(mv) or not np.isfinite(sv):
                flags.append("nan")
            else:
                if abs(mv) > mean_atol: flags.append(f"mean={mv:.4f}")
                if abs(sv - 1.0) > std_atol: flags.append(f"std={sv:.4f}")
            if flags:
                bad_cols.append((c, flags))

        summary = {
            "checked": list(cont_cols),
            "violations": [{"col": c, "flags": fl} for c, fl in bad_cols],
            "mean_atol": mean_atol,
            "std_atol": std_atol,
        }
        if bad_cols:
            self.log.write("norm_check_fail", **summary)
            if hard_assert:
                raise AssertionError("[norm] standardization off: " + ", ".join(f"{c}({';'.join(fl)})" for c, fl in bad_cols))
        else:
            self.log.write("norm_check_ok", checked=len(cont_cols))
        return summary

    def check_binary_flags(self, df: pd.DataFrame, cols: Iterable[str], only_if_used: Optional[Iterable[str]] = None):
        if not self.s.enabled: return
        cols = [c for c in cols if c in df.columns]
        if only_if_used:
            used = set(only_if_used)
            cols = [c for c in cols if c in used]
        issues = []
        for c in cols:
            vals = set(np.unique(df[c].to_numpy()))
            if not vals.issubset({0.0, 1.0, 0, 1}):
                issues.append((c, sorted(map(float, vals))))
        if issues:
            self.log.write("binary_flag_bad", issues=[{"col": c, "values": v} for c, v in issues])
        else:
            self.log.write("binary_flag_ok", checked=cols)

    def record_mu_sigma(self, mu: pd.Series, sigma: pd.Series):
        if not self.s.enabled: return
        self.log.write("norm_params",
                       mu={k: float(v) for k, v in mu.items()},
                       sigma={k: float(v) for k, v in sigma.items()})
