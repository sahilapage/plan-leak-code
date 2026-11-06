"""Audit of A32: are the step1-trained policy's real crashes caused by the plan leak (the lead pushed ahead by the
plan) or by generic forecast error? For each eval window, at the policy's chosen extra accel a (lambda 64, policy
seed 0, from policy_eval.parquet): step1's predicted overlap vs the plan-free model's (marg) and GT-CV's at the same
a. Leak-attributable = real crash, step1 says safe (< 1 m), the plan-free model says crash (>= 1 m). Effective
beta at the chosen plan = (depth_marg - depth_step1) / extra ego displacement at 3 s (0.5 a 9).
Usage: python audit3/a32_mech.py <objwm_dir>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O = Path(sys.argv[1])
g = pd.read_parquet(O / "grid_a29_v2.parquet").sort_values(["token", "a"])
toks = g.token.unique()
pe = pd.read_parquet(O / "policy_eval.parquet")
for lam in sorted(pe.lam.unique()):
    for m in ("step1", "chunk", "step2", "step0"):
        p = pe[(pe.lam == lam) & (pe.model == m)].reset_index(drop=True)
        if not len(p):
            continue
        p["token"] = toks
        x = p.merge(g, on=["token", "a"], how="left")
        cr = x[x.crash == 1]
        if not len(cr):
            continue
        safe_m = cr[m] < 1.0
        leak = safe_m & (cr.marg >= 1.0)
        cv = safe_m & (cr.gtcv >= 1.0)
        ext = 0.5 * cr.a * 9.0
        beta = ((cr.marg - cr[m]) / ext.where(ext > 0)).median()
        print(f"lambda {lam:5.0f} {m:6s}: real crashes {len(cr):3d} | WM said safe {safe_m.mean():.2f} | of those, "
              f"plan-free model flags crash {(leak.sum() / max(safe_m.sum(), 1)):.2f}, GT-CV flags {(cv.sum() / max(safe_m.sum(), 1)):.2f}"
              f" | effective beta at chosen plan (median) {beta:+.2f} | mean chosen a {x.a.mean():.2f}")
