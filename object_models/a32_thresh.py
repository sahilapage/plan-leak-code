"""Audit of A32 belief test: sensitivity to the crash threshold (penetration >= 0.5 / 1 / 2 m), same windows rule
(lead windows, finite true limit). Usage: python audit3/a32_thresh.py <grid_a29_v2.parquet> <het_f0> <het_f1>"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRID = np.round(np.arange(0.0, 8.01, 0.25), 2)
g = pd.read_parquet(sys.argv[1]).sort_values(["token", "a"])
strat = {}
for h in sys.argv[2:]:
    strat |= {k: v["stratum"] for k, v in pickle.load(open(Path(h) / "acc" / "truth.pkl", "rb")).items()}
n = len(GRID)
toks = g.token.to_numpy()[::n]
models = ["step0", "marg", "step6", "step3", "step2", "chunk", "step1", "gtcv", "pm_chunk", "pm_step1"]


def first_safe(col, thr):
    c = (g[col] >= thr).to_numpy().reshape(-1, n)
    first = np.where(c.any(1), c.argmax(1), n)
    return np.where(first == 0, 0.0, GRID[np.maximum(first, 1) - 1])


for thr in (0.5, 1.0, 2.0):
    at = first_safe("truth", thr)
    ok = np.array([strat.get(t) != "free" for t in toks]) & (at > 0) & (at < GRID[-1])
    line = []
    for m in models:
        r = first_safe(m, thr)[ok] / at[ok]
        line.append(f"{m} {np.median(r):.2f}/{(r >= 1.5).mean():.2f}")
    print(f"threshold {thr} m ({ok.sum()} windows) median ratio / share >= 1.5x: " + " | ".join(line))
