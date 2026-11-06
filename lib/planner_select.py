"""Decision payoff of each judge: a minimal planner on real-car windows. Candidates per window = the logged path plus a
fixed extra acceleration a in ACCS (audit3/fixed_acc.py; kinematic crash truth vs the lead's logged track). The
planner picks the MOST AGGRESSIVE candidate (largest a; progress) whose judge score is below a threshold t; if none,
it keeps the logged path (a = 0, never a crash). Sweeping t traces each judge's crash-rate vs progress curve.
Reported: crash rate at matched progress (mean extra acceleration chosen) and progress at matched crash rate (5 %),
for every judge in the speed-blind table (original self, RPT self, cvref, oracle, physics, heuristics), plus the
privileged truth planner (picks the largest truly safe a). Ambiguous candidates (0 < overlap < ... ) count as crash
only if overlap >= 1 m (as labelled). Usage: python audit3/planner_select.py <speedblind_parquet>"""
import sys

import numpy as np
import pandas as pd


def curve(df, j, higher_is_danger=True):
    """for thresholds over the judge's score quantiles: (mean chosen a, crash rate)."""
    s = df[j] if higher_is_danger else -df[j]
    out = []
    for t in np.unique(np.quantile(s.dropna(), np.linspace(0, 1, 101))):
        ok = df[s < t]
        pick = ok.groupby("token").a.max().reindex(df.token.unique()).fillna(0.0)
        lab = df.set_index(["token", "a"]).y
        crash = np.array([lab.get((tk, a), 0) if a > 0 else 0 for tk, a in pick.items()])
        out.append((float(pick.mean()), float(crash.mean())))
    return np.array(sorted(set(out)))


def at_progress(c, p):
    ok = c[c[:, 0] >= p]
    return float(ok[:, 1].min()) if len(ok) else np.nan


def at_crash(c, r):
    ok = c[c[:, 1] <= r]
    return float(ok[:, 0].max()) if len(ok) else 0.0


def main(path):
    df = pd.read_parquet(path)
    df = df[df.y >= 0]
    judges = [c for c in df.columns if c in ("cvref", "oracle", "physics", "h_gap0", "h_v_ego") or c.startswith("self_")]
    truth = df[df.y == 0].groupby("token").a.max().reindex(df.token.unique()).fillna(0.0)
    print(f"{df.token.nunique()} windows; truth planner (largest truly safe a): mean a {truth.mean():.2f}, crash 0")
    for p in (1.0, 1.5, 2.0):
        line = " | ".join(f"{j} {at_progress(curve(df, j), p):.3f}" for j in judges)
        print(f"crash rate at mean extra accel >= {p}: {line}")
    line = " | ".join(f"{j} {at_crash(curve(df, j), 0.05):.2f}" for j in judges)
    print(f"mean extra accel at crash rate <= 5 %: {line}")


if __name__ == "__main__":
    main(sys.argv[1])
