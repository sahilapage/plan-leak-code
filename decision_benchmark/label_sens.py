"""Audit: are the speed-blind conclusions an artefact of the label thresholds (crash = front overlap >= 1 m, safe = gap
>= 2 m; audit3/fixed_acc.py)? Rebuilds the full candidate table (every window x every a, no label filter), keeps the
continuous overlap depth, relabels under a grid of (crash, safe) thresholds and reports each judge's speed-blind AUROC
and the planner's crash rate at mean extra acceleration >= 1.5 m/s^2 (audit3/planner_select.py).
Usage (server, vla env): python audit3/label_sens.py <swrh_dir> <swcv_dir> <fold> self=<acc_dir> ... [physics]"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_acc as fa  # noqa: E402
from fixed_acc_extra import physics_scores, self_scores  # noqa: E402
from planner_select import at_progress, curve  # noqa: E402
from sw_realhazard import EGO_W, plan_pose  # noqa: E402

GRID = ((1.0, 2.0), (0.5, 1.0), (0.5, 2.0), (2.0, 2.0), (1.0, 4.0), (2.0, 4.0), (0.0, 0.5))


def depth(t, plan):
    L, W = t["LW"]
    ov = -99.0
    for h in (1, 2, 3):
        x, y, _ = plan_pose(plan, h)
        lx, ly, _ = t["track"][2 * h]
        if abs(ly - y) <= (EGO_W + W) / 2:
            ov = max(ov, x + fa.FRONT - (lx - L / 2))
    return ov


def main(args):
    src, cvd, fold = args[0], args[1], int(args[2])
    fa.label = lambda t, plan: 1                              # keep every candidate; relabelled below
    df = pd.DataFrame(fa.rows_for(src, cvd)).drop(columns="y")
    T = pickle.load(open(Path(cvd) / "truth.pkl", "rb"))
    df["ov"] = [depth(T[t], fa.acc_plan(T[t]["plans"]["logged"], a)) for t, a in zip(df.token, df.a)]
    for extra in args[3:]:
        if extra.startswith("self="):
            d = extra.split("=", 1)[1]
            sc = self_scores(d)
            df[f"self_{Path(d).name}"] = [sc.get((t, a), np.nan) for t, a in zip(df.token, df.a)]
        elif extra == "physics":
            sc = physics_scores(src, fold)
            df["physics"] = [sc.get((t, a), np.nan) for t, a in zip(df.token, df.a)]
    df["h_gap0"] = [-T[t]["track"][0][0] for t in df.token]
    df["h_v_ego"] = [T[t]["plans"]["logged"][0][0] / 0.5 for t in df.token]
    df.to_parquet(Path(src).parent / f"labelsens_{Path(src).name}.parquet")
    judges = [c for c in df.columns if c in ("cvref", "oracle", "physics", "h_gap0", "h_v_ego") or c.startswith("self_")]
    print(f"{df.token.nunique()} windows, {len(df)} candidates (unfiltered)")
    for cr, sf in GRID:
        f = df.copy()
        f["y"] = np.where(f.ov >= cr, 1, np.where(f.ov <= -sf, 0, -1))
        f = f[f.y >= 0].dropna(subset=judges)
        sb = " | ".join(f"{j} {fa.strat(f, j):.3f}" for j in judges)
        pc = " | ".join(f"{j} {at_progress(curve(f, j), 1.5):.3f}" for j in judges)
        print(f"crash >= {cr} m, safe gap >= {sf} m: n {len(f)} ({int(f.y.sum())} crash)\n  speed-blind: {sb}\n"
              f"  planner crash @ +1.5: {pc}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
