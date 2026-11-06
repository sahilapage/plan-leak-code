"""Kill the "faster = more dangerous" confound of A27: in A27 the crash plan (collide) is always the faster one, so a
judge that only penalises forward travel would also separate collide from logged. Here every window gets the SAME extra
acceleration a in {1, 2, 3} m/s^2 (logged path + 0.5 a t^2 along x, as collide); whether that plan hits the real lead
depends on the gap and speeds only. Labels (kinematic, non-reactive lead = its logged track): positive if the ego's front
overlaps the lead's true box by >= 1 m longitudinally (laterally overlapping) at any of +1 / +2 / +3 s; negative if the
gap stays >= 2 m at all three; else excluded. Scores (no new forecasts): cvref (the past-only CV forecast, A26) and
oracle (the logged forecast): max obstacle probability in the candidate's ego box placed in that forecast, over +1 / +2
/ +3 s. Within each a, a speed heuristic is constant (AUROC 0.5 by construction). Reports AUROC pooled over a and per a,
scene bootstrap. Usage: python audit3/fixed_acc.py <swrh_dir> <swcv_dir> [<swrh_dir> <swcv_dir> ...]"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roc import auroc  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, PLANS, box_in_frame, plan_pose  # noqa: E402

ACCS = tuple(np.arange(1.0, 3.51, 0.25).round(2))
FRONT = EGO_C + EGO_L / 2


def acc_plan(logged, a):
    return [[x + 0.5 * a * (0.5 * (k + 1)) ** 2, y] for k, (x, y) in enumerate(logged)]


def label(t, plan):
    L, W = t["LW"]
    ov = []
    for h in (1, 2, 3):
        x, y, _ = plan_pose(plan, h)
        lx, ly, _ = t["track"][2 * h]
        lat = abs(ly - y) <= (EGO_W + W) / 2
        ov.append((x + FRONT - (lx - L / 2)) if lat else -99.0)
    if max(ov) >= 1.0:
        return 1
    if max(ov) <= -2.0:
        return 0
    return -1


def rows_for(src, cvd):
    src, cvd = Path(src), Path(cvd)
    T = pickle.load(open(cvd / "truth.pkl", "rb"))
    cvs = np.load(cvd / "soft.npy", mmap_mode="r")
    own = np.load(src / "soft.npy", mmap_mode="r")
    oi = {t: i for i, t in enumerate(json.load(open(src / "soft_tokens.json")))}
    out = []
    for i, tok in enumerate(json.load(open(cvd / "soft_tokens.json"))):
        t = T[tok]
        for a in ACCS:
            plan = acc_plan(t["plans"]["logged"], a)
            lab = label(t, plan)
            if lab < 0:
                continue
            r = {"scene": t["scene"], "token": tok, "a": a, "y": lab}
            for judge in ("cvref", "oracle"):
                best = 0.0
                for h in (1, 2, 3):
                    if judge == "cvref":
                        m, (fx, fy, fyaw) = np.asarray(cvs[i, 0, h - 1], np.float32), plan_pose(t["cv"], h)
                    else:
                        m = np.asarray(own[oi[tok], PLANS.index("logged"), h - 1], np.float32)
                        fx, fy, fyaw = plan_pose(t["plans"]["logged"], h)
                    cx, cy, cyaw = plan_pose(plan, h)
                    ob = box_in_frame(fx, fy, fyaw, cx + EGO_C * math.cos(cyaw), cy + EGO_C * math.sin(cyaw), cyaw,
                                      EGO_L, EGO_W)
                    if ob.any():
                        best = max(best, float(m[ob].max()))
                r[judge] = best
            out.append(r)
    return out


def au(f, j):
    return auroc(f[f.y == 1][j], f[f.y == 0][j]) if (f.y == 1).any() and (f.y == 0).any() else np.nan


def strat(f, j):
    """speed-blind AUROC: only crash / no-crash pairs with the SAME extra acceleration, pair-weighted over a."""
    num = den = 0.0
    for _, g in f.groupby("a"):
        p, n = g[g.y == 1][j].to_numpy(), g[g.y == 0][j].to_numpy()
        if len(p) and len(n):
            num += ((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum())
            den += len(p) * len(n)
    return num / den if den else np.nan


def main(a):
    df = pd.DataFrame(sum((rows_for(a[i], a[i + 1]) for i in range(0, len(a), 2)), []))
    sc = df.scene.unique()
    g = {s: df[df.scene == s] for s in sc}
    rng = np.random.default_rng(0)
    res = [pd.concat([g[s] for s in rng.choice(sc, len(sc))]) for _ in range(2000)]
    print(f"{df.token.nunique()} windows, {len(sc)} scenes; candidates by a and label:")
    print(df.groupby(["a", "y"]).size().unstack().fillna(0).astype(int).T.to_string())
    for j in [c for c in ("self", "cvref", "oracle") if c in df]:
        reps = [strat(f, j) for f in res]
        rp = [au(f, j) for f in res]
        print(f"  {j:6s} SPEED-BLIND AUROC (same-a pairs) {strat(df, j):.3f} [{np.nanquantile(reps, 0.025):.3f}, "
              f"{np.nanquantile(reps, 0.975):.3f}] | pooled (speed-confounded) {au(df, j):.3f} "
              f"[{np.nanquantile(rp, 0.025):.3f}, {np.nanquantile(rp, 0.975):.3f}]")
    df.to_parquet(Path(a[0]).parent / f"fixed_acc_{Path(a[0]).name}.parquet")


if __name__ == "__main__":
    main(sys.argv[1:])
