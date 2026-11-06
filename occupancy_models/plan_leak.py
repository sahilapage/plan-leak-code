"""Does SWTC read the real lead car's future off the ego plan (reverse causation learned from car-following logs)?
Model side: for each real-car window and plan (logged / collide / brake / still), the imagined lead position = the
obstacle-mass centroid in the lead's lane corridor (|y - lead y| <= 2 m, -12..+25 m around the lead's true position,
own ego footprint excluded), mapped to the t0 frame. beta_model(h) = slope of [imagined lead x(plan) - imagined lead
x(logged)] on [ego x(plan) - ego x(logged)], pooled over windows and the non-logged plans. A lead does not react to the
car behind it, so the causal answer is ~0; beta = 1 means the lead keeps its gap to wherever the plan puts the ego.
Data side: across the same windows, the observational coupling between the REAL lead's and the REAL (logged) ego's
accelerations over 0-3 s (least-squares x(t) = x0 + v t + a t^2 / 2): beta_data = slope of a_lead on a_ego.
Scene-cluster bootstrap B = 10,000, seed 0. Usage: python audit3/plan_leak.py <dir_f0> <dir_f1>"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import probe_insert as pi  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, PLANS, plan_pose  # noqa: E402

MIN_MASS = 3.0
T = 0.5 * np.arange(7)


def accel(xs, ts):
    A = np.stack([np.ones_like(ts), ts, 0.5 * ts ** 2], 1)
    return float(np.linalg.lstsq(A, xs, rcond=None)[0][2])


rows, data = [], []
for d in map(Path, sys.argv[1:]):
    truth = pickle.load(open(d / "truth.pkl", "rb"))
    soft = np.load(d / "soft.npy", mmap_mode="r")
    for i, tok in enumerate(json.load(open(d / "soft_tokens.json"))):
        t = truth[tok]
        tr = np.asarray(t["track"], float)
        lg = np.vstack([[0.0, 0.0], np.asarray(t["plans"]["logged"], float)])
        data.append({"scene": t["scene"], "a_lead": accel(tr[:, 0], T), "a_ego": accel(lg[:, 0], T),
                     "gap0": tr[0, 0]})
        for h in (1, 2, 3):
            lx, ly, _ = tr[2 * h]
            r = {"scene": t["scene"], "token": tok, "h": h, "lead_true": lx}
            for p_i, plan in enumerate(PLANS):
                fx, fy, fyaw = plan_pose(t["plans"][plan], h)
                m = np.asarray(soft[i, p_i, h - 1], np.float32)
                c, s_ = math.cos(-fyaw), math.sin(-fyaw)
                bx, by = c * (lx - fx) - s_ * (ly - fy), s_ * (lx - fx) + c * (ly - fy)
                corr = (np.abs(pi.GY - by) <= 2.0) & (pi.GX > bx - 12) & (pi.GX < bx + 25)
                own = (np.abs(pi.GX - EGO_C) <= EGO_L / 2 + 0.5) & (np.abs(pi.GY) <= EGO_W / 2 + 0.3)
                w = m * (corr & ~own)
                tot = float(w.sum())
                r[f"ego_{plan}"] = fx
                r[f"mass_{plan}"] = tot
                if tot >= MIN_MASS:
                    gx, gy = float((w * pi.GX).sum() / tot), float((w * pi.GY).sum() / tot)
                    r[f"lead_{plan}"] = fx + math.cos(fyaw) * gx - math.sin(fyaw) * gy
                else:
                    r[f"lead_{plan}"] = np.nan
            rows.append(r)
df, dd = pd.DataFrame(rows), pd.DataFrame(data)


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float((x * y).sum() / (x * x).sum())          # through the origin (logged plan = reference)


def boot(frame, fn, B=10000):
    scenes = frame.scene.unique()
    groups = {s: frame[frame.scene == s] for s in scenes}
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(B):
        pick = rng.choice(scenes, len(scenes))
        vals.append(fn(pd.concat([groups[s] for s in pick])))
    q = np.quantile(vals, [0.025, 0.975])
    return f"{fn(frame):+.3f} [{q[0]:+.3f}, {q[1]:+.3f}]"


print(f"{df.token.nunique()} windows, {df.scene.nunique()} scenes")
print("sanity: imagined lead (logged plan) - true lead, median m by h:",
      (df.lead_logged - df.lead_true).groupby(df.h).median().round(2).to_dict())
for h in (1, 2, 3):
    s = df[df.h == h]
    long = []
    for p in ("collide", "brake", "still"):
        ok = s[f"lead_{p}"].notna() & s.lead_logged.notna()
        long.append(pd.DataFrame({"scene": s.scene[ok], "plan": p, "dx_ego": (s[f"ego_{p}"] - s.ego_logged)[ok],
                                  "dx_lead": (s[f"lead_{p}"] - s.lead_logged)[ok]}))
    L = pd.concat(long)
    print(f"+{h} s  beta_model (all plans) {boot(L, lambda f: slope(f.dx_ego, f.dx_lead))}  n {len(L)}")
    for p in ("collide", "brake", "still"):
        Lp = L[L.plan == p]
        print(f"        {p:7s}: median dx_ego {Lp.dx_ego.median():+.2f} m, median dx_lead {Lp.dx_lead.median():+.2f} m, "
              f"ratio of medians {Lp.dx_lead.median() / Lp.dx_ego.median():+.2f}  (n {len(Lp)}, "
              f"lead mass kept {(s[f'mass_{p}'] >= MIN_MASS).mean():.2f})")


def ols(f):
    x = f.a_ego - f.a_ego.mean()
    return float((x * (f.a_lead - f.a_lead.mean())).sum() / (x * x).sum())


print(f"beta_data (real lead accel on real ego accel, observational): {boot(dd, ols)}; "
      f"corr {np.corrcoef(dd.a_ego, dd.a_lead)[0, 1]:+.2f}")
for lo, hi in ((0, 15), (15, 30), (30, 99)):
    sub = dd[(dd.gap0 >= lo) & (dd.gap0 < hi)]
    if len(sub) > 10:
        print(f"   gap {lo}-{hi} m: beta_data {ols(sub):+.2f} (n {len(sub)})")
