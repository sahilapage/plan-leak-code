"""Control for audit3/plan_leak.py: is beta ~ 1 lead-specific (the plan moves the lead) or does SWTC ignore the plan's
displacement for everything (the whole map copied in the ego frame)? For each window / non-logged plan / h with
|dx_ego| >= 3 m, the longitudinal shift s that best aligns the plan's obstacle profile (summed over y) with the logged
plan's (profile_p(x) ~ profile_logged(x + s)) in two regions: the lead's lane corridor (|y - lead y| <= 2 m) and the
background (|y - lead y| > 4 m, |y| <= 20 m). beta = 1 - s / dx_ego: 0 = world-consistent (things stay put), 1 = kept at
the same distance from the ego. Scene-cluster bootstrap of the median. Usage: python audit3/plan_leak_bg.py <d0> <d1>"""
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

SH = np.arange(-75, 76)                                   # shifts in cells (0.4 m)
rows = []
for d in map(Path, sys.argv[1:]):
    truth = pickle.load(open(d / "truth.pkl", "rb"))
    soft = np.load(d / "soft.npy", mmap_mode="r")
    for i, tok in enumerate(json.load(open(d / "soft_tokens.json"))):
        t = truth[tok]
        for h in (1, 2, 3):
            lx, ly, _ = t["track"][2 * h]
            prof = {}
            for p_i, plan in enumerate(PLANS):
                fx, fy, fyaw = plan_pose(t["plans"][plan], h)
                m = np.asarray(soft[i, p_i, h - 1], np.float32)
                by = math.sin(-fyaw) * (lx - fx) + math.cos(-fyaw) * (ly - fy)
                own = (np.abs(pi.GX - EGO_C) <= EGO_L / 2 + 0.5) & (np.abs(pi.GY) <= EGO_W / 2 + 0.3)
                lane = (np.abs(pi.GY - by) <= 2.0) & ~own
                bg = (np.abs(pi.GY - by) > 4.0) & (np.abs(pi.GY) <= 20.0)
                prof[plan] = (fx, (m * lane).sum(1), (m * bg).sum(1))
            x0 = prof["logged"][0]
            for plan in ("collide", "brake", "still"):
                dx = prof[plan][0] - x0
                if abs(dx) < 3.0:
                    continue
                r = {"scene": t["scene"], "h": h, "plan": plan, "dx": dx}
                for k, name in ((1, "lane"), (2, "bg")):
                    a, b = prof[plan][k], prof["logged"][k]
                    if a.sum() < 3 or b.sum() < 3:
                        r[name] = np.nan
                        continue
                    sc = []
                    for s in SH:                                  # profile_p(x) vs profile_logged(x + s)
                        if s >= 0:
                            u, v = a[: 200 - s], b[s:]
                        else:
                            u, v = a[-s:], b[: 200 + s]
                        sc.append(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
                    r[name] = 1.0 - (SH[int(np.argmax(sc))] * 0.4) / dx
                rows.append(r)
df = pd.DataFrame(rows)


def boot_med(f, col, B=5000):
    f = f[f[col].notna()]
    scenes = f.scene.unique()
    g = {s: f[f.scene == s][col].to_numpy() for s in scenes}
    rng = np.random.default_rng(0)
    v = [np.median(np.concatenate([g[s] for s in rng.choice(scenes, len(scenes))])) for _ in range(B)]
    q = np.quantile(v, [0.025, 0.975])
    return f"{f[col].median():+.2f} [{q[0]:+.2f}, {q[1]:+.2f}] (n {len(f)})"


print(f"{len(df)} plan-window-horizon rows with |dx_ego| >= 3 m")
for h in (1, 2, 3):
    s = df[df.h == h]
    print(f"+{h} s  beta lead lane {boot_med(s, 'lane')} | beta background {boot_med(s, 'bg')}")
    for p in ("collide", "brake", "still"):
        sp = s[s.plan == p]
        if len(sp):
            print(f"      {p:7s} lane {sp.lane.median():+.2f}  background {sp.bg.median():+.2f}  (n {len(sp)})")
