"""audit3/plan_leak_bg.py for I2-World real cars (src/ii_realhazard.py outputs: veh maps under the LOGGED and the
COLLIDE plan, each in its own future ego frame). For each window and h, the longitudinal shift that best aligns the
collide forecast's vehicle profile with the logged forecast's, in the lead's lane corridor (|y - lead y| <= 2 m, own
ego box excluded) and in the background (|y - lead y| > 4 m, |y| <= 20 m). beta = 1 - s / dx_ego (0 = stays put in the
world, 1 = keeps its distance to the ego). Usage: python audit3/ii_plan_leak.py <dir> <fold> [<dir> <fold> ...]"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dome_infer as di  # noqa: E402
import probe_insert as pi  # noqa: E402
import realhazard as rh  # noqa: E402
from planswap import ego_box_in  # noqa: E402

SH = np.arange(-75, 76)


def best_beta(a, b, dx):
    if a.sum() < 3 or b.sum() < 3:
        return np.nan
    sc = []
    for s in SH:
        u, v = (a[: 200 - s], b[s:]) if s >= 0 else (a[-s:], b[: 200 + s])
        sc.append(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
    return 1.0 - SH[int(np.argmax(sc))] * 0.4 / dx


rows = []
a = sys.argv[1:]
for d, fold in [(Path(a[i]), int(a[i + 1])) for i in range(0, len(a), 2)]:
    W = pickle.load(open(d / "windows.pkl", "rb"))
    infos, _ = di.windows({fold})
    for tok, s in W.items():
        fp = d / "plan" / f"{tok}_maps.npz"
        if not fp.exists():
            continue
        Vp = np.load(fp)["veh"].astype(np.float32)
        lp = [di.pose_of(f) for f in infos[s["scene"]][s["idx"]: s["idx"] + 10]]
        P0 = lp[3]
        cp = lp[:4] + [rh.shift_pose(lp[k], P0, 0.5 * s["a"] * (0.5 * (k - 3)) ** 2) for k in range(4, 10)]
        for h in (1, 2, 3):
            Pl, Pc = lp[3 + 2 * h], cp[3 + 2 * h]
            dx = float((np.linalg.inv(P0) @ Pc[:, 3])[0] - (np.linalg.inv(P0) @ Pl[:, 3])[0])
            if abs(dx) < 1.0:
                continue
            x, y, _ = s["lead_track"][2 * h]
            g = P0 @ np.array([x, y, 0.0, 1.0])
            prof = {}
            for name, V, Pf in (("logged", Vp[0, 0, h], Pl), ("collide", Vp[1, 0, h], Pc)):
                by = (np.linalg.inv(Pf) @ g)[1]
                own = ego_box_in(Pf, Pf)
                lane = (np.abs(pi.GY - by) <= 2.0) & ~own
                bg = (np.abs(pi.GY - by) > 4.0) & (np.abs(pi.GY) <= 20.0)
                prof[name] = ((V * lane).sum(1), (V * bg).sum(1))
            rows.append({"scene": s["scene"], "h": h, "dx": dx,
                         "lane": best_beta(prof["collide"][0], prof["logged"][0], dx),
                         "bg": best_beta(prof["collide"][1], prof["logged"][1], dx)})
df = pd.DataFrame(rows)


def boot_med(f, col, B=5000):
    f = f[f[col].notna()]
    scenes = f.scene.unique()
    g = {sc: f[f.scene == sc][col].to_numpy() for sc in scenes}
    rng = np.random.default_rng(0)
    v = [np.median(np.concatenate([g[sc] for sc in rng.choice(scenes, len(scenes))])) for _ in range(B)]
    q = np.quantile(v, [0.025, 0.975])
    return f"{f[col].median():+.2f} [{q[0]:+.2f}, {q[1]:+.2f}] (n {len(f)})"


print(f"I2-World real cars: {df.scene.nunique()} scenes; collide vs logged plan")
for h in (1, 2, 3):
    s = df[df.h == h]
    print(f"+{h} s  median dx_ego {s.dx.median():+.2f} m | beta lead lane {boot_med(s, 'lane')} | "
          f"beta background {boot_med(s, 'bg')}")
