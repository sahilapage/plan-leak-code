"""audit3/relation_leak.py for I2-World real-car windows (src/ii_realhazard.py outputs: vehicle maps under the LOGGED and
the COLLIDE plan, each in its own future ego frame). Same agent classes, isolation filter, windowed profile alignment
and beta = delta / dx_ego. Usage (server, vla env): python audit3/ii_relation_leak.py <dir> <fold> [<dir> <fold> ...]"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_insert as pi  # noqa: E402
import realhazard as rh  # noqa: E402
from planswap import ego_box_in  # noqa: E402
from relation_leak import LW, PW, RES, accel, classify  # noqa: E402


def main(a):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-trainval", dataroot=str(rh.DATA / "nuscenes"), verbose=False)
    ctr = pi.GX[:, 0]
    rows, data = [], []
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
            ego_x = np.array([(np.linalg.inv(P0) @ P[:, 3])[0] for P in lp[3:10]])
            agents = []
            for at in nusc.get("sample", tok)["anns"]:
                ann = nusc.get("sample_annotation", at)
                if ann["category_name"].startswith(rh.VEH_CATS):
                    tr = rh.lead_track(nusc, ann, P0)
                    if tr is not None:
                        agents.append(tr)
            for k, tr in enumerate(agents):
                rel = classify(tr)
                if rel is None:
                    continue
                iso = all(not ((np.abs(o[:, 1] - tr[:, 1]) <= 2.5) & (np.abs(o[:, 0] - tr[:, 0]) <= 25)).any()
                          for j, o in enumerate(agents) if j != k)
                data.append({"scene": s["scene"], "rel": rel, "iso": iso, "a_ag": accel(tr[:, 0]),
                             "a_ego": accel(ego_x)})
                if not iso:
                    continue
                for h in (1, 2, 3):
                    Pl, Pc = lp[3 + 2 * h], cp[3 + 2 * h]
                    dx = float((np.linalg.inv(P0) @ Pc[:, 3])[0] - (np.linalg.inv(P0) @ Pl[:, 3])[0])
                    if abs(dx) < 2.0:
                        continue
                    g = P0 @ np.array([tr[2 * h, 0], tr[2 * h, 1], 0.0, 1.0])
                    prof = {}
                    for name, V, Pf in (("logged", Vp[0, 0, h], Pl), ("collide", Vp[1, 0, h], Pc)):
                        q = np.linalg.inv(Pf) @ g
                        own = ego_box_in(Pf, Pf)
                        prof[name] = ((V * ((np.abs(pi.GY - q[1]) <= 1.5) & ~own)).sum(1),
                                      int(np.argmin(np.abs(ctr - q[0]))))
                    p0, c0 = prof["logged"]
                    p1, c1 = prof["collide"]
                    if not (LW <= c0 < len(ctr) - LW):
                        continue
                    ref = p0[c0 - LW: c0 + LW + 1]
                    if ref.sum() < 3:
                        continue
                    best, arg = -1.0, None
                    for sh in range(-PW + LW, PW - LW + 1):
                        lo = c1 + sh - LW
                        if lo < 0 or lo + 2 * LW + 1 > len(ctr):
                            continue
                        seg = p1[lo: lo + 2 * LW + 1]
                        if seg.sum() < 1.0:
                            continue
                        sc = float(seg @ ref / (np.linalg.norm(seg) * np.linalg.norm(ref) + 1e-9))
                        if sc > best:
                            best, arg = sc, sh
                    if arg is not None:
                        rows.append({"scene": s["scene"], "rel": rel, "h": h, "dx": dx, "beta": arg * RES / dx})
    df, dd = pd.DataFrame(rows), pd.DataFrame(data)
    rng = np.random.default_rng(0)

    def bmed(f, B=2000):
        sc = f.scene.unique()
        g = {x: f[f.scene == x].beta.to_numpy() for x in sc}
        v = [np.median(np.concatenate([g[x] for x in rng.choice(sc, len(sc))])) for _ in range(B)]
        q = np.quantile(v, [0.025, 0.975])
        return f"{np.median(f.beta):+.2f} [{q[0]:+.2f}, {q[1]:+.2f}]"
    print("I2-World, collide vs logged; agents (all / isolated):",
          dd.groupby("rel").iso.agg(["size", "sum"]).to_dict("index"))
    for rel in ("lead", "follower", "adjacent", "oncoming", "parked"):
        for h in (1, 2, 3):
            f = df[(df.rel == rel) & (df.h == h)]
            if len(f) >= 8:
                print(f"{rel:9s} +{h} s beta {bmed(f)} n {len(f)} ({f.scene.nunique()} scenes), "
                      f"median dx_ego {f.dx.median():+.1f} m")
        e = dd[dd.rel == rel]
        if len(e) > 10:
            print(f"{'':9s} data: corr(a_agent, a_ego) {np.corrcoef(e.a_ego, e.a_ag)[0, 1]:+.2f} (n {len(e)})")


if __name__ == "__main__":
    main(sys.argv[1:])
