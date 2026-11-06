"""Decision-level intervention test ("inserted hazard"). A real-shape car is inserted in the ego's OWN lane ahead, moving
straight at a constant speed below the ego's, chosen so that the ego's logged path rear-ends it at t_c = +2 s or +3 s
(front of the ego 1.5 m into the car at t_c). The world model receives the 4 past keyframes with the car and is
conditioned on the logged ego path (the path under test). The occupancy collision check -- vehicle probability >= tau
anywhere in the ego's footprint at +1 / +2 / +3 s -- must flag the path. Paired baseline: the same window without the car
(false-alarm rate of the same check). Reported: hit rate at tau = 0.2 by t_c and horizon; max vehicle p in the ego
footprint (ROC-free summary); where the car went (extra probability ahead of / at the ego).
A constant-velocity physics check on the inserted car flags every case by construction (the car moves exactly at constant
velocity), as does the permanence layer given the car as a detection: the test isolates the world model.
Usage: python src/probe_hazard.py <dome|come> <fold> <n_windows> <out.parquet> [--seed S]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_insert as pi  # noqa: E402

EGO = (1.3, 0.0, 4.1, 1.75)          # ego box centre ahead of the origin, length, width (as in the figures / verifier)
OVERLAP = 1.5                        # the ego's front is 1.5 m into the car at t_c
GAP_MIN = 8.0                        # the car's rear is at least 8 m ahead of the ego's front at t0


def ego_mask():
    cx, cy, L, W = EGO
    return (np.abs(pi.GX - cx) <= L / 2) & (np.abs(pi.GY - cy) <= W / 2)


def along(P0, P):
    """distance travelled from pose P0 to pose P along P0's heading."""
    yaw0 = pi.yaw_of(P0)
    d = P[:2, 3] - P0[:2, 3]
    return float(d[0] * math.cos(yaw0) + d[1] * math.sin(yaw0)), float(-d[0] * math.sin(yaw0) + d[1] * math.cos(yaw0))


def plan(ps, tc, L_car):
    """speed and start offset of the lead car so that the ego front is OVERLAP m into it at tc; None if infeasible."""
    se = {t: along(ps[3], P)[0] for t, P in zip((1.0, 2.0, 3.0), (ps[5], ps[7], ps[9]))}
    ego_front0 = EGO[0] + EGO[2] / 2
    v_ego = se[3.0] / 3.0
    if v_ego < 4.0:
        return None
    for gap in (GAP_MIN, 12.0, 16.0):
        rear0 = ego_front0 + gap                                     # car rear at t0, along the ego heading
        v = (se[tc] + ego_front0 - OVERLAP - rear0) / tc            # rear(tc) = ego front(tc) - OVERLAP
        if 0.0 <= v <= v_ego - 1.0:
            return v, rear0 + L_car / 2                              # (speed, centre ahead of the ego origin at t0)
    return None


def car_xy(P0, xa, v, t):
    yaw0 = pi.yaw_of(P0)
    e = np.array([math.cos(yaw0), math.sin(yaw0)])
    return P0[:2, 3] + (xa + v * t) * e, yaw0


def select(fold, n, seed, bank):
    infos, wins = di.windows({fold})
    rng = np.random.default_rng(seed)
    out = []
    for j in rng.permutation(len(wins)):
        scene, idx = wins[j]
        fr = infos[scene][idx: idx + 10]
        ps = [di.pose_of(f) for f in fr]
        if abs((pi.yaw_of(ps[9]) - pi.yaw_of(ps[0]) + math.pi) % (2 * math.pi) - math.pi) > math.radians(8):
            continue
        if max(abs(along(ps[3], ps[k])[1]) for k in (5, 7, 9)) > 1.0:          # the ego stays in its lane
            continue
        patch = bank[int(fr[3]["token"][:8], 16) % len(bank)]
        keep = [0, 1, 2, 3, 5, 7, 9]
        occs = [di.load_occ(scene, fr[k]["token"]) for k in keep]
        kp = [ps[k] for k in keep]
        for tc in (2.0, 3.0):
            pl = plan(ps, tc, patch["dims"][0])
            if pl is None:
                continue
            v, xa = pl
            ok = True
            for k, t in enumerate(pi.FT):
                xy, yaw = car_xy(ps[3], xa, v, t)
                m, c = pi.footprint(kp[k], xy, yaw, dims=patch["dims"], grow=0.4)
                if not m.any() or abs(c[0]) > 36 or np.isin(occs[k][m], pi.OBST).any() or \
                        (occs[k][m] == pi.DRIVABLE).any(1).mean() < 0.8:
                    ok = False
                    break
            if ok:
                out.append({"scene": scene, "idx": idx, "token": fr[3]["token"], "tc": tc, "v": v, "xa": xa,
                            "v_ego": along(ps[3], ps[9])[0] / 3.0})
                break
        if len(out) == n:
            break
    return infos, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("fold", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    import pandas as pd
    bank = pi.build_bank(0, 60, exclude=set())
    infos, sel = select(a.fold, a.n, a.seed, bank)
    print(f"{len(sel)} valid windows (t_c = 2 s: {sum(s['tc'] == 2.0 for s in sel)})", flush=True)
    net = {"dome": pi.Dome, "come": pi.Come}[a.model]()
    em = ego_mask()
    rows = []
    for b0 in range(0, len(sel), a.bs):
        batch = sel[b0: b0 + a.bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s in batch]
        poses = [[di.pose_of(f) for f in fr] for fr in frs]
        base = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for s, fr in zip(batch, frs)]
        patches = [bank[int(s["token"][:8], 16) % len(bank)] for s in batch]
        ins = [np.stack([pi.insert(bp[k], ps[k], *car_xy(ps[3], s["xa"], s["v"], pi.FT[k]), patch=pa) for k in range(4)])
               for s, bp, ps, pa in zip(batch, base, poses, patches)]
        Pb = net(base, poses, seed=1000 * a.seed + b0)
        Pi = net(ins, poses, seed=1000 * a.seed + b0)
        for s, ps, pa, pb, pin in zip(batch, poses, patches, Pb, Pi):
            for h in range(1, 4):
                fp = [ps[3], ps[5], ps[7], ps[9]][h]
                xy, yaw = car_xy(ps[3], s["xa"], s["v"], float(h))
                mc, c = pi.footprint(fp, xy, yaw, dims=pa["dims"])
                d = np.clip(pin[h] - pb[h], 0, None)
                rows.append({"token": s["token"], "scene": s["scene"], "tc": s["tc"], "v_car": s["v"], "v_ego": s["v_ego"],
                             "h": h, "geom_overlap": float((mc & em).sum()),
                             "ego_max_ins": float(pin[h][em].max()), "ego_max_base": float(pb[h][em].max()),
                             "car_q50": pi.q50(pin[h], mc), "car_mass": float(d[np.hypot(pi.GX - c[0], pi.GY - c[1]) <= 6].sum()),
                             "ahead_mass": float(d[(pi.GX > c[0] + 3) & (np.abs(pi.GY - c[1]) <= 3)].sum())})
            rows.append({"token": s["token"], "scene": s["scene"], "tc": s["tc"], "h": 0,
                         "car_q50": pi.q50(pin[0], pi.footprint(ps[3], *car_xy(ps[3], s["xa"], s["v"], 0.0), dims=pa["dims"])[0])})
        print(f"{b0 + len(batch)}/{len(sel)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    d = df[df.h > 0].assign(hit=lambda x: x.ego_max_ins >= 0.2, fa=lambda x: x.ego_max_base >= 0.2)
    print("perceived at t0:", round(float((df[df.h == 0].car_q50 >= 0.2).mean()), 3))
    print(d.groupby(["tc", "h"]).agg(n=("hit", "size"), geom=("geom_overlap", lambda x: (x > 0).mean()), hit=("hit", "mean"),
                                    false_alarm=("fa", "mean"), car_kept=("car_q50", lambda x: (x >= 0.2).mean()),
                                    ego_p=("ego_max_ins", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
