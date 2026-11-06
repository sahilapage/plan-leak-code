"""Does a plan-conditioned world model clear the path it is asked to evaluate? (2 x 2, same scene, same seed)
For each window (straight logged ego path, t0 speed >= 4 m/s), a real-shape PARKED car is inserted
  inpath   in the ego's lane, placed so the logged path drives 1.5 m into it at t_c = 2 s
  adjacent the same car one lane over (3.6 m, the side whose footprint is free), same distance ahead
and the model is conditioned on two plans for the next 3 s (the 4 past frames are always the logged ones):
  logged   the recorded path (drives into the in-path car)
  stop     the same heading, braking at constant deceleration from the t0 speed to stop with the front 3 m short of the
           car's rear (deceleration <= 6 m/s^2)
Kept at h = car footprint >= 50 % covered at p >= 0.2 (q50 >= 0.2), the footprint taken in the frame the plan puts the ego
in at h. Also: q50 of the part of the footprint outside the ego box, and a no-car run under each plan (base_q50).
A parked car needs no carrying; a plan-blind model (COME: it forecasts in the t0 frame and is warped to the plan's
frames afterwards) is the negative control.
Usage: python src/plan_cond.py <dome|come> <fold> <n_windows> <out.parquet> [--seed S]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_hazard as ph  # noqa: E402
import probe_insert as pi  # noqa: E402

TC = 2.0
STOP_MARGIN = 3.0
A_MAX = 6.0
FUT = [0.5 * k for k in range(1, 7)]                      # times of poses 4..9


def stop_poses(ps, rear0, v0):
    """poses 4..9 of the stop plan: heading of t0, constant deceleration to stop STOP_MARGIN m short of the car."""
    s_stop = rear0 - (ph.EGO[0] + ph.EGO[2] / 2) - STOP_MARGIN
    a = v0 ** 2 / (2 * s_stop)
    yaw0 = pi.yaw_of(ps[3])
    e = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    out = []
    for t in FUT:
        s = v0 * t - 0.5 * a * t ** 2 if t < v0 / a else s_stop
        P = ps[3].copy()
        P[:3, 3] = ps[3][:3, 3] + s * e
        out.append(P)
    return out, a


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
        if max(abs(ph.along(ps[3], ps[k])[1]) for k in (5, 7, 9)) > 1.0:
            continue
        v0 = ph.along(ps[2], ps[3])[0] / 0.5
        if v0 < 4.0:
            continue
        patch = bank[int(fr[3]["token"][:8], 16) % len(bank)]
        se2 = ph.along(ps[3], ps[7])[0]
        front0 = ph.EGO[0] + ph.EGO[2] / 2
        rear0 = se2 + front0 - ph.OVERLAP
        if rear0 - front0 - STOP_MARGIN < 1.0:
            continue
        cf, a = stop_poses(ps, rear0, v0)
        if a > A_MAX:
            continue
        xa = rear0 + patch["dims"][0] / 2
        keep = [0, 1, 2, 3, 5, 7, 9]
        occs = [di.load_occ(scene, fr[k]["token"]) for k in keep]
        kp = [ps[k] for k in keep]

        def valid(side):
            yaw0 = pi.yaw_of(ps[3])
            xy = ps[3][:2, 3] + xa * np.array([math.cos(yaw0), math.sin(yaw0)]) + \
                side * pi.LANE * np.array([-math.sin(yaw0), math.cos(yaw0)])
            for k in range(7):
                m, c = pi.footprint(kp[k], xy, yaw0, dims=patch["dims"], grow=0.4)
                if not m.any() or abs(c[0]) > 36 or np.isin(occs[k][m], pi.OBST).any() or \
                        (occs[k][m] == pi.DRIVABLE).any(1).mean() < 0.8:
                    return None
            return xy
        xy_in = valid(0)
        if xy_in is None:
            continue
        side = 1 if int(fr[3]["token"][8:16], 16) % 2 else -1
        xy_adj = valid(side)
        if xy_adj is None:
            side = -side
            xy_adj = valid(side)
        if xy_adj is None:
            continue
        out.append({"scene": scene, "idx": idx, "token": fr[3]["token"], "xy": {"inpath": xy_in, "adjacent": xy_adj},
                    "yaw": pi.yaw_of(ps[3]), "v0": v0, "decel": a, "side": side, "cf": cf})
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
    print(f"{len(sel)} windows; decel median {np.median([s['decel'] for s in sel]):.2f} m/s^2", flush=True)
    net = {"dome": pi.Dome, "come": pi.Come}[a.model]()
    em = ph.ego_mask()
    rows = []
    for b0 in range(0, len(sel), a.bs):
        batch = sel[b0: b0 + a.bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s in batch]
        logged = [[di.pose_of(f) for f in fr] for fr in frs]
        base = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for s, fr in zip(batch, frs)]
        patches = [bank[int(s["token"][:8], 16) % len(bank)] for s in batch]
        for plan in ("logged", "stop"):
            poses = logged if plan == "logged" else [lp[:4] + s["cf"] for lp, s in zip(logged, batch)]
            seed = 1000 * a.seed + b0
            Pb = net(base, poses, seed=seed)
            for pos in ("inpath", "adjacent"):
                ins = [np.stack([pi.insert(bp[k], ps[k], s["xy"][pos], s["yaw"], patch=pa) for k in range(4)])
                       for s, bp, ps, pa in zip(batch, base, poses, patches)]
                Pi = net(ins, poses, seed=seed)
                for s, ps, pa, pb, pin in zip(batch, poses, patches, Pb, Pi):
                    for h in range(4):
                        P = ps[3 + 2 * h] if h else ps[3]
                        m, _ = pi.footprint(P, s["xy"][pos], s["yaw"], dims=pa["dims"])
                        mo = m & ~em
                        rows.append({"token": s["token"], "scene": s["scene"], "plan": plan, "pos": pos, "h": h,
                                     "v0": s["v0"], "decel": s["decel"], "q50": pi.q50(pin[h], m),
                                     "q50_outside_ego": pi.q50(pin[h], mo) if mo.any() else np.nan,
                                     "overlap_ego": float((m & em).sum()), "base_q50": pi.q50(pb[h], m),
                                     "ego_max_ins": float(pin[h][em].max()), "ego_max_base": float(pb[h][em].max())})
        print(f"{b0 + len(batch)}/{len(sel)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    print(df.assign(kept=df.q50 >= 0.2).groupby(["pos", "plan", "h"]).agg(
        kept=("kept", "mean"), q50=("q50", "median"), out_ego=("q50_outside_ego", "median"),
        base=("base_q50", "median")).round(3).unstack("h").to_string())


if __name__ == "__main__":
    main()
