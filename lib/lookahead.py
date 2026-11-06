"""Effective look-ahead of a world model's collision check: one hazard seen from successive replanning moments.
For each window (decision time t0, straight logged ego path, mean speed >= 8 m/s), a real-shape car is inserted in the
ego's lane ahead, moving straight at a constant HALF the ego's mean speed (the same car and speed at every lead). Its
start gap is set so that the logged ego path is OVERLAP = 1.5 m into it at t_c = L, for leads L = 1, 2, 3 s. Only the
distance at the decision moment changes with L; scene, car shape and car speed are held fixed. This is how the same
hazard appears to a planner replanning 3, 2 and 1 s before contact.
The model receives the car in its 1.5 s history. The check (vehicle p >= 0.2 anywhere in the ego footprint) must fire
at h = L. A paired run without the car (same seed) gives false alarms. A constant-velocity check notices every one by
construction; a static copy of the present frame is recorded geometrically ("copy_hit": the car at its t0 position
overlaps the ego footprint at h).
For the planner test (src/exploit.py), it also saves the predicted vehicle probability along the ego's lane
(x in [-12, 40) m, |y| <= 2.4 m, at h = 1, 2, 3) with and without the car.
Models: dome | come | come_eval.py specs (released, <ckpt>, flow:<ckpt>).
Usage: python src/lookahead.py <model> <fold> <n_windows> <out.parquet> [--seed S]
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

LEADS = (1.0, 2.0, 3.0)
V_EGO_MIN = 8.0
GAP_FLOOR = 1.0                     # the car's rear is at least 1 m ahead of the ego's front at t0
STRIP = (slice(70, 200), slice(94, 106))


def plan_fixed(ps, L, v_car):
    """start offset (centre ahead of the ego origin at t0) so that the ego front is OVERLAP m into a car moving at
    v_car at t = L; None if the gap at t0 would be below GAP_FLOOR."""
    se = {t: ph.along(ps[3], P)[0] for t, P in zip((1.0, 2.0, 3.0), (ps[5], ps[7], ps[9]))}
    front0 = ph.EGO[0] + ph.EGO[2] / 2
    rear0 = se[L] + front0 - ph.OVERLAP - v_car * L
    return None if rear0 - front0 < GAP_FLOOR else rear0


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
        v_ego = ph.along(ps[3], ps[9])[0] / 3.0
        if v_ego < V_EGO_MIN:
            continue
        v_car = 0.5 * v_ego
        patch = bank[int(fr[3]["token"][:8], 16) % len(bank)]
        keep = [0, 1, 2, 3, 5, 7, 9]
        occs = [di.load_occ(scene, fr[k]["token"]) for k in keep]
        kp = [ps[k] for k in keep]
        xas = {}
        for L in LEADS:
            r0 = plan_fixed(ps, L, v_car)
            if r0 is None:
                break
            xa = r0 + patch["dims"][0] / 2
            ok = True
            for k, t in enumerate(pi.FT):
                xy, yaw = ph.car_xy(ps[3], xa, v_car, t)
                m, c = pi.footprint(kp[k], xy, yaw, dims=patch["dims"], grow=0.4)
                if not m.any() or abs(c[0]) > 36 or np.isin(occs[k][m], pi.OBST).any() or \
                        (occs[k][m] == pi.DRIVABLE).any(1).mean() < 0.8:
                    ok = False
                    break
            if not ok:
                break
            xas[L] = xa
        if len(xas) == len(LEADS):
            out.append({"scene": scene, "idx": idx, "token": fr[3]["token"], "v_car": v_car, "xas": xas,
                        "v_ego": v_ego})
        if len(out) == n:
            break
    return infos, out


def get_model(spec):
    if spec == "dome":
        return pi.Dome()
    if spec == "come":
        return pi.Come()
    import come_eval as ce
    return ce.come_with(spec)()


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
    print(f"{len(sel)} windows valid at all leads {LEADS}", flush=True)
    net = get_model(a.model)
    em = ph.ego_mask()
    rows, strips_in, strips_base = [], [], []
    for b0 in range(0, len(sel), a.bs):
        batch = sel[b0: b0 + a.bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s in batch]
        poses = [[di.pose_of(f) for f in fr] for fr in frs]
        base = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for s, fr in zip(batch, frs)]
        patches = [bank[int(s["token"][:8], 16) % len(bank)] for s in batch]
        Pb = net(base, poses, seed=1000 * a.seed + b0)
        for L in LEADS:
            ins = [np.stack([pi.insert(bp[k], ps[k], *ph.car_xy(ps[3], s["xas"][L], s["v_car"], pi.FT[k]), patch=pa)
                             for k in range(4)]) for s, bp, ps, pa in zip(batch, base, poses, patches)]
            Pi = net(ins, poses, seed=1000 * a.seed + b0)
            for s, ps, pa, pb, pin in zip(batch, poses, patches, Pb, Pi):
                v, xa = s["v_car"], s["xas"][L]
                fp = [ps[3], ps[5], ps[7], ps[9]]
                h = int(L)
                m0, _ = pi.footprint(ps[3], *ph.car_xy(ps[3], xa, v, 0.0), dims=pa["dims"])
                mc, _ = pi.footprint(fp[h], *ph.car_xy(ps[3], xa, v, float(h)), dims=pa["dims"])
                mcopy, _ = pi.footprint(fp[h], *ph.car_xy(ps[3], xa, v, 0.0), dims=pa["dims"])
                rows.append({"token": s["token"], "scene": s["scene"], "lead": L, "v_car": v, "v_ego": s["v_ego"],
                             "xa": xa, "car_len": float(pa["dims"][0]),
                             "s_ego": [ph.along(ps[3], P)[0] for P in (ps[5], ps[7], ps[9])],
                             "perceived": pi.q50(pin[0], m0) >= 0.2, "geom": bool((mc & em).any()),
                             "copy_hit": bool((mcopy & em).any()),
                             "ego_max_ins": [float(pin[k][em].max()) for k in (1, 2, 3)],
                             "ego_max_base": [float(pb[k][em].max()) for k in (1, 2, 3)],
                             "car_q50": pi.q50(pin[h], mc)})
                strips_in.append(pin[1:4][(slice(None),) + STRIP].astype(np.float16))
                strips_base.append(pb[1:4][(slice(None),) + STRIP].astype(np.float16))
        print(f"{b0 + len(batch)}/{len(sel)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    np.savez_compressed(a.out.replace(".parquet", "_strips.npz"), ins=np.stack(strips_in), base=np.stack(strips_base))
    d = df[df.perceived].copy()
    d["hit"] = [r.ego_max_ins[int(r.lead) - 1] >= 0.2 for r in d.itertuples()]
    d["fa"] = [r.ego_max_base[int(r.lead) - 1] >= 0.2 for r in d.itertuples()]
    print(d.groupby("lead").agg(n=("hit", "size"), geom=("geom", "mean"), copy=("copy_hit", "mean"),
                                noticed=("hit", "mean"), false_alarm=("fa", "mean"),
                                car_kept=("car_q50", lambda x: (x >= 0.2).mean())).round(3).to_string())


if __name__ == "__main__":
    main()
