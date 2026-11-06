"""Plan-swap judging: judge a candidate path against a forecast that is NOT conditioned on it.
A plan-conditioned world model (DOME) clears the path it is conditioned on (A19). Test-time fix: forecast once under a fixed
reference plan that does not depend on the scene -- brake from the t0 speed at A_REF m/s^2 along the t0 heading -- and
check the candidate geometrically: the candidate's ego box at h, placed in the reference forecast's frame at h.
Compared with the usual self-conditioned check (forecast conditioned on the candidate itself; same diffusion seed).
Candidate = the logged path, which collides with the inserted car.
  mode pc: A19 windows (src/plan_cond.py selection): PARKED car in the ego's lane, logged path 1.5 m into it at +2 s.
  mode la: A18 windows (src/lookahead.py selection): car at half the ego's speed, contact at L = 1, 2, 3 s.
A no-car run under each conditioning gives false alarms (the logged path never collides with real traffic).
  mode real: random windows of the fold, NO insertion: the logged path is collision-free by definition, so any firing of
             the check at +1 / +2 / +3 s is a false alarm on real traffic (self-conditioned vs plan-swap).
Usage: python src/planswap.py <pc|la|real> <fold> <n> <out.parquet> [--aref 4.0]
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

EGO_DIMS = (ph.EGO[2], ph.EGO[3])


def ref_poses(ps, a_ref):
    """poses 4..9 (t = 0.5 .. 3 s) of the reference plan: t0 heading, constant deceleration a_ref from the t0 speed."""
    v0 = ph.along(ps[2], ps[3])[0] / 0.5
    yaw0 = pi.yaw_of(ps[3])
    e = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    out = []
    for k in range(1, 7):
        t = 0.5 * k
        s = v0 * t - 0.5 * a_ref * t ** 2 if t < v0 / a_ref else v0 ** 2 / (2 * a_ref)
        P = ps[3].copy()
        P[:3, 3] = ps[3][:3, 3] + s * e
        out.append(P)
    return out


def ego_box_in(P_frame, P_ego):
    """the ego box of pose P_ego, as a (200, 200) mask in the ego frame of P_frame."""
    c = P_ego @ np.array([ph.EGO[0], ph.EGO[1], 0.0, 1.0])
    m, _ = pi.footprint(P_frame, c[:2], pi.yaw_of(P_ego), dims=EGO_DIMS)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("fold", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--aref", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    import pandas as pd
    bank = pi.build_bank(0, 60, exclude=set()) if a.mode != "real" else None
    if a.mode == "real":
        infos, wins = di.windows({a.fold})
        rng = np.random.default_rng(0)
        sel = [{"scene": wins[j][0], "idx": wins[j][1], "token": infos[wins[j][0]][wins[j][1] + 3]["token"]}
               for j in rng.permutation(len(wins))[: a.n]]
        cases = [(s, None) for s in sel]
    elif a.mode == "pc":
        import plan_cond as pc
        infos, sel = pc.select(a.fold, a.n, 0, bank)
        cases = [(s, None) for s in sel]
    else:
        import lookahead as la
        infos, sel = la.select(a.fold, a.n, 0, bank)
        cases = [(s, L) for s in sel for L in la.LEADS]
    print(f"{len(sel)} windows, {len(cases)} cases", flush=True)
    net = pi.Dome()
    if a.mode == "real":
        return real(a, net)
    rows = []
    for b0 in range(0, len(cases), a.bs):
        batch = cases[b0: b0 + a.bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s, _ in batch]
        logged = [[di.pose_of(f) for f in fr] for fr in frs]
        refp = [lp[:4] + ref_poses(lp, a.aref) for lp in logged]
        base = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for (s, _), fr in zip(batch, frs)]
        patches = [bank[int(s["token"][:8], 16) % len(bank)] for s, _ in batch]

        ins = []
        for (s, L), bp, lp, pa in zip(batch, base, logged, patches):
            fr_ins = []
            for k in range(4):
                if L is None:
                    xy, yaw = s["xy"]["inpath"], s["yaw"]
                else:
                    xy, yaw = ph.car_xy(lp[3], s["xas"][L], s["v_car"], pi.FT[k])
                fr_ins.append(pi.insert(bp[k], lp[k], xy, yaw, patch=pa))
            ins.append(np.stack(fr_ins))
        seed = b0
        out = {(cond, car_): net(occ, poses, seed=seed)
               for cond, poses in (("self", logged), ("swap", refp))
               for car_, occ in (("car", ins), ("base", base))}
        for i, ((s, L), lp, rp, pa) in enumerate(zip(batch, logged, refp, patches)):
            hs = [2] if L is None else [int(L)]
            for h in (1, 2, 3):
                Pl, Pr = lp[3 + 2 * h], rp[3 + 2 * h]
                if L is None:
                    xy, yaw = s["xy"]["inpath"], s["yaw"]
                else:
                    xy, yaw = ph.car_xy(lp[3], s["xas"][L], s["v_car"], float(h))
                row = {"token": s["token"], "scene": s["scene"], "lead": L if L is not None else 2.0, "h": h,
                       "contact": h in hs}
                for cond, Pf in (("self", Pl), ("swap", Pr)):
                    em = ego_box_in(Pf, Pl)                    # the logged candidate's ego box, in this forecast's frame
                    mc, _ = pi.footprint(Pf, xy, yaw, dims=pa["dims"])
                    row[f"{cond}_check"] = float(out[(cond, "car")][i][h][em].max())
                    row[f"{cond}_check_base"] = float(out[(cond, "base")][i][h][em].max())
                    row[f"{cond}_car_q50"] = pi.q50(out[(cond, "car")][i][h], mc)
                m0, _ = pi.footprint(lp[3], *(((s["xy"]["inpath"], s["yaw"]) if L is None else
                                               ph.car_xy(lp[3], s["xas"][L], s["v_car"], 0.0))), dims=pa["dims"])
                row["perceived"] = pi.q50(out[("self", "car")][i][0], m0) >= 0.2
                rows.append(row)
        print(f"{b0 + len(batch)}/{len(cases)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    d = df[df.contact & df.perceived]
    print(d.groupby("lead").agg(n=("token", "size"),
                                self_noticed=("self_check", lambda x: (x >= 0.2).mean()),
                                swap_noticed=("swap_check", lambda x: (x >= 0.2).mean()),
                                self_fa=("self_check_base", lambda x: (x >= 0.2).mean()),
                                swap_fa=("swap_check_base", lambda x: (x >= 0.2).mean()),
                                self_kept=("self_car_q50", lambda x: (x >= 0.2).mean()),
                                swap_kept=("swap_car_q50", lambda x: (x >= 0.2).mean())).round(3).to_string())


def real(a, net):
    import pandas as pd
    infos, wins = di.windows({a.fold})
    rng = np.random.default_rng(0)
    sel = [wins[j] for j in rng.permutation(len(wins))[: a.n]]
    rows = []
    for b0 in range(0, len(sel), a.bs):
        batch = sel[b0: b0 + a.bs]
        frs = [infos[sc][ix: ix + 10] for sc, ix in batch]
        logged = [[di.pose_of(f) for f in fr] for fr in frs]
        refp = [lp[:4] + ref_poses(lp, a.aref) for lp in logged]
        base = [np.stack([di.load_occ(sc, f["token"]) for f in fr[:4]]) for (sc, _), fr in zip(batch, frs)]
        out = {c: net(base, poses, seed=b0) for c, poses in (("self", logged), ("swap", refp))}
        for i, ((sc, ix), lp, rp, fr) in enumerate(zip(batch, logged, refp, frs)):
            for h in (1, 2, 3):
                rows.append({"token": fr[3]["token"], "scene": sc, "h": h,
                             "v_ego": ph.along(lp[3], lp[9])[0] / 3.0,
                             "self_check": float(out["self"][i][h][ego_box_in(lp[3 + 2 * h], lp[3 + 2 * h])].max()),
                             "swap_check": float(out["swap"][i][h][ego_box_in(rp[3 + 2 * h], lp[3 + 2 * h])].max())})
        print(f"{b0 + len(batch)}/{len(sel)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    w = df.groupby("token").agg(self_fa=("self_check", lambda x: (x >= 0.2).any()),
                                swap_fa=("swap_check", lambda x: (x >= 0.2).any()))
    print(f"real traffic, {len(w)} windows: false alarm on the logged (collision-free) path, any of +1/+2/+3 s: "
          f"self {w.self_fa.mean():.3f} | plan-swap {w.swap_fa.mean():.3f}")
    print(df.groupby("h").agg(self=("self_check", lambda x: (x >= 0.2).mean()),
                              swap=("swap_check", lambda x: (x >= 0.2).mean())).round(3).to_string())


if __name__ == "__main__":
    main()
