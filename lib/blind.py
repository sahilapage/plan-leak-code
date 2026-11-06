"""The blind judge: judge candidates against a forecast of an EGO-NEUTRAL world.
Plan-conditioned and ego-motion-conditioned world models clear what the ego would hit (A19; I2-World: the ego's approach).
Blind judging removes the ego's intent and motion from the world model's input: the past occupancy frames are re-expressed
in the t0 ego frame (as if the ego had stood still at its t0 pose; nearest-voxel resampling, unseen cells = free), and the
model is conditioned on a stationary ego (all past and future poses = the t0 pose). Every candidate is then checked
geometrically: its ego box at h, placed in the t0 frame. One forecast serves all candidates.
Compared on the same windows and seed with the self-conditioned judge (forecast conditioned on the candidate, the logged
path) and plan-swap (src/planswap.py reference plan). DOME here; I2-World via src/ii_blind.py.
  mode pc  : parked car on the logged path's collision course (src/plan_cond.py windows)
  mode la  : car at half the ego's speed, contact at L = 1, 2, 3 s (src/lookahead.py windows)
  mode real: random windows, no insertion: any firing on the collision-free logged path is a false alarm
Usage: python src/blind.py <pc|la|real> <fold> <n> <out.parquet>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_hazard as ph  # noqa: E402
import probe_insert as pi  # noqa: E402
from planswap import ego_box_in, ref_poses  # noqa: E402

A_REF_DEFAULT = 4.0                     # the plan-swap reference plan (A22)

FREE = 17
IJ = np.stack(np.meshgrid(np.arange(200), np.arange(200), indexing="ij"), -1).reshape(-1, 2)
XY0 = -40 + 0.4 * (IJ + 0.5)


def to_frame(occ, P_src, P_dst):
    """resample a (200, 200, 16) occupancy given in the ego frame of P_src into the ego frame of P_dst (top-down rigid
    transform, nearest voxel; cells not covered by the source grid are free)."""
    T = np.linalg.inv(P_src) @ P_dst
    p = XY0 @ T[:2, :2].T + T[:2, 3]
    ij = np.floor((p + 40) / 0.4).astype(int)
    ok = (ij >= 0).all(1) & (ij < 200).all(1)
    out = np.full_like(occ, FREE)
    src = occ.reshape(200, 200, 16)
    out.reshape(-1, 16)[ok] = src[ij[ok, 0], ij[ok, 1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("fold", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    import pandas as pd
    bank = pi.build_bank(0, 60, exclude=set())
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
    rows = []
    for b0 in range(0, len(cases), a.bs):
        batch = cases[b0: b0 + a.bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s, _ in batch]
        logged = [[di.pose_of(f) for f in fr] for fr in frs]
        refp = [lp[:4] + ref_poses(lp, A_REF_DEFAULT) for lp in logged]
        still = [[lp[3]] * 10 for lp in logged]
        base = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for (s, _), fr in zip(batch, frs)]
        patches = [bank[int(s["token"][:8], 16) % len(bank)] for s, _ in batch]
        if a.mode == "real":
            ins = base
        else:
            ins = []
            for (s, L), bp, lp, pa in zip(batch, base, logged, patches):
                fr_ins = []
                for k in range(4):
                    xy, yaw = (s["xy"]["inpath"], s["yaw"]) if L is None else \
                        ph.car_xy(lp[3], s["xas"][L], s["v_car"], pi.FT[k])
                    fr_ins.append(pi.insert(bp[k], lp[k], xy, yaw, patch=pa))
                ins.append(np.stack(fr_ins))

        def neutral(occs):
            return [np.stack([to_frame(o[k], lp[k], lp[3]) for k in range(4)]) for o, lp in zip(occs, logged)]

        out = {}
        for cond, poses, f in (("self", logged, lambda x: x), ("swap", refp, lambda x: x), ("blind", still, neutral)):
            out[(cond, "car")] = net(f(ins), poses, seed=b0)
            if a.mode != "real":
                out[(cond, "base")] = net(f(base), poses, seed=b0)
        for i, ((s, L), lp, rp) in enumerate(zip(batch, logged, refp)):
            frames = {"self": lambda h: lp[3 + 2 * h], "swap": lambda h: rp[3 + 2 * h], "blind": lambda h: lp[3]}
            Lc = 2.0 if L is None else L
            for h in (1, 2, 3):
                row = {"token": s["token"], "scene": s["scene"], "lead": Lc, "h": h, "contact": h == int(Lc),
                       "v_ego": ph.along(lp[3], lp[9])[0] / 3.0}
                if a.mode != "real":
                    pa = patches[i]
                    xy, yaw = (s["xy"]["inpath"], s["yaw"]) if L is None else ph.car_xy(lp[3], s["xas"][L], s["v_car"], float(h))
                    m0, _ = pi.footprint(lp[3], *(((s["xy"]["inpath"], s["yaw"]) if L is None else
                                                   ph.car_xy(lp[3], s["xas"][L], s["v_car"], 0.0))), dims=pa["dims"])
                    row["perceived"] = pi.q50(out[("self", "car")][i][0], m0) >= 0.2
                for cond in ("self", "swap", "blind"):
                    Pf = frames[cond](h)
                    em = ego_box_in(Pf, lp[3 + 2 * h])
                    row[f"{cond}_check"] = float(out[(cond, "car")][i][h][em].max())
                    if a.mode != "real":
                        row[f"{cond}_check_base"] = float(out[(cond, "base")][i][h][em].max())
                        mc, _ = pi.footprint(Pf, xy, yaw, dims=pa["dims"])
                        row[f"{cond}_car_q50"] = pi.q50(out[(cond, "car")][i][h], mc)
                rows.append(row)
        print(f"{b0 + len(batch)}/{len(cases)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    if a.mode == "real":
        w = df.groupby("token")[[f"{c}_check" for c in ("self", "swap", "blind")]].max() >= 0.2
        print(f"real traffic, {len(w)} windows, false alarm on the collision-free logged path:", w.mean().round(3).to_dict())
    else:
        d = df[df.contact & df.perceived]
        agg = {f"{c}_{k}": (f"{c}_{col}", lambda x: (x >= 0.2).mean())
               for c in ("self", "swap", "blind") for k, col in (("hit", "check"), ("fa", "check_base"), ("kept", "car_q50"))}
        print(d.groupby("lead").agg(n=("token", "size"), **agg).round(3).T.to_string())


if __name__ == "__main__":
    main()
