"""A37 side-lane control (descriptive, amendment 2): does the plan leak hit only the agent whose future is coupled to
the ego plan in the data (the in-lane lead), or every vehicle (a generic 'keep objects in place' bias)?
For every annotated vehicle 2-25 m ahead at t0 (|y| <= 8 m, nuScenes boxes in the t0 frame), track its generated
point cluster in the logged and the collide generation with an exclusive multi-object tracker (vehicle-class
components, src/a37_score.py components; frame 0: Hungarian match of annotated t0 centres to component centroids,
gate 2.5 m; then per frame Hungarian on centroid distance, gate 1.5 m, a track dies after 3 missed frames); at the last frame k <= true contact where it is tracked in both and the commanded extra advance is >= 1 m:
d_gap = (x_collide - s_collide) - (x_logged - s_logged) (ego-frame gap change); realized extra advance d_s_hat =
ctrl * d_s (ctrl from PnP odometry, src/a37_pnp_run.py; commanded if unmeasurable); beta_world = (d_gap + d_s_hat) /
d_s_hat. Truth 0 for all of them (no vehicle ahead reacts to the car behind within 2.5 s).
Groups: lead | same lane beyond the lead (|y| <= 1.5) | adjacent moving (1.5 < |y| <= 5.5, speed > 0.5 m/s) |
side stopped / parked (|y| > 1.5, speed <= 0.5).
Usage: python src/a37_control.py <windows.json> <out_dir> <scores.csv> <pnp.csv> <pilot_tokens.txt> <out.csv>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_score as S  # noqa: E402


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mot(dyn, M, seeds, gate0=2.5, gate=1.5, max_miss=3):
    """exclusive multi-object tracking of vehicle components (t0 frame) -> (n_seeds, T, 2) centroids, NaN if lost."""
    from scipy.optimize import linear_sum_assignment
    n, T = len(seeds), len(dyn)
    out = np.full((n, T, 2), np.nan)
    last, miss = np.array(seeds, float), np.zeros(n, int)
    alive = np.ones(n, bool)
    for k in range(T):
        Q = S.to_t0(dyn[k], M[k])
        Q = Q[(Q[:, 2] > 0.3) & (Q[:, 2] < 3.5)] if len(Q) else Q
        C = np.array([np.median(c[:, :2], 0) for c in S.components(Q)]).reshape(-1, 2)
        idx = np.nonzero(alive)[0]
        if len(C) and len(idx):
            D = np.linalg.norm(last[idx, None] - C[None], axis=2)
            r, c = linear_sum_assignment(D)
            g = gate0 if k == 0 else gate
            got = set()
            for i, j in zip(r, c):
                if D[i, j] < g:
                    out[idx[i], k], last[idx[i]], miss[idx[i]] = C[j], C[j], 0
                    got.add(idx[i])
            for i in idx:
                if i not in got:
                    miss[i] += 1
        else:
            miss[idx] += 1
        alive &= ~((miss > max_miss) | ((k == 0) & (miss > 0)))
    return out


def main(windows, out, scores, pnp, excl, dst):
    W = {w["token"]: w for w in json.load(open(windows))}
    sc = pd.read_csv(scores).set_index("token")
    pn = pd.read_csv(pnp).set_index("tok")
    ex = set(open(excl).read().split())
    rows = []
    for tok in sc.index:
        if tok in ex or not np.isfinite(sc.loc[tok, "t_star"]):
            continue
        w = W[tok]
        d = Path(out) / tok
        t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
        K = int(np.argmin(np.abs(t - sc.loc[tok, "t_star"])))
        G = {}
        for pl in ("logged", "collide"):
            z = np.load(d / f"{pl}_s0_n25" / "front3.npz")
            G[pl] = (np.load(d / f"{pl}_s0_n25" / "rel_poses.npy"),
                     S.backproject(z["depth"], z["seg"], z["cam_k"], z["cam2ego"], S.DYN))
        ds = G["collide"][0][:, 0, 3] - G["logged"][0][:, 0, 3]
        p = pn.loc[tok[:8]] if tok[:8] in pn.index else None
        ctrl = p.ds_chain / p.ds_cmd if p is not None and np.isfinite(p.ds_chain) else np.nan
        f0 = w["frames"][0]
        R0, T0 = quat_R(f0["ego_q"]), np.array(f0["ego_t"])
        V = []
        for v in w["vehicles"]:
            tr = np.array([(R0.T @ (np.array(b["t"]) - T0)) for b in v["track"]])
            x0, y0 = tr[0, :2]
            if not (2 <= x0 <= 25 and abs(y0) <= 8):
                continue
            spd = np.hypot(*(tr[1, :2] - tr[0, :2])) / 0.5 if len(tr) > 1 else 0.0
            lead = v["instance"] == w["lead_instance"]
            grp = ("lead" if lead else "same_lane" if abs(y0) <= 1.5 else
                   "adjacent_moving" if abs(y0) <= 5.5 and spd > 0.5 else "side_stopped" if spd <= 0.5 else "far_moving")
            V.append((grp, x0, y0, spd))
        if not V:
            continue
        xy = {pl: mot(G[pl][1], G[pl][0], [(v[1], v[2]) for v in V]) for pl in G}
        for i, (grp, x0, y0, spd) in enumerate(V):
            ok = [k for k in range(K + 1) if np.isfinite(xy["logged"][i, k, 0]) and np.isfinite(xy["collide"][i, k, 0])
                  and ds[k] >= 1.0]
            base = {"token": tok, "scene": w["scene"], "group": grp, "x0": x0, "y0": y0, "speed": spd}
            if not ok:
                rows.append(base | {"tracked": False})
                continue
            k = ok[-1]
            dgap = (xy["collide"][i, k, 0] - G["collide"][0][k, 0, 3]) - (xy["logged"][i, k, 0] - G["logged"][0][k, 0, 3])
            dsh = (ctrl if np.isfinite(ctrl) else 1.0) * ds[k]
            rows.append(base | {"tracked": True, "t": t[k], "ds_cmd": ds[k], "ds_hat": dsh,
                                "ctrl_measured": np.isfinite(ctrl), "dgap": dgap, "beta_world": (dgap + dsh) / dsh,
                                "beta_cmd": (dgap + ds[k]) / ds[k]})
    df = pd.DataFrame(rows)
    df.to_csv(dst, index=False)
    T = df[df.tracked == True]  # noqa: E712
    print(f"vehicles {len(df)} (tracked in both plans {len(T)}) in {df.token.nunique()} windows")
    rng = np.random.default_rng(0)
    for g in ("lead", "same_lane", "adjacent_moving", "side_stopped", "far_moving"):
        x = T[T.group == g]
        if x.empty:
            continue
        sc_ = x.scene.unique()
        bs = [np.median(np.concatenate([x[x.scene == s].beta_world.to_numpy() for s in rng.choice(sc_, len(sc_))]))
              for _ in range(2000)]
        print(f"  {g:16s} n {len(x):3d} ({len(sc_)} scenes) beta_world median {x.beta_world.median():+.2f} "
              f"[{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}] | beta_cmd {x.beta_cmd.median():+.2f} | "
              f"untracked {(df[(df.group == g)].tracked == False).sum()}")  # noqa: E712


if __name__ == "__main__":
    main(*sys.argv[1:])
