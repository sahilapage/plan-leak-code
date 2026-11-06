"""A37 scoring (CPU; registered before the main run). Reads <out>/<token>/<plan>_s<seed>/front3.npz + rel_poses.npy
written by src/a37_omni.py and the windows of src/a37_windows.py.
Geometry: generated metric depth (z-depth, as OmniNWM's reward uses it) and semantics of the three front cameras,
back-projected with the batch intrinsics / cam2ego extrinsics into the ego frame at frame t (ego_t), and into the t0
ego frame with the plan's own conditioning pose M_t. Dynamic = semantic ids {3 car, 4 person, 5 bicycle} with
0 < depth < 25 m (OmniNWM's reward rule); static = {0 object, 6 lane, 7 sidewalk, 8 vegetation}, 0 < depth < 25 m.
Judges of a candidate plan P (crash = flag at ANY of the 33 frames):
  released   OmniNWM's calculate_rewards collision rule as shipped: a dynamic point inside the axis-aligned bounds of the
             +-2 m x +-1 m box rotated to the plan pose (x, y, yaw) = traj_gt[t], points in the ego_t frame
  self       frame-consistent: dynamic points of P's own generation, in ego_t, inside the ego box (centre 1.3 m ahead
             of the rear axle, 4.1 x 1.75 m)
  pm         plan-marginal (the training-free fix): dynamic points of each past-only generation Q in {pm0,
             pm+1.5} mapped ego_t(Q) -> t0 -> ego_t(P); per frame the share of Q with a point inside P's ego box;
             crash if the share >= 0.5 at any frame
Truth: collide plans collide (>= 2 m penetration into the annotated lead by construction), logged plans do not.
Leak: lead rear in the t0 frame = 5th percentile x of the generated vehicle component tracked from the annotated lead
(track_lead); beta = (rear_collide - rear_logged) / ds at t* = the first frame of true contact of the collide plan
(true gap <= 0), with ds = s_collide - s_logged >= 1 m (window dropped otherwise);
beta 0 = the lead stays where the world puts it (truth for a non-reacting lead), 1 = it keeps its gap to the ego.
Readout validity (G2): |rear_logged(t) - true rear(t)| at t ~ 1 s and 2 s. Controllability (G1): realized ego progress
s_hat(t) = the x shift that best aligns the static-point occupancy grid (0.5 m) of frame t with frame 0, same generation;
ratio (s_hat_collide - s_hat_logged) / ds at t*.
Usage: python src/a37_score.py <windows.json> <out_dir> <dir suffix, e.g. _s0_n25> [pilot_tokens.txt]
(scene-cluster bootstrap; pilot windows are reported separately and never enter the confirmatory numbers)"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DYN, STAT = (3, 4, 5), (0, 8)   # static for progress: object + vegetation (lane / sidewalk / road are shift-invariant)
BOX = (1.3, 4.1, 1.75)
FRONT0 = BOX[0] + BOX[1] / 2
PRIORS = ("pm0", "pm-1.5")
STRIDE = 2
DMIN, NMIN = 1.0, 50            # our judges / readout: depth >= 1 m (lens artifacts), >= 50 points (stride 2; A37 amendment 1)


def backproject(depth, seg, K, C, classes, stride=STRIDE, dmin=None):
    """depth/seg (3, T, H, W); K (3, T, 3, 3); C (3, T, 4, 4) -> list over t of (N, 3) points in ego_t."""
    T = depth.shape[1]
    out = []
    for t in range(T):
        pts = []
        for v in range(depth.shape[0]):
            d = depth[v, t, ::stride, ::stride].astype(np.float32)
            m = np.isin(seg[v, t, ::stride, ::stride], classes) & (d >= (DMIN if dmin is None else dmin)) & (d < 25)
            yy, xx = np.nonzero(m)
            if len(xx) == 0:
                continue
            uv1 = np.stack([xx * stride, yy * stride, np.ones_like(xx)], 0).astype(np.float64)
            pc = np.linalg.inv(K[v, t]) @ uv1 * d[yy, xx]
            pe = (C[v, t] @ np.vstack([pc, np.ones((1, pc.shape[1]))]))[:3].T
            pts.append(pe)
        out.append(np.concatenate(pts) if pts else np.zeros((0, 3)))
    return out


def yaw(M):
    return math.atan2(M[1, 0], M[0, 0])


def to_t0(P, M):
    return (M[:3, :3] @ P.T).T + M[:3, 3] if len(P) else P


def from_t0(P, M):
    return (M[:3, :3].T @ (P - M[:3, 3]).T).T if len(P) else P


def in_box(P):
    cx, L, W = BOX
    return (np.abs(P[:, 0] - cx) <= L / 2) & (np.abs(P[:, 1]) <= W / 2) if len(P) else np.zeros(0, bool)


def released_flags(dyn, M):
    f = []
    for t, P in enumerate(dyn):
        x, y, th = M[t, 0, 3], M[t, 1, 3], yaw(M[t])
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        c = (R @ np.array([[2, 1], [2, -1], [-2, -1], [-2, 1]], float).T).T + [x, y]
        lo, hi = c.min(0), c.max(0)
        f.append(bool(len(P)) and bool(((P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) &
                                        (P[:, 1] <= hi[1])).any()))
    return np.array(f)


def lead_rear(dyn, M, w, t):
    tr = np.array(w["lead_track_t0frame"])
    tt = np.arange(len(tr)) * 0.5
    ly = np.interp(t, tt, tr[:, 1])
    rear = []
    for k, P in enumerate(dyn):
        Q = to_t0(P, M[k])
        sel = (np.abs(Q[:, 1] - ly[k]) <= 1.5) & (Q[:, 0] >= M[k, 0, 3] + FRONT0 - 1.0) if len(Q) else []
        rear.append(np.percentile(Q[sel, 0], 10) if len(Q) and sel.sum() >= 30 else np.nan)
    return np.array(rear)


def components(Q, cell=0.5, min_pts=30):
    """connected components (8-neighbour, cells with >= 2 points) of t0-frame points Q -> list of point arrays."""
    from scipy import ndimage
    if len(Q) < min_pts:
        return []
    ij = np.floor(Q[:, :2] / cell).astype(int)
    lo = ij.min(0)
    ij -= lo
    G = np.zeros(ij.max(0) + 1, int)
    np.add.at(G, (ij[:, 0], ij[:, 1]), 1)
    lab, n = ndimage.label(G >= 2, structure=np.ones((3, 3)))
    pl = lab[ij[:, 0], ij[:, 1]]
    return [Q[pl == k] for k in range(1, n + 1) if (pl == k).sum() >= min_pts]


def track_lead_xy(dyn, M, w):
    """(rear x, median y) of the tracked lead per frame (descriptive: is a missed crash a flee or a lateral offset?)."""
    x0, y0 = w["lead_track_t0frame"][0][:2]
    out, prev, miss = [], np.array([x0, y0]), 0
    for k, P in enumerate(dyn):
        Q = to_t0(P, M[k])
        Q = Q[(Q[:, 2] > 0.3) & (Q[:, 2] < 3.5)] if len(Q) else Q
        best, bd = None, (4.0 if k == 0 else 3.0 + 1.5 * miss)
        for C in components(Q):
            c = np.median(C[:, :2], 0)
            if np.hypot(*(c - prev)) < bd:
                best, bd = C, np.hypot(*(c - prev))
        if best is None:
            out.append((np.nan, np.nan))
            miss += 1
            continue
        miss, prev = 0, np.median(best[:, :2], 0)
        out.append((float(np.percentile(best[:, 0], 5)), float(prev[1])))
    return np.array(out)


def track_lead(dyn, M, w):
    """rear x (t0 frame) of the generated lead per frame: the vehicle component nearest the annotated lead at t0,
    followed by nearest centroid (gate 3 m, +1.5 m per missed frame). NaN where lost."""
    x0, y0 = w["lead_track_t0frame"][0][:2]
    rear, prev, miss = [], np.array([x0, y0]), 0
    for k, P in enumerate(dyn):
        Q = to_t0(P, M[k])
        Q = Q[(Q[:, 2] > 0.3) & (Q[:, 2] < 3.5)] if len(Q) else Q
        best, bd = None, (4.0 if k == 0 else 3.0 + 1.5 * miss)
        for C in components(Q):
            c = np.median(C[:, :2], 0)
            dd = np.hypot(*(c - prev))
            if dd < bd:
                best, bd = C, dd
        if best is None:
            rear.append(np.nan)
            miss += 1
            continue
        miss, prev = 0, np.median(best[:, :2], 0)
        rear.append(float(np.percentile(best[:, 0], 5)))
    return np.array(rear)


def progress(stat, cell=0.25):
    """realized ego progress: frame-to-frame x shift (-0.5 .. 2 m) maximizing the correlation of static-point
    (object / vegetation, 0.3-4 m high, < 22 m ahead) sqrt-count grids, accumulated from frame 0."""
    def grid(P):
        P = P[(P[:, 2] > 0.3) & (P[:, 2] < 4.0) & (P[:, 0] < 22)] if len(P) else P
        H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=(np.arange(-5, 45.01, cell), np.arange(-15, 15.01, cell))) \
            if len(P) else (np.zeros((200, 120)), 0, 0)
        return np.sqrt(H)
    s = [0.0]
    for k in range(1, len(stat)):
        g0 = grid(stat[k - 1])
        best = max(np.arange(-0.5, 2.01, 0.05), key=lambda dx: float((grid(stat[k] + [dx, 0, 0]) * g0).sum()))
        s.append(s[-1] + best)
    return np.array(s)


def load(d):
    z = np.load(d / "front3.npz")
    M = np.load(d / "rel_poses.npy")
    dyn = backproject(z["depth"], z["seg"], z["cam_k"], z["cam2ego"], DYN)
    dyn_rel = backproject(z["depth"], z["seg"], z["cam_k"], z["cam2ego"], DYN, dmin=1e-6)  # released rule: d > 0
    return {"M": M, "dyn": dyn, "dyn_rel": dyn_rel, "z": z}


def main(windows, out, suf="_s0", excl=None):
    W = {w["token"]: w for w in json.load(open(windows))}
    ex = set(open(excl).read().split()) if excl else set()
    out = Path(out)
    rows = []
    for tok in sorted(p.name for p in out.iterdir() if p.is_dir()):
        w = W[tok]
        G = {}
        for pl in ("logged", "collide") + PRIORS:
            d = out / tok / f"{pl}{suf}"
            if (d / "front3.npz").exists():
                G[pl] = load(d)
        if not {"logged", "collide"} <= set(G):
            continue
        t = np.array(json.load(open(out / tok / f"logged{suf}" / "reward.json"))["t"])
        tr = np.array(w["lead_track_t0frame"])
        L = w["lead_LW"][0]
        x_true = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 0]) - L / 2      # true lead rear (t0 frame)
        s_c, s_l = G["collide"]["M"][:, 0, 3], G["logged"]["M"][:, 0, 3]
        gap_c = x_true - (s_c + FRONT0)
        ds = s_c - s_l
        hit = np.nonzero(gap_c <= 0)[0]                                      # true contact of the collide plan
        ok = [hit[0]] if len(hit) and ds[hit[0]] >= 1.0 else []
        r_l = track_lead(G["logged"]["dyn"], G["logged"]["M"], w)
        r_c = track_lead(G["collide"]["dyn"], G["collide"]["M"], w)
        row = {"token": tok, "scene": w["scene"], "pilot": tok in ex, "a_star": w["a_star"],
               "lead_x0": tr[0, 0], "lead_speed": w["lead_speed"]}
        if len(ok):
            k = ok[-1]
            st_c = backproject(G["collide"]["z"]["depth"], G["collide"]["z"]["seg"], G["collide"]["z"]["cam_k"],
                               G["collide"]["z"]["cam2ego"], STAT)
            st_l = backproject(G["logged"]["z"]["depth"], G["logged"]["z"]["seg"], G["logged"]["z"]["cam_k"],
                               G["logged"]["z"]["cam2ego"], STAT)
            sh_c, sh_l = progress(st_c[:k + 1]), progress(st_l[:k + 1])
            dsh = sh_c[-1] - sh_l[-1]
            xy = track_lead_xy(G["collide"]["dyn"], G["collide"]["M"], w)[k]
            row.update({"collide_gen_gap_at_contact": xy[0] - (s_c[k] + FRONT0),     # > 0: the lead fled ahead
                        "collide_gen_lat_at_contact": xy[1] - G["collide"]["M"][k, 1, 3],  # |.| > ~2 m: beside
                        "lead_stopped": w["lead_speed"] < 0.5})
            row.update({"t_star": t[k], "ds": ds[k], "beta": (r_c[k] - r_l[k]) / ds[k],
                        "beta_realized": (r_c[k] - r_l[k]) / dsh if dsh >= 0.5 else np.nan,
                        "ctrl_ratio": dsh / ds[k], "s_hat_logged": sh_l[-1], "s_logged": s_l[k]})
        for tt in (1.0, 2.0):
            j = int(np.argmin(np.abs(t - tt)))
            row[f"readout_err_{int(tt)}s"] = abs(r_l[j] - x_true[j])
        for pl, truth in (("collide", 1), ("logged", 0)):
            P = G[pl]
            row[f"{pl}_released"] = int(released_flags(P["dyn_rel"], P["M"]).any())
            row[f"{pl}_self"] = int(any(in_box(D).sum() >= NMIN for D in P["dyn"]))
            if all(q in G for q in PRIORS):
                share = np.zeros(len(t))
                for q in PRIORS:
                    Q = G[q]
                    share += np.array([in_box(from_t0(to_t0(Q["dyn"][k], Q["M"][k]), P["M"][k])).sum() >= NMIN
                                       for k in range(len(t))]) / len(PRIORS)
                row[f"{pl}_pm"] = int((share >= 0.5).any())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"a37_scores{suf}.csv", index=False)
    conf = df[~df.pilot]
    print(f"windows scored {len(df)} (confirmatory {len(conf)}, pilot {df.pilot.sum()})")
    rng = np.random.default_rng(0)

    def boot(x, f=np.nanmedian, B=2000):
        """scene-cluster bootstrap of statistic f over the finite values of the Series x."""
        x = x[np.isfinite(x.astype(float))]
        if len(x) == 0:
            return np.nan, np.nan, np.nan
        sc = df.loc[x.index, "scene"]
        by = {k: x[sc == k].to_numpy(float) for k in sc.unique()}
        keys = list(by)
        bs = [f(np.concatenate([by[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)]
        return f(x.to_numpy(float)), np.quantile(bs, .025), np.quantile(bs, .975)
    for lab, d in (("confirmatory", conf), ("pilot", df[df.pilot])):
        if d.empty:
            continue
        print(f"== {lab} (n {len(d)})")
        if "beta" in d:
            print("  beta (lead)        median %.2f [%.2f, %.2f]" % boot(d.beta), f"n {d.beta.notna().sum()}")
            print("  controllability    median %.2f [%.2f, %.2f]" % boot(d.ctrl_ratio))
        print("  readout err 1 s / 2 s  median %.2f / %.2f m" % (d.readout_err_1s.median(), d.readout_err_2s.median()))
        if "lead_stopped" in d and "beta" in d:
            for lab2, m in (("stopped", d.lead_stopped == True), ("moving", d.lead_stopped == False)):  # noqa: E712
                if m.any():
                    print(f"  beta {lab2:8s} median %.2f (n {int(m.sum())})" % d.beta[m].median())
        if "collide_self" in d and "collide_gen_gap_at_contact" in d:
            ms = d[d.collide_self == 0]
            print(f"  self misses: {len(ms)}; generated lead ahead of the ego front at contact (fled) "
                  f"{(ms.collide_gen_gap_at_contact > 0.5).mean():.2f}, laterally >= 2 m off {(ms.collide_gen_lat_at_contact.abs() >= 2).mean():.2f}")
        for j in ("released", "self", "pm"):
            if f"collide_{j}" in d:
                print(f"  {j:9s} miss %.2f [%.2f, %.2f]" % boot(1 - d[f"collide_{j}"], np.mean) +
                      f" | false alarm %.2f" % d[f"logged_{j}"].mean())
        if "collide_pm" in d:
            diff = (1 - d.collide_pm) - (1 - d.collide_self)
            print("  miss pm - self  %.2f [%.2f, %.2f]" % boot(diff, np.mean))


if __name__ == "__main__":
    main(*sys.argv[1:])
