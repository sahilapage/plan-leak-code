"""FRESH AUDIT (2026-10-19): OmniNWM plan leak re-derived from raw generated depth + semantics, without importing any
project scorer. Different estimator from the registered one (src/a37_score.py tracks the generated vehicle component
nearest the annotated lead): here the lead = car-class points (seg 3 only) inside a 1.2 m lateral corridor around the
TRUE lead's lateral position (t0 frame), ahead of the ego front, nearest surface = 5th percentile of x.
beta = (lead_collide - lead_logged) / (s_collide - s_logged) at t* = first frame where the collide plan's ego front
(rear axle + 3.35 m) reaches the annotated lead rear (same contact rule as registered, recomputed here).
Also: (i) a fresh self-check (>= 50 car points inside the ego box at any frame, ego_t frame) on collide and logged;
(ii) the dose slope over d-2 / d1 / collide at t_m = min(1.5 s, contact) on the moving-ego windows.
Usage: python fresh_omninwm.py <windows.json> <out_omni> <main18.txt> <dose_tokens.txt>"""
import json
import sys
from pathlib import Path

import numpy as np

CAR, STRIDE, DMIN, DMAX = 3, 2, 1.0, 40.0
FRONT = 1.3 + 4.1 / 2                       # ego front ahead of the rear axle (m)
EGO_X, EGO_Y = (1.3 - 4.1 / 2, FRONT), 1.75 / 2


def cars_t0(d):
    """per frame: (N, 3) car points in the t0 frame and in the ego_t frame."""
    z = np.load(d / "front3.npz")
    M = np.load(d / "rel_poses.npy")
    dep, seg, K, C = z["depth"], z["seg"], z["cam_k"], z["cam2ego"]
    out_t0, out_e = [], []
    for t in range(dep.shape[1]):
        P = []
        for v in range(dep.shape[0]):
            dd = dep[v, t, ::STRIDE, ::STRIDE].astype(np.float32)
            m = (seg[v, t, ::STRIDE, ::STRIDE] == CAR) & (dd >= DMIN) & (dd < DMAX)
            yy, xx = np.nonzero(m)
            if len(xx) == 0:
                continue
            uv1 = np.stack([xx * STRIDE, yy * STRIDE, np.ones_like(xx)]).astype(np.float64)
            pc = np.linalg.solve(K[v, t].astype(np.float64), uv1) * dd[yy, xx]
            pe = (C[v, t].astype(np.float64) @ np.vstack([pc, np.ones((1, pc.shape[1]))]))[:3].T
            P.append(pe)
        pe = np.concatenate(P) if P else np.zeros((0, 3))
        out_e.append(pe)
        out_t0.append((M[t] @ np.c_[pe, np.ones(len(pe))].T)[:3].T if len(pe) else pe)
    return out_t0, out_e, M


def lead_x(pts, y_lead, x_min):
    sel = pts[(np.abs(pts[:, 1] - y_lead) <= 1.2) & (pts[:, 0] > x_min)] if len(pts) else pts
    return float(np.percentile(sel[:, 0], 5)) if len(sel) >= 20 else np.nan


def self_hit(pts_e):
    """any frame with >= 50 car points inside the ego box (ego_t frame)."""
    for p in pts_e:
        if len(p) and ((p[:, 0] >= EGO_X[0]) & (p[:, 0] <= EGO_X[1]) & (np.abs(p[:, 1]) <= EGO_Y)).sum() >= 50:
            return True
    return False


def boot_median(x, B=4000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    bs = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return np.median(x), *np.percentile(np.median(x[bs], 1), [2.5, 97.5]), len(x)


def main(windows, out, toks, dose_toks):
    W = {w["token"]: w for w in json.load(open(windows))}
    out = Path(out)
    betas, misses, fas, rows = [], 0, 0, []
    for tok in open(toks).read().split():
        w, d = W[tok], out / tok
        t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
        tr = np.array(w["lead_track_t0frame"])
        L = w["lead_LW"][0]
        x_rear = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 0]) - L / 2
        y_lead = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 1])
        Pl, El, Ml = cars_t0(d / "logged_s0_n25")
        Pc, Ec, Mc = cars_t0(d / "collide_s0_n25")
        s_l, s_c = Ml[:, 0, 3], Mc[:, 0, 3]
        hit = np.nonzero(x_rear - (s_c + FRONT) <= 0)[0]
        miss = not self_hit(Ec)
        fa = self_hit(El)
        misses += miss
        fas += fa
        b = np.nan
        if len(hit):
            k = hit[0]
            ds = s_c[k] - s_l[k]
            xl = lead_x(Pl[k], y_lead[k], s_l[k] + FRONT - 1.0)
            xc = lead_x(Pc[k], y_lead[k], s_c[k] + FRONT - 1.0)
            if ds >= 1.0 and np.isfinite(xl) and np.isfinite(xc):
                b = (xc - xl) / ds
        betas.append(b)
        rows.append((tok[:8], w["scene"], round(b, 2) if np.isfinite(b) else None, miss, fa))
    for r in rows:
        print("  ", r)
    m, lo, hi, n = boot_median(betas)
    print(f"FRESH OmniNWM lead beta: median {m:+.2f} [{lo:+.2f}, {hi:+.2f}] n {n} of {len(rows)} | registered 1.02 [0.98, 1.08] n 16")
    print(f"FRESH self-check: misses {misses}/{len(rows)} collide plans | false alarms {fas}/{len(rows)} logged plans "
          f"| registered 18/18 misses, 0 FA")
    # dose slope (d-2, d1, collide) on the moving-ego windows
    slopes, pull = [], []
    for tok in open(dose_toks).read().split():
        w, d = W[tok], out / tok
        if not (d / "d1_s0_n25" / "front3.npz").exists():
            continue
        t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
        tr = np.array(w["lead_track_t0frame"])
        x_rear = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 0]) - w["lead_LW"][0] / 2
        y_lead = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 1])
        Pl, _, Ml = cars_t0(d / "logged_s0_n25")
        pts = []
        for pl in ("d-2", "d1", "collide"):
            P, _, M = cars_t0(d / f"{pl}_s0_n25")
            hit = np.nonzero(x_rear - (M[:, 0, 3] + FRONT) <= 0)[0]
            k = min(int(np.argmin(np.abs(t - 1.5))), hit[0] if len(hit) else 99)
            ds = M[k, 0, 3] - Ml[k, 0, 3]
            xl = lead_x(Pl[k], y_lead[k], Ml[k, 0, 3] + FRONT - 1.0)
            xp = lead_x(P[k], y_lead[k], M[k, 0, 3] + FRONT - 1.0)
            if abs(ds) >= 0.3 and np.isfinite(xl) and np.isfinite(xp):
                pts.append((ds, xp - xl))
                if pl == "d-2":
                    pull.append(xp - xl < 0)
        if len(pts) >= 3:
            x, y = np.array(pts).T
            slopes.append(float((x * y).sum() / (x * x).sum()))
    m, lo, hi, n = boot_median(slopes)
    print(f"FRESH OmniNWM dose slope: median {m:+.2f} [{lo:+.2f}, {hi:+.2f}] n {n} | registered 0.94 [0.82, 1.07] n 8")
    print(f"FRESH braking pulls lead in: {np.mean(pull):.2f} of {len(pull)} | registered 8/8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
