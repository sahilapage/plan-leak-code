"""A37b scoring (registered before any dose generation): dose-response of the plan leak in OmniNWM.
For each window and plan P in {d-2, d1, d3, collide} (dA = logged + constant extra acceleration A; d0 = logged), the
generated lead rear (t0 frame, src/a37_score.py track_lead) is compared with the logged plan's at t_m = min(1.5 s,
first true contact of P): d_rear = rear_P - rear_logged, d_s = s_P - s_logged. Per window, the leak slope = least-
squares slope of d_rear on d_s through the origin over the plans with |d_s| >= 0.3 m (>= 3 plans). Truth slope: 0
(the lead does not react to the car behind it; stopped leads exactly 0).
Prints per-window slopes, the median with a scene-cluster bootstrap CI, the share of windows whose braking plan (d-2)
pulls the generated lead TOWARDS the ego (d_rear < 0), and the pooled R^2 of the through-origin fit.
Usage: python src/a37b_dose.py <windows.json> <out_dir> <suffix e.g. _s0_n25> <tokens.txt>"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_score as S  # noqa: E402

PLANS = ("d-2", "d1", "d3", "collide")
T_M = 1.5


def main(windows, out, suf, toks):
    W = {w["token"]: w for w in json.load(open(windows))}
    out = Path(out)
    rows, P2 = [], []
    for tok in open(toks).read().split():
        w = W[tok]
        d = out / tok
        if not (d / f"logged{suf}" / "front3.npz").exists():
            continue
        t = np.array(json.load(open(d / f"logged{suf}" / "reward.json"))["t"])
        tr = np.array(w["lead_track_t0frame"])
        x_true = np.interp(t, np.arange(len(tr)) * .5, tr[:, 0]) - w["lead_LW"][0] / 2

        def get(pl):
            z = np.load(d / f"{pl}{suf}" / "front3.npz")
            M = np.load(d / f"{pl}{suf}" / "rel_poses.npy")
            return M, S.track_lead(S.backproject(z["depth"], z["seg"], z["cam_k"], z["cam2ego"], S.DYN), M, w)
        Ml, rl = get("logged")
        pts = []
        for pl in PLANS:
            if not (d / f"{pl}{suf}" / "front3.npz").exists():
                continue
            M, r = get(pl)
            hit = np.nonzero(x_true - (M[:, 0, 3] + S.FRONT0) <= 0)[0]
            k = min(int(np.argmin(np.abs(t - T_M))), hit[0] if len(hit) else 99)
            ds, dr = M[k, 0, 3] - Ml[k, 0, 3], r[k] - rl[k]
            if np.isfinite(dr) and abs(ds) >= 0.3:
                pts.append((pl, ds, dr))
        brk = next((p for p in pts if p[0] == "d-2"), None)
        if brk is not None:                                       # P2 needs only the braking plan (final-plan deviation)
            P2.append({"token": tok, "ds": brk[1], "dr": brk[2], "beta_brake": brk[2] / brk[1]})
        if len(pts) >= 3:
            x = np.array([p[1] for p in pts])
            y = np.array([p[2] for p in pts])
            slope = float((x * y).sum() / (x * x).sum())
            rows.append({"token": tok, "scene": w["scene"], "slope": slope, "pts": pts,
                         "brake_towards": next((p[2] < 0 for p in pts if p[0] == "d-2"), None)})
    if P2:
        bb = np.array([q["beta_brake"] for q in P2])
        print(f"P2 (braking arm, all windows with a measurable d-2 point): pulls the lead towards the ego in "
              f"{np.mean([q['dr'] < 0 for q in P2]):.2f} of {len(P2)} windows; braking-plan beta median {np.median(bb):+.2f}")
        for q in P2:
            print(f"    {q['token'][:8]} ds {q['ds']:+.2f} dr {q['dr']:+.2f} beta {q['beta_brake']:+.2f}")
    print(f"windows with >= 3 dose points: {len(rows)}")
    for r in rows:
        print(f"  {r['scene']} {r['token'][:8]} slope {r['slope']:+.2f} | " +
              " ".join(f"{p[0]}: ds {p[1]:+.1f} dr {p[2]:+.1f}" for p in r["pts"]))
    if not rows:
        return
    sl = np.array([r["slope"] for r in rows])
    sc = np.array([r["scene"] for r in rows])
    rng = np.random.default_rng(0)
    keys = np.unique(sc)
    bs = [np.median(np.concatenate([sl[sc == k] for k in rng.choice(keys, len(keys))])) for _ in range(2000)]
    print(f"median leak slope {np.median(sl):+.2f} [{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}] (truth 0)")
    bt = [r["brake_towards"] for r in rows if r["brake_towards"] is not None]
    if bt:
        print(f"braking plan pulls the generated lead towards the ego: {np.mean(bt):.2f} of {len(bt)} windows")
    X = np.concatenate([[p[1] for p in r["pts"]] for r in rows])
    Y = np.concatenate([[p[2] for p in r["pts"]] for r in rows])
    b = (X * Y).sum() / (X * X).sum()
    print(f"pooled through-origin slope {b:+.2f}, R^2 {1 - ((Y - b * X) ** 2).sum() / (Y ** 2).sum():.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
