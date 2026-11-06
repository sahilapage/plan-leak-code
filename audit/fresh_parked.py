"""FRESH AUDIT: OmniNWM selectivity (parked cars) from raw generated depth, independent of src/a37_control2.py (which
used a 2D detector + PnP). Parked = annotated vehicle that moves < 1 m over its track; its t0-frame position from the
annotation; generated position = median x of car points within +-3 m (x) / +-1.2 m (y) of it; beta_parked =
(x_collide - x_logged) / ds at the same t* as the lead (collide plan's true contact). Truth 0.
Usage: python fresh_parked.py <windows.json> <out_omni> <main18.txt>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fresh_omninwm import cars_t0, FRONT  # noqa

def qR(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

W = {w["token"]: w for w in json.load(open(sys.argv[1]))}
out = Path(sys.argv[2]); vals, wins = [], []
for tok in open(sys.argv[3]).read().split():
    w, d = W[tok], out / tok
    fr0 = w["frames"][0]; R0 = qR(fr0["ego_q"]); T0 = np.asarray(fr0["ego_t"])
    t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
    tr = np.array(w["lead_track_t0frame"]); x_rear = np.interp(t, np.arange(len(tr)) * .5, tr[:, 0]) - w["lead_LW"][0] / 2
    Pl, _, Ml = cars_t0(d / "logged_s0_n25"); Pc, _, Mc = cars_t0(d / "collide_s0_n25")
    hit = np.nonzero(x_rear - (Mc[:, 0, 3] + FRONT) <= 0)[0]
    if not len(hit): continue
    k = hit[0]; ds = Mc[k, 0, 3] - Ml[k, 0, 3]
    if ds < 1.0: continue
    for v in w["vehicles"]:
        trk = v["track"]
        if len(trk) < 5 or np.linalg.norm(np.asarray(trk[-1]["t"][:2]) - np.asarray(trk[0]["t"][:2])) >= 1.0: continue
        p = (np.asarray(trk[0]["t"]) - T0) @ R0
        if not (3 <= p[0] <= 35 and 2.0 <= abs(p[1]) <= 8): continue      # parked beside the road, ahead
        def gx(P):
            s = P[(np.abs(P[:, 0] - p[0]) <= 3) & (np.abs(P[:, 1] - p[1]) <= 1.2)] if len(P) else P
            return np.median(s[:, 0]) if len(s) >= 30 else np.nan
        xl, xc = gx(Pl[k]), gx(Pc[k])
        if np.isfinite(xl) and np.isfinite(xc):
            vals.append((xc - xl) / ds); wins.append(tok)
v = np.array(vals); bs = np.random.default_rng(0).integers(0, len(v), (4000, len(v)))
print(f"FRESH OmniNWM parked-car beta: median {np.median(v):+.2f} [{np.percentile(np.median(v[bs],1),2.5):+.2f}, "
      f"{np.percentile(np.median(v[bs],1),97.5):+.2f}] n {len(v)} vehicles / {len(set(wins))} windows | registered 0.01 "
      f"[-0.36, 1.14] (18 vehicles, detector-based); lead (fresh) 1.00")
