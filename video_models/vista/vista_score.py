"""Vista leak scoring (A37 P4, A37g P1, A37h P1/P2, A37b Vista dose; written before any Vista beta was computed).
Reads <out>/read.parquet from src/vista_leak.py read (detector box-height distance d_est per frame) and the plans.
Lead rear in the t0 frame (up to a constant camera offset that cancels between plans): x(t) = s_plan(t) + d_est(t),
s_plan = the plan's forward progress (src/vista_leak.py plan_wps; h- / s- plans use the clip's own t2 from meta.json).
beta(A vs B) = (x_A - x_B) / (s_A - s_B) at k* = the last frame before the TRUE contact of plan A (true gap = annotated
lead rear - (s_A + FRONT0) <= 0; frame 24 if no contact within the 2 s clip), stepping back to the last frame <= k*
where the lead is tracked in both clips; dropped if s_A - s_B < 1 m there. beta 0 = the lead stays where the world puts
it, 1 = it keeps its gap to the ego. Readout validity: REAL keyframes, median |d_est - d_true| at 1 s and 2 s (P4 needs
<= 3 m). Window bootstrap (2000).
  P4   (collide vs logged)     median beta >= 0.30 (if the readout is valid)
  A37g precondition: median beta(s-collide vs logged) >= 0.5;  P1: median beta(h) <= median beta(s) - 0.2 and the paired
       bootstrap upper bound of median(beta_h - beta_s) < 0
  A37h P1: median beta(g-collide vs g-logged) <= 0.25 and <= median beta(h) - 0.2;  P2: share of windows whose generated
       lead camera distance at the true contact frame is <= 4 m: g-collide >= s-collide + 0.3
Usage: python src/vista_score.py <windows.json> <out_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vista_leak import FR_T, interp, plan_wps  # noqa: E402

FRONT0 = 1.3 + 4.1 / 2
EXCL = {"ad26ae1c24e14afd9bc1e0bcc4e378b7"}      # A37h code-check pilot (prereg: excluded from A37g / A37h)                         # ego front ahead of the reference point (src/a37_score.py BOX)


def lead_series(R, tok, clip_plan):
    g = R[(R.token == tok) & (R.plan == clip_plan) & (R.kind == "lead") & (R.src == "gen")]
    if g.empty:
        return None
    g = g.groupby("t").d_est.median()
    return np.array([g.get(t, np.nan) for t in FR_T])


def s_plan(w, out, tok, plan):
    t2 = None
    meta = Path(out) / tok / f"{plan}_s0" / "meta.json"
    if meta.exists():
        t2 = json.load(open(meta)).get("t2")
    return interp(plan_wps(w, plan, t2), FR_T)[:, 0]


def beta(w, R, out, tok, a, b):
    da, db = lead_series(R, tok, a), lead_series(R, tok, b)
    if da is None or db is None:
        return None
    sa, sb = s_plan(w, out, tok, a), s_plan(w, out, tok, b)
    tr = np.array(w["lead_track_t0frame"])
    x_true = np.interp(FR_T, np.arange(len(tr)) * 0.5, tr[:, 0]) - w["lead_LW"][0] / 2
    hit = np.nonzero(x_true - (sa + FRONT0) <= 0)[0]
    k = (hit[0] - 1) if len(hit) else len(FR_T) - 1
    while k >= 0 and not (np.isfinite(da[k]) and np.isfinite(db[k])):
        k -= 1
    if k < 0 or sa[k] - sb[k] < 1.0:
        return None
    kc = hit[0] if len(hit) else None
    return {"beta": ((sa[k] + da[k]) - (sb[k] + db[k])) / (sa[k] - sb[k]), "t_eval": FR_T[k], "ds": sa[k] - sb[k],
            "d_at_contact": da[kc] if kc is not None and kc < len(da) else np.nan, "contact_in_clip": kc is not None}


def boot_median(x, B=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return np.median(x), *np.percentile(np.median(x[bs], 1), [2.5, 97.5])


def main(windows, out):
    W = {w["token"]: w for w in json.load(open(windows))}
    R = pd.read_csv(Path(out) / "read.csv") if (Path(out) / "read.csv").exists() else pd.read_parquet(Path(out) / "read.parquet")
    real = R[R.src == "real"].copy()
    real["err"] = (real.d_est - real.d_true).abs()
    print("readout validity (REAL keyframes, lead + static): median |d_est - d_true| "
          + " | ".join(f"{t:.0f} s {real[real.t == t].err.median():.2f} m (n {int((real.t == t).sum())})" for t in (1.0, 2.0)))
    pairs = {"P4": ("collide", "logged"), "s": ("s-collide", "logged"), "h": ("h-collide", "h-logged"),
             "g": ("g-collide", "g-logged"), "d1": ("d1", "logged"), "d3": ("d3", "logged"), "d-2": ("d-2", "logged")}
    rows = []
    for tok in sorted(set(R.token)):
        if tok not in W:
            continue
        for name, (a, b) in pairs.items():
            if name in ("s", "h", "g") and tok in EXCL:                 # A37g / A37h exclude the code-check pilot window
                continue
            r = beta(W[tok], R, out, tok, a, b)
            if r is not None:
                rows.append({"token": tok, "pair": name, **r})
    D = pd.DataFrame(rows)
    D.to_csv(Path(out) / "vista_beta.csv", index=False)
    for name in pairs:
        d = D[D.pair == name]
        if len(d):
            m, lo, hi = boot_median(d.beta)
            print(f"{name:4s} {pairs[name][0]:>10s} vs {pairs[name][1]:9s} median beta {m:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"n {len(d)} | contact in clip {d.contact_in_clip.mean():.2f}")
    P = D.pivot(index="token", columns="pair", values="beta")
    if {"h", "s"} <= set(P.columns):
        x = (P.h - P.s).dropna()
        m, lo, hi = boot_median(x)
        print(f"A37g: median beta(h) - beta(s) paired {m:+.2f} [{lo:+.2f}, {hi:+.2f}] n {len(x)}")
    if {"g", "s"} <= set(P.columns):
        C = D.pivot(index="token", columns="pair", values="d_at_contact")
        for c in ("g", "s"):
            if c in C:
                v = C[c].dropna()
                print(f"A37h P2: share lead camera distance <= 4 m at true contact, {c}: {(v <= 4).mean():.2f} (n {len(v)})")


def dose(windows, out, B=2000):
    """A37b Vista arm (src/a37b_dose.py rule on the P4 readout): per window and plan P in {d-2, d1, d3, collide},
    d_rear = (s_P + d_P) - (s_logged + d_logged), d_s = s_P - s_logged at t_m = min(1.5 s, first true contact of P),
    stepping back to the last frame tracked in both; slope through the origin over plans with |d_s| >= 0.3 m (>= 3
    plans). P1: median slope >= 0.3 with bootstrap lower bound > 0; P2: d-2 pulls the lead towards the ego (d_rear < 0)
    in >= 60 % of windows."""
    W = {w["token"]: w for w in json.load(open(windows))}
    R = pd.read_csv(Path(out) / "read.csv") if (Path(out) / "read.csv").exists() else pd.read_parquet(Path(out) / "read.parquet")
    slopes, pull = [], []
    for tok in sorted(set(R.token)):
        if tok not in W:
            continue
        w = W[tok]
        dl = lead_series(R, tok, "logged")
        if dl is None:
            continue
        sl = s_plan(w, out, tok, "logged")
        tr = np.array(w["lead_track_t0frame"])
        x_true = np.interp(FR_T, np.arange(len(tr)) * 0.5, tr[:, 0]) - w["lead_LW"][0] / 2
        pts = []
        for pl in ("d-2", "d1", "d3", "collide"):
            dp = lead_series(R, tok, pl)
            if dp is None:
                continue
            sp = s_plan(w, out, tok, pl)
            hit = np.nonzero(x_true - (sp + FRONT0) <= 0)[0]
            k = int(np.argmin(np.abs(FR_T - 1.5)))
            if len(hit):
                k = min(k, hit[0])
            while k >= 0 and not (np.isfinite(dp[k]) and np.isfinite(dl[k])):
                k -= 1
            if k < 0:
                continue
            ds, dr = sp[k] - sl[k], (sp[k] + dp[k]) - (sl[k] + dl[k])
            if abs(ds) >= 0.3:
                pts.append((ds, dr))
                if pl == "d-2":
                    pull.append(dr < 0)
        if len(pts) >= 3:
            x, y = np.array(pts).T
            slopes.append(float((x * y).sum() / (x * x).sum()))
    m, lo, hi = boot_median(slopes, B)
    print(f"A37b Vista: median slope {m:+.2f} [{lo:+.2f}, {hi:+.2f}] n {len(slopes)} | braking pulls the lead towards "
          f"the ego in {np.mean(pull):.2f} of {len(pull)} windows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    dose(sys.argv[1], sys.argv[2])
