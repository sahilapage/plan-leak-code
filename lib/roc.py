"""Threshold-free crash discrimination of SWTC variants on real-car windows: per window, danger(plan) = max over +1 / +2 /
+3 s of the obstacle probability inside the plan's own future ego box (the A25 veto score before the tau = 0.2 cut).
collide = crash (positive), logged = no crash (negative). Reports AUROC (pooled), the paired win rate
P(danger(collide) > danger(logged)), recall at false-alarm 5 % / 10 % (threshold = quantile of the logged scores),
scene-cluster bootstrap B = 2000, seed 0. Reference judges: cvref*=<src_dir>,<cv_dir> (past-only CV forecast, A26) and oracle*=<src_dir> (logged-plan forecast).
Usage: python audit3/roc.py name=<dir> [name=<dir> ...]
       python audit3/roc.py --diff name=<dir> name=<dir>      (paired AUROC difference)"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sw_realhazard import EGO_C, EGO_L, EGO_W, PLANS, box_in_frame, plan_pose  # noqa: E402


def scores(d):
    d = Path(d)
    truth = pickle.load(open(d / "truth.pkl", "rb"))
    soft = np.load(d / "soft.npy", mmap_mode="r")
    rows = []
    for i, tok in enumerate(json.load(open(d / "soft_tokens.json"))):
        t = truth[tok]
        r = {"token": tok, "scene": t["scene"]}
        for p_i, plan in enumerate(PLANS):
            if plan not in ("logged", "collide"):
                continue
            best = 0.0
            for h in (1, 2, 3):
                fx, fy, fyaw = plan_pose(t["plans"][plan], h)
                ob = box_in_frame(fx, fy, fyaw, fx + EGO_C * math.cos(fyaw), fy + EGO_C * math.sin(fyaw), fyaw,
                                  EGO_L, EGO_W)
                best = max(best, float(np.asarray(soft[i, p_i, h - 1], np.float32)[ob].max()))
            r[plan] = best
        rows.append(r)
    return pd.DataFrame(rows)


def scores_ref(src, cvdir=None, ref="cv"):
    """reference judges: every candidate's ego box scored inside ONE forecast, imagined under the past-only CV plan
    (ref = cv, src/sw_cvjudge.py outputs in cvdir) or under the logged plan (ref = logged: privileged oracle)."""
    src = Path(src)
    own = np.load(src / "soft.npy", mmap_mode="r")
    oi = {t: i for i, t in enumerate(json.load(open(src / "soft_tokens.json")))}
    if ref == "cv":
        cvdir = Path(cvdir)
        T = pickle.load(open(cvdir / "truth.pkl", "rb"))
        cvs = np.load(cvdir / "soft.npy", mmap_mode="r")
        toks = json.load(open(cvdir / "soft_tokens.json"))
    else:
        T = pickle.load(open(src / "truth.pkl", "rb"))
        toks = list(oi)
    rows = []
    for i, tok in enumerate(toks):
        t = T[tok]
        r = {"token": tok, "scene": t["scene"]}
        for cand in ("collide", "logged"):
            best = 0.0
            for h in (1, 2, 3):
                if ref == "cv":
                    m, (fx, fy, fyaw) = np.asarray(cvs[i, 0, h - 1], np.float32), plan_pose(t["cv"], h)
                else:
                    m = np.asarray(own[oi[tok], PLANS.index("logged"), h - 1], np.float32)
                    fx, fy, fyaw = plan_pose(t["plans"]["logged"], h)
                cx, cy, cyaw = plan_pose(t["plans"][cand], h)
                ob = box_in_frame(fx, fy, fyaw, cx + EGO_C * math.cos(cyaw), cy + EGO_C * math.sin(cyaw), cyaw,
                                  EGO_L, EGO_W)
                if ob.any():
                    best = max(best, float(m[ob].max()))
            r[cand] = best
        rows.append(r)
    return pd.DataFrame(rows)


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def metrics(f):
    out = {"auroc": auroc(f.collide, f.logged), "win": float((f.collide > f.logged).mean() + 0.5 * (f.collide == f.logged).mean())}
    for fa in (0.05, 0.10):
        th = np.quantile(f.logged, 1 - fa)
        out[f"rec@{int(fa * 100)}"] = float((f.collide > th).mean())
    return out


def load(a):
    name, d = a.split("=", 1)
    if name.startswith("cvref"):
        return name, scores_ref(*d.split(","), ref="cv")
    if name.startswith("oracle"):
        return name, scores_ref(d, ref="logged")
    return name, scores(d)


def diff(a, b, B=2000):
    """paired scene-cluster bootstrap of AUROC(a) - AUROC(b) on the same windows."""
    (na, fa), (nb, fb) = load(a), load(b)
    m = fa.merge(fb, on=["token", "scene"], suffixes=("_a", "_b"))
    sc = m.scene.unique()
    g = {s: m[m.scene == s] for s in sc}
    rng = np.random.default_rng(0)

    def d(f):
        return auroc(f.collide_a, f.logged_a) - auroc(f.collide_b, f.logged_b)
    reps = [d(pd.concat([g[s] for s in rng.choice(sc, len(sc))])) for _ in range(B)]
    q = np.quantile(reps, [0.025, 0.975])
    print(f"AUROC {na} - {nb}: {d(m):+.3f} [{q[0]:+.3f}, {q[1]:+.3f}] (n {len(m)}, {len(sc)} scenes)", flush=True)


def main(args):
    if args[0] == "--diff":
        diff(args[1], args[2])
        return
    rng = np.random.default_rng(0)
    for a in args:
        name, d = a.split("=", 1)
        if name.startswith("cvref"):
            f = scores_ref(*d.split(","), ref="cv")
        elif name.startswith("oracle"):
            f = scores_ref(d, ref="logged")
        else:
            f = scores(d)
        m = metrics(f)
        sc = f.scene.unique()
        g = {s: f[f.scene == s] for s in sc}
        reps = [metrics(pd.concat([g[s] for s in rng.choice(sc, len(sc))])) for _ in range(2000)]
        txt = " | ".join(f"{k} {m[k]:.3f} [{np.quantile([r[k] for r in reps], 0.025):.3f}, "
                         f"{np.quantile([r[k] for r in reps], 0.975):.3f}]" for k in m)
        print(f"{name:10s} n {len(f)} ({len(sc)} scenes) | {txt}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
