"""Depth on the plan-marginal fix (A30), from forecasts already computed (no new model runs).
  build  <het_dir> <orig_marg_dir> <rpt_marg_dir>   per-candidate score in each of the K prior-plan forecasts
         (mk_<m>_<k>), label-free leak slope per window, logged-plan accel, lead accel, physics detection error
         -> <het_dir>/table_deep.parquet
  ana    <het_dir> <het_dir>                        pooled analyses:
    A  back-door weights: p(a_k | h) fitted on the OTHER fold (logged accel ~ N(mu + b v0, sd)), weighted marginal;
       K = 1 / 3 / 5 / 7 subsets; worst-case (max over k) marginal.
    B  label-free Plan-Leak Index per window = -slope of a FIXED candidate box's score vs the conditioning plan's
       accel (no leak for non-reactive agents -> 0); does it predict the self-judge's under-scoring of true crashes?
    C  where physics breaks (lead |accel| >= 1 m/s^2, or detection CV error >= 2 m / missed): WM marginal vs physics.
Usage (server, vla env): python audit3/marg_deep.py build ... | ana het_f0 het_f1"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_acc import acc_plan  # noqa: E402
from het_marg import PRIOR, mplan  # noqa: E402
from sw_hetero import ego_max, plan_pose  # noqa: E402

K = len(PRIOR)


def per_k(S, i, t, a):
    plan = acc_plan(t["plans"]["logged"], a)
    return [max(ego_max(np.asarray(S[i, q, h - 1], np.float32), plan_pose(mplan(t["v0"], pa), h), plan, h)
                for h in (1, 2, 3)) for q, pa in enumerate(PRIOR)]


def lead_info(t):
    tr = t["track"]
    if tr is None:
        return np.nan, np.nan
    v = (tr[1, 0] - tr[0, 0]) / 0.5
    acc = 2 * (tr[6, 0] - tr[0, 0] - 3 * v) / 9
    best = None
    for d in t["dets"]:
        r = np.hypot(d["x"] - tr[0, 0], d["y"] - tr[0, 1])
        if r <= 3.0 and (best is None or r < best[0]):
            best = (r, d)
    err = 99.0 if best is None else abs(best[1]["x"] + 3 * best[1]["v"][0] - tr[6, 0])
    return float(acc), float(err)


def build(het, dorig, drpt):
    het = Path(het)
    T = pickle.load(open(het / "acc" / "truth.pkl", "rb"))
    df = pd.read_parquet(het / "table_marg.parquet")
    for m, d in (("orig", dorig), ("rpt", drpt)):
        S = np.load(Path(d) / "soft.npy", mmap_mode="r")
        ix = {t: i for i, t in enumerate(json.load(open(Path(d) / "soft_tokens.json")))}
        M = np.array([per_k(S, ix[tok], T[tok], a) for tok, a in zip(df.token, df.a)])
        for k in range(K):
            df[f"mk_{m}_{k}"] = M[:, k]
    li = {tok: lead_info(T[tok]) for tok in df.token.unique()}
    df["lead_acc"] = df.token.map(lambda x: li[x][0])
    df["det_err"] = df.token.map(lambda x: li[x][1])
    df["a_log"] = df.token.map(lambda x: 2 * (T[x]["plans"]["logged"][-1][0] - 3 * T[x]["v0"]) / 9)
    df["v0"] = df.token.map(lambda x: T[x]["v0"])
    df["fold"] = het.name
    df.to_parquet(het / "table_deep.parquet")
    print(f"{het}: {len(df)} rows, {df.token.nunique()} windows")


def boot(df, js, pairs, n=300, seed=0):
    from sw_hetero import gain
    sc = df.scene.unique()
    g = {s: df[df.scene == s] for s in sc}
    rng = np.random.default_rng(seed)
    G = {j: [] for j in js}
    for _ in range(n):
        parts = []
        for k, s in enumerate(rng.choice(sc, len(sc))):
            p = g[s].copy()
            p["token"] = p.token + f"#{k}"
            parts.append(p)
        r = pd.concat(parts)
        for j in js:
            G[j].append(gain(r, j))
    G = {j: np.array(v) for j, v in G.items()}
    print(f"    {df.token.nunique()} windows, {len(sc)} scenes, {int(df.y.sum())} crash candidates")
    for j in js:
        print(f"    {j:16s} gain {gain(df, j):+.3f} [{np.nanquantile(G[j], .025):+.3f}, {np.nanquantile(G[j], .975):+.3f}]")
    for a, b in pairs:
        dd = G[a] - G[b]
        print(f"    paired {a} - {b}: {gain(df, a) - gain(df, b):+.3f} [{np.nanquantile(dd, .025):+.3f}, "
              f"{np.nanquantile(dd, .975):+.3f}]", flush=True)


def ana(dirs):
    from scipy.stats import spearmanr
    D = [pd.read_parquet(Path(d) / "table_deep.parquet") for d in dirs]
    for X in D:
        X.rename(columns={"self_rptA": "self_rpt", "cvref_rptA": "cvref_rpt"}, inplace=True)
    pa = np.array(PRIOR)
    # A: back-door weights fitted on the other fold
    for i, X in enumerate(D):
        W = D[1 - i].drop_duplicates("token")
        b, mu = np.polyfit(W.v0, W.a_log, 1)
        sd = float(np.std(W.a_log - (mu + b * W.v0)))
        print(f"A  prior from other fold: a_log ~ N({mu:+.2f} {b:+.3f} v0, {sd:.2f})")
        for m in ("orig", "rpt"):
            mk = X[[f"mk_{m}_{k}" for k in range(K)]].to_numpy()
            w = np.exp(-0.5 * ((pa[None, :] - (mu + b * X.v0.to_numpy()[:, None])) / sd) ** 2)
            X[f"wmarg_{m}"] = (mk * w).sum(1) / w.sum(1)
            X[f"maxmarg_{m}"] = mk.max(1)
            for name, ks in (("k1", [3]), ("k3", [1, 3, 5]), ("k5", [0, 1, 3, 5, 6]), ("k7", list(range(K)))):
                X[f"{name}_{m}"] = mk[:, ks].mean(1)
    df = pd.concat(D, ignore_index=True)
    df["const_a"] = df.a
    print("A  back-door weighting / K / worst-case (pooled held-out)")
    boot(df, ["k1_orig", "k3_orig", "k5_orig", "k7_orig", "wmarg_orig", "maxmarg_orig", "wmarg_rpt", "physics"],
         [("wmarg_orig", "k7_orig"), ("k7_orig", "k1_orig"), ("maxmarg_orig", "k7_orig"), ("physics", "wmarg_orig")])
    # B: label-free Plan-Leak Index (per window; uses only forecasts + candidate geometry)
    print("B  label-free Plan-Leak Index (PLI = -slope of a fixed box's score vs conditioning accel)")
    rows = []
    for m in ("orig", "rpt"):
        mk = df[[f"mk_{m}_{k}" for k in range(K)]].to_numpy()
        c = pa - pa.mean()
        df[f"pli_{m}"] = -(mk - mk.mean(1, keepdims=True)) @ c / (c @ c)
        w = df.groupby("token").agg(scene=("scene", "first"), pli=(f"pli_{m}", "mean"))
        cr = df[df.y == 1]
        under = (cr[f"cvmarg_{m}"] - cr[f"self_{m}"]).groupby(cr.token).mean()
        w = w.join(under.rename("under"), how="left")
        rows.append((m, w))
        ww = w.dropna()
        rho = spearmanr(ww.pli, ww.under)[0]
        rng = np.random.default_rng(0)
        sc = ww.scene.unique()
        bs = []
        for _ in range(1000):
            s = rng.choice(sc, len(sc))
            q = pd.concat([ww[ww.scene == x] for x in s])
            bs.append(spearmanr(q.pli, q.under)[0])
        print(f"   {m}: mean PLI {w.pli.mean():+.4f} (median {w.pli.median():+.4f}, share > 0 {(w.pli > 0).mean():.2f}) | "
              f"Spearman(PLI, self under-scoring of true crashes) {rho:+.2f} [{np.quantile(bs, .025):+.2f}, "
              f"{np.quantile(bs, .975):+.2f}] over {len(ww)} windows with a crash candidate")
    # C: where physics breaks
    mv = df[df.stratum != "free"]
    for name, sel in (("lead |accel| >= 1", mv.lead_acc.abs() >= 1.0), ("lead |accel| < 1", mv.lead_acc.abs() < 1.0),
                      ("lead det error >= 2 m or missed", mv.det_err >= 2.0), ("lead det error < 2 m", mv.det_err < 2.0)):
        sub = mv[sel.values]
        print(f"C  [{name}]")
        boot(sub, ["self_orig", "cvmarg_orig", "wmarg_orig", "physics", "h_gap0"],
             [("cvmarg_orig", "physics"), ("wmarg_orig", "physics"), ("cvmarg_orig", "self_orig")], n=200)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:5])
    else:
        ana(sys.argv[2:])
