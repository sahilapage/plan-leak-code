"""FRESH AUDIT (2026-10-19), run on the 4090 where the checkpoints / tables live.
(A) Object-model plan leak re-derived by re-running the frozen A31 checkpoints with a DIFFERENT counterfactual plan
    family and lead rule than the registered obj_wm.leak: plan = logged waypoints with x scaled by 1.3 (30 % more
    progress, any speed profile), lead = agent observed at t0-0.5 s and t0, 3-40 m ahead, |y| <= 1.5 m, heading within
    20 deg; beta(h) = (pred x under scaled plan - pred x under logged plan) / (ego x difference), |ego diff| >= 0.5 m.
    Models: chunk, step1, step0, gc_step1 (seed 0). Median over leads, scene-cluster bootstrap.
(B) Policy frontiers (A32 / A36): real crash at matched mean extra accel 3.0 by fresh linear interpolation per (model,
    policy seed), mean over policy seeds, then over WM seeds.
(C) Simulator (A35) lead beta at +2 s vs the TRUE beta, fresh aggregation of the per-agent table.
Usage (4090, vla env): python fresh_tables.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "anon/drive/src"))
O, SW = Path.home() / "anon/objwm", Path.home() / "anon/simwm"


def cluster_boot_median(v, cl, B=2000, seed=0):
    v, cl = np.asarray(v, float), np.asarray(cl)
    ok = np.isfinite(v)
    v, cl = v[ok], cl[ok]
    u = np.unique(cl)
    idx = {c: np.nonzero(cl == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    meds = [np.median(np.concatenate([v[idx[c]] for c in rng.choice(u, len(u))])) for _ in range(B)]
    return np.median(v), *np.percentile(meds, [2.5, 97.5]), len(v), len(u)


def part_a():
    import torch
    import gcomp  # noqa: F401  (gc_ wrapper)
    import obj_wm as o
    z = np.load(O / "val.npz")
    E0, A0, M0, S0, scn = z["E"], z["A"], z["M"], z["S"], z["scene"]
    P = o.P
    leads = []
    for i in range(len(E0)):
        for j in range(o.NA):
            if not (M0[i, j, P] and M0[i, j, P - 1]):
                continue
            x, y, yw = A0[i, j, P]
            if 3 <= x <= 40 and abs(y) <= 1.5 and abs((yw + np.pi) % (2 * np.pi) - np.pi) <= np.radians(20):
                leads.append((i, j))
    print(f"(A) {len(leads)} leads in {len(set(scn[i] for i, _ in leads))} scenes")
    by_win = {}
    for i, j in leads:
        by_win.setdefault(i, []).append(j)
    wins = np.array(sorted(by_win))
    for ck in ("chunk", "step1", "step0", "gc_step1"):
        net = o.load_net(str(O / f"{ck}.pt"))
        vals = {1: [], 2: []}
        cls = {1: [], 2: []}
        for b in range(0, len(wins), 512):
            ii = wins[b:b + 512]
            E, A, M, S = (torch.tensor(q[ii], device="cuda") for q in (E0, A0, M0, S0))
            wp = E[:, P + 1:, :2]
            wpc = wp.clone()
            wpc[..., 0] = wp[..., 0] * 1.3
            with torch.no_grad():
                base = o.rollout(net, E, A, M, S).cpu().numpy()
                cf = o.rollout(net, o.plan_E(E, wpc), A, M, S).cpu().numpy()
            dxe = (wpc - wp)[..., 0].cpu().numpy()
            for r, gi in enumerate(ii):
                for j in by_win[gi]:
                    for h in (1, 2):
                        k = 2 * h - 1
                        if abs(dxe[r, k]) >= 0.5:
                            vals[h].append((cf[r, j, k, 0] - base[r, j, k, 0]) / dxe[r, k])
                            cls[h].append(scn[gi])
        for h in (1, 2):
            m, lo, hi, n, nc = cluster_boot_median(vals[h], cls[h])
            print(f"  FRESH {ck:9s} lead beta +{h} s: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] (n {n}, {nc} scenes)")
    print("  registered (+1 s, plans +0.5 a t^2): chunk 0.68 | step1 0.74 | step0 0.00 | gc_step1 -0.02")


def part_b():
    for f, pairs in (("frontier.parquet", [("step1", "step0"), ("chunk", "step0")]),
                     ("a36_frontier_rerun.parquet", [("gc_step1", "step0"), ("step1", "gc_step1")])):
        d = pd.read_parquet(O / f)

        def at(model, a0=3.0):
            v = []
            for _, g in d[d.model == model].groupby("seed"):
                g = g.groupby("a").crash.mean().sort_index()
                if g.index.min() <= a0 <= g.index.max():
                    v.append(np.interp(a0, g.index.values, g.values))
            return np.mean(v) if v else np.nan
        print(f"(B) {f}: real crash at matched mean extra accel 3.0")
        for a, b in pairs:
            ma = np.mean([at(m) for m in d.model.unique() if m == a or m.startswith(a + "_s")])
            mb = np.mean([at(m) for m in d.model.unique() if m == b or m.startswith(b + "_s")])
            print(f"  FRESH {a} {ma:.3f} vs {b} {mb:.3f}: diff {ma - mb:+.3f}, ratio {ma / mb:.2f}")
    print("  registered: step1 - step0 +0.040 (2.8x mean over seeds); gc_step1 - step0 -0.000; step1 - gc_step1 +0.040")


def part_c():
    d = pd.read_parquet(SW / "a35_test_leak.parquet")
    d = d[(d.h == 2)]
    print("(C) A35 rel codes:", sorted(d.rel.unique())[:10])
    for m in ("chunk", "step0", "step1", "gc_chunk", "gc_step1"):
        g = d[(d.model == m) & (d.rel == 1)]
        if len(g):
            r = cluster_boot_median(g.beta, g.scene)
            print(f"  FRESH sim lead beta +2 s {m:9s} {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] (n {r[3]}) | true median "
                  f"{np.median(g.beta_true):+.2f}")
    print("  registered: chunk 1.59 | step0 -0.01 | step1 0.01 | gc_chunk 0.08 | gc_step1 0.00 | true 0.00")


if __name__ == "__main__":
    part_b()
    part_c()
    part_a()
