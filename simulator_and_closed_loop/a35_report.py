"""A35 scoring (registered 8679e1f): leak beta vs TRUE beta per relation, judge errors under reactive truth, belief.
Usage (server, vla env): python audit3/a35_report.py <work>/simwm/a35_test"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REL = {1: "lead", 2: "follower", 3: "adjacent", 4: "parked"}


def boot_med(g, col, B=500, rng=np.random.default_rng(0)):
    sc = g.scene.unique()
    by = {s: g.loc[g.scene == s, col].to_numpy() for s in sc}
    bs = [np.median(np.concatenate([by[s] for s in rng.choice(sc, len(sc))])) for _ in range(B)]
    return np.median(g[col]), np.quantile(bs, .025), np.quantile(bs, .975)


def main(pre):
    L = pd.read_parquet(f"{pre}_leak.parquet")
    print("== leak beta (median, episode bootstrap 95 % CI) vs TRUE beta")
    for h in (1, 2, 3):
        for r, name in REL.items():
            g = L[(L.rel == r) & (L.h == h)]
            if g.empty:
                continue
            tr = g[g.model == g.model.iloc[0]]
            t = boot_med(tr, "beta_true")
            line = [f"TRUE {t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"]
            for m, gm in g.groupby("model", sort=False):
                b = boot_med(gm, "beta")
                line.append(f"{m} {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]")
            print(f"  {name:9s} +{h} s (n {len(tr)}, {tr.scene.nunique()} ep): " + " | ".join(line))
    J = pd.read_parquet(f"{pre}_judge.parquet")
    models = [c[6:] for c in J.columns if c.startswith("front_") and c != "front_true"]
    rng = np.random.default_rng(1)
    sc = J.scene.unique()
    by = {s: J[J.scene == s] for s in sc}
    boots = [pd.concat([by[s] for s in rng.choice(sc, len(sc))]) for _ in range(500)]

    def err(d, side, m, sel):
        x = d[sel(d)]
        p, t = x[f"{side}_{m}"] >= 1.0, x[f"{side}_true"] >= 1.0
        return (p & ~t).mean(), (~p & t).mean(), (p != t).mean()
    for side, sel, lab in (("rear", lambda d: d.a <= -2, "rear (follower into ego), braking a <= -2"),
                           ("front", lambda d: d.a >= 1, "front (ego into lead), accelerating a >= 1")):
        x = J[sel(J)]
        print(f"== judge {lab}: {len(x)} candidates, true crash rate {(x[f'{side}_true'] >= 1).mean():.3f}")
        for m in models:
            fa, mi, e = err(J, side, m, sel)
            bs = [err(b, side, m, sel)[2] for b in boots]
            print(f"  {m:10s} false alarm {fa:.3f} | miss {mi:.3f} | error {e:.3f} [{np.quantile(bs, .025):.3f}, "
                  f"{np.quantile(bs, .975):.3f}]")
        for a_, b_ in (("pm_chunk", "gc_chunk"), ("pm_step1", "gc_step1"), ("chunk", "gc_chunk"), ("step1", "gc_step1"),
                       ("pm_chunk", "step0"), ("gc_chunk", "step0")):
            if a_ in models and b_ in models:
                d = [err(b, side, a_, sel)[2] - err(b, side, b_, sel)[2] for b in boots]
                print(f"  error {a_} - {b_} {err(J, side, a_, sel)[2] - err(J, side, b_, sel)[2]:+.3f} "
                      f"[{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]")
    bp = Path(f"{pre}_belief.parquet")
    if bp.exists():
        Bf = pd.read_parquet(bp)
        V = Bf[(Bf.a_true > 0) & (Bf.a_true < 8)]
        print(f"== belief: {len(V)} lead windows with a finite true limit ({V.scene.nunique()} ep)")
        for c in [c for c in V.columns if c.startswith("a_") and c != "a_true"]:
            r = V[c] / V.a_true
            V2 = V.assign(r=r)
            b = boot_med(V2, "r")
            print(f"  {c[2:]:10s} median ratio {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] | share >= 1.5x {(r >= 1.5).mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
