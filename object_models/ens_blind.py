"""A36 P4: is the plan leak visible to the standard model-based-RL safeguard (ensemble disagreement, MOPO / MOReL)?
Every ensemble member learns the same observational regression, so members should AGREE where they leak. On the A32
A29 grid (grid_a29_seeds.parquet: 3 WM seeds per family, 653 windows x 33 extra accelerations), among the plans the
ensemble mean calls safe (mean predicted penetration < 1 m), how well does each label-free score separate real crashes
(truth >= 1 m, the hacked plans) from truly safe plans? Scores: ensemble std of predicted penetration (MOPO-style);
self vs plan-marginal discrepancy (mean pm_<fam> - mean <fam> depth). AUROC with scene-bootstrap 95 % CI.
Usage (server, vla env): python audit3/ens_blind.py <work>/objwm/grid_a29_seeds.parquet"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auroc(s, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main(path, B=1000):
    df = pd.read_parquet(path)
    rng = np.random.default_rng(0)
    for fam in ("step1", "chunk"):
        cols = [fam, f"{fam}_s1", f"{fam}_s2"]
        pms = [f"pm_{c}" for c in cols]
        m = df[cols].mean(1)
        d = df.assign(mean=m, std=df[cols].std(1), disc=df[pms].mean(1) - m, y=df.truth >= 1.0)
        V = d[d["mean"] < 1.0]
        sc = V.scene.unique()
        byS = {s: V[V.scene == s] for s in sc}
        res = {}
        for k in ("std", "disc"):
            pt = auroc(V[k].to_numpy(), V.y.to_numpy())
            bs = []
            for _ in range(B):
                b = pd.concat([byS[s] for s in rng.choice(sc, len(sc))])
                bs.append(auroc(b[k].to_numpy(), b.y.to_numpy()))
            res[k] = (pt, np.nanquantile(bs, .025), np.nanquantile(bs, .975), np.array(bs))
        diff = res["disc"][3] - res["std"][3]
        print(f"{fam}: ensemble-safe points {len(V)} ({V.scene.nunique()} scenes), real crashes among them "
              f"{int(V.y.sum())} ({V.y.mean():.3f})")
        for k, lab in (("std", "ensemble std (MOPO)"), ("disc", "self vs plan-marginal")):
            print(f"  AUROC {lab:22s} {res[k][0]:.3f} [{res[k][1]:.3f}, {res[k][2]:.3f}]")
        print(f"  AUROC disc - std {res['disc'][0] - res['std'][0]:+.3f} [{np.nanquantile(diff, .025):+.3f}, "
              f"{np.nanquantile(diff, .975):+.3f}]")
        print(f"  median std at hacked {V[V.y]['std'].median():.3f} m vs truly safe {V[~V.y]['std'].median():.3f} m")


if __name__ == "__main__":
    main(sys.argv[1])
