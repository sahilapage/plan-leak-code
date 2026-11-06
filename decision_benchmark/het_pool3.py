import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/audit3")
import numpy as np, pandas as pd
from sw_hetero import gain
from planner_select import at_crash, curve
from fixed_acc import strat
A = pd.read_parquet("<work>/het_f0/table_marg.parquet")
B = pd.read_parquet("<work>/het_f1/table_marg.parquet").rename(columns={"self_rptA": "self_rpt", "cvref_rptA": "cvref_rpt"})
cols = ["token", "scene", "a", "y", "const_a", "self_orig", "self_rpt", "oracle", "cvref", "cvref_rpt", "physics", "h_gap0", "cvmarg_orig", "cvmarg_rpt"]
df = pd.concat([A[cols], B[cols]], ignore_index=True)
js = ["self_orig", "self_rpt", "cvref", "cvref_rpt", "oracle", "physics", "h_gap0", "cvmarg_orig", "cvmarg_rpt"]
sc = df.scene.unique(); g = {s: df[df.scene == s] for s in sc}
rng = np.random.default_rng(0)
res = []
for _ in range(300):
    parts = []
    for k, s in enumerate(rng.choice(sc, len(sc))):
        p = g[s].copy(); p["token"] = p.token + f"#{k}"; parts.append(p)
    res.append(pd.concat(parts))
G = {j: np.array([gain(r, j) for r in res]) for j in js}
P2 = {j: np.array([at_crash(curve(r, j), 0.02) for r in res]) for j in js + ["const_a"]}
p2 = {j: at_crash(curve(df, j), 0.02) for j in js + ["const_a"]}
print(f"POOLED held-out: {df.token.nunique()} windows, {len(sc)} scenes")
for j in js:
    print(f"  {j:10s} gain {gain(df, j):+.3f} [{np.quantile(G[j], .025):+.3f}, {np.quantile(G[j], .975):+.3f}] | progress@<=2% {p2[j]:.2f} [{np.quantile(P2[j], .025):.2f}, {np.quantile(P2[j], .975):.2f}] | speed-blind {strat(df, j):.3f}")
print(f"  plan-blind progress@<=2% {p2['const_a']:.2f} [{np.quantile(P2['const_a'], .025):.2f}, {np.quantile(P2['const_a'], .975):.2f}]")
for a, b in (("cvmarg_orig", "cvref"), ("cvmarg_orig", "self_orig"), ("cvmarg_rpt", "cvref_rpt"), ("cvmarg_rpt", "self_rpt"), ("cvmarg_orig", "cvref_rpt"), ("physics", "cvmarg_orig"), ("physics", "cvmarg_rpt"), ("h_gap0", "cvmarg_orig"), ("oracle", "cvmarg_orig")):
    d = G[a] - G[b]
    print(f"  paired gain {a} - {b}: {gain(df, a) - gain(df, b):+.3f} [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]")
d = P2["self_orig"] - P2["const_a"]
print(f"  progress@<=2% self_orig - plan-blind: {p2['self_orig'] - p2['const_a']:+.2f} [{np.quantile(d, .025):+.2f}, {np.quantile(d, .975):+.2f}]")
