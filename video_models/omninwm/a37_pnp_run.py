"""A37 amendment 2: controllability via PnP odometry (src/a37_pnp.py) and world-anchored beta.
Usage: python src/a37_pnp_run.py <windows.json> <out_dir with a37_scores_s0_n25.csv> <out.csv>"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_score as S, a37_pnp as pnp
W = {w["token"]: w for w in json.load(open(sys.argv[1]))}
out = Path(sys.argv[2]); sc = pd.read_csv(out / "a37_scores_s0_n25.csv").set_index("token")
H = 3; rows = []
def chain(G, K):
    z, v, M = G; s = 0.0; n = []
    for k in range(H, K + 1, H):
        dx, m = pnp.rel_x(z, v, k - H, z, v, k); n.append(m); s += dx if np.isfinite(dx) else np.nan
    return s, int(np.median(n))
for tok in sc.index:
    if not np.isfinite(sc.loc[tok, "t_star"]): continue
    w = W[tok]; d = out / tok
    t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
    K = int(np.argmin(np.abs(t - sc.loc[tok, "t_star"]))) // H * H
    G = {pl: pnp.load(d / f"{pl}_s0_n25") for pl in ("logged", "collide")}
    (zl, vl, Ml), (zc, vc, Mc) = G["logged"], G["collide"]
    cl, nl = chain(G["logged"], K); cc, nc = chain(G["collide"], K)
    cross, ncr = pnp.rel_x(zl, vl, K, zc, vc, K)                 # collide camera in logged ego frame at K
    gap = {pl: S.track_lead(S.backproject(G[pl][0]["depth"], G[pl][0]["seg"], G[pl][0]["cam_k"], G[pl][0]["cam2ego"], S.DYN), G[pl][2], w)[K] - G[pl][2][K, 0, 3] for pl in G}
    dgap = gap["collide"] - gap["logged"]; ds = Mc[K, 0, 3] - Ml[K, 0, 3]
    rows.append(dict(tok=tok[:8], scene=w["scene"], t=t[K], log_cmd=Ml[K, 0, 3], log_pnp=cl, n_l=nl, col_cmd=Mc[K, 0, 3], col_pnp=cc,
                     ds_cmd=ds, ds_chain=cc - cl, ds_cross=cross, n_cross=ncr, dgap=dgap,
                     beta_cmd=(dgap + ds) / ds, beta_cross=(dgap + cross) / cross if cross > 0.5 else np.nan))
pd.set_option("display.width", 250); df = pd.DataFrame(rows); print(df.round(2).to_string())
df.to_csv(Path(sys.argv[3]), index=False)
