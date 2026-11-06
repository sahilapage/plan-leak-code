"""Exploratory audit: does OmniNWM execute the BRAKING plan (A37b d-2)? PnP odometry (src/a37_pnp.py, as the
registered G1 check) on the logged and d-2 clips at t ~ 1.5 s: realized extra advance ds_chain vs commanded; the lead's
ego-frame gap change dgap; beta_world = (dgap + ds_chain) / ds_chain (lead displacement per metre of REALIZED change).
Usage: python audit3/omni_brake_pnp.py <windows.json> <out_omni> <tokens.txt>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import a37_pnp as pnp  # noqa: E402
import a37_score as S  # noqa: E402

H = 3


def chain(G, K):
    z, v, M = G
    s, n = 0.0, []
    for k in range(H, K + 1, H):
        dx, m = pnp.rel_x(z, v, k - H, z, v, k)
        n.append(m)
        s += dx if np.isfinite(dx) else np.nan
    return s, int(np.median(n))


W = {w["token"]: w for w in json.load(open(sys.argv[1]))}
out = Path(sys.argv[2])
rows = []
for tok in open(sys.argv[3]).read().split():
    w, d = W[tok], out / tok
    if not (d / "d-2_s0_n25" / "front3.npz").exists():
        continue
    t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
    K = int(np.argmin(np.abs(t - 1.5))) // H * H
    G = {pl: pnp.load(d / f"{pl}_s0_n25") for pl in ("logged", "d-2")}
    cl, nl = chain(G["logged"], K)
    cb, nb = chain(G["d-2"], K)
    gap = {pl: S.track_lead(S.backproject(G[pl][0]["depth"], G[pl][0]["seg"], G[pl][0]["cam_k"], G[pl][0]["cam2ego"],
                                          S.DYN), G[pl][2], w)[K] - G[pl][2][K, 0, 3] for pl in G}
    ds_cmd = G["d-2"][2][K, 0, 3] - G["logged"][2][K, 0, 3]
    ds_real = cb - cl
    dgap = gap["d-2"] - gap["logged"]
    rows.append({"tok": tok[:8], "t": t[K], "ds_cmd": ds_cmd, "ds_real": ds_real, "ratio": ds_real / ds_cmd,
                 "n_inl": min(nl, nb), "dgap": dgap,
                 "beta_world": (dgap + ds_real) / ds_real if abs(ds_real) >= 0.5 else np.nan})
D = pd.DataFrame(rows)
print(D.round(2).to_string())
print(f"realized / commanded braking: median {D.ratio.median():+.2f} (n {len(D)}); beta_world median "
      f"{D.beta_world.median():+.2f} (n {D.beta_world.notna().sum()}); lead pulled towards the ego relative to the "
      f"REALIZED camera: {((D.dgap + D.ds_real) < 0).mean():.2f}")
