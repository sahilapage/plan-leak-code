"""Vista controllability, generated vs generated (registered fallback, see prereg): does Vista's camera execute the
plan? For each moving-ego window and plan P in {collide, d3, d1, d-2}, every frame k of P with t_k in [0.5, 1.5] s is
matched (ORB + ratio test + RANSAC fundamental inliers) against every frame of the SAME seed's logged-plan clip; the
best match k'' places P's camera at the logged progress s_logged(t_k''). ratio = (s_logged(t_k'') - s_logged(t_k)) /
(s_P(t_k) - s_logged(t_k)) over frames with |commanded difference| >= 0.5 m (median per window). Plan ignored -> ~0,
executed -> ~1 (capped by the clip: matches beyond the last logged frame saturate).
Usage: python audit3/vista_ctrl2.py <pack> <out_vista> <out.csv>"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vista_leak import FR_T, interp, plan_wps  # noqa: E402

PLANS = ("collide", "d3", "d1", "d-2")


def main(pack, out, dst):
    pack, out = Path(pack), Path(out)
    W = {w["token"]: w for w in json.load(open(pack / "windows.json"))}
    toks = (pack / "main_vista.txt").read_text().split()
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def feats(p):
        return orb.detectAndCompute(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY), None)

    def inl(fa, fb):
        (ka, da), (kb, db) = fa, fb
        if da is None or db is None:
            return 0
        m = [p[0] for p in bf.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(m) < 15:
            return 0
        pa = np.float32([ka[x.queryIdx].pt for x in m])
        pb = np.float32([kb[x.trainIdx].pt for x in m])
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.99)
        return int(mask.sum()) if mask is not None else 0
    rows = []
    for tok in toks:
        w = W[tok]
        sl = interp(plan_wps(w, "logged"), FR_T)[:, 0]
        if sl[12] / 1.0 < 2.0:                           # moving ego only (>= 2 m/s over the first second)
            continue
        L = [feats(out / tok / "logged_s0" / f"f{k:02d}.jpg") for k in range(25)]
        row = {"token": tok, "v0": float(sl[12])}
        for pl in PLANS:
            if not (out / tok / f"{pl}_s0" / "f24.jpg").exists():
                continue
            sp = interp(plan_wps(w, pl), FR_T)[:, 0]
            rs = []
            for k in range(25):
                if not (0.5 <= FR_T[k] <= 1.5) or abs(sp[k] - sl[k]) < 0.5:
                    continue
                fp = feats(out / tok / f"{pl}_s0" / f"f{k:02d}.jpg")
                sc = np.array([inl(fp, fl) for fl in L])
                kk = int(sc.argmax())
                rs.append((sl[kk] - sl[k]) / (sp[k] - sl[k]))
            row[f"ratio_{pl}"] = float(np.median(rs)) if rs else np.nan
        rows.append(row)
        print(tok[:8], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k != "token"},
              flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(dst, index=False)
    print(f"\nmoving-ego windows {len(D)}")
    for pl in PLANS:
        c = f"ratio_{pl}"
        if c in D:
            r = D[c].dropna()
            print(f"  {pl:8s} realized / commanded: median {r.median():+.2f} (n {len(r)}), share > 0.5 {(r > 0.5).mean():.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
