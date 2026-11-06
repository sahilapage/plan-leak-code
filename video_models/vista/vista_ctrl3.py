"""Vista controllability, EXPLORATORY (after both registered localizers failed their validity checks): calibrated
scene expansion. A forward camera move of D expands static image points radially about the focus of expansion
(r_B / r_A ~ Z / (Z - D)). Calibration per window and start frame k: the logged clip's own pairs (k, k + m) give the
median radial expansion e_m for the known logged advance D_m = s_logged(t_{k+m}) - s_logged(t_k) (same scene, same
depths). The plan P's frame k against the logged frame k gives its expansion e_P, converted to an advance D_P by
interpolating (e_m, D_m); ratio = D_P / (s_P(t_k) - s_logged(t_k)). Matches: ORB + ratio test + RANSAC fundamental
inliers, radius >= 80 px from the image centre (straight driving: FOE ~ centre); expansion = median log(r_B / r_A).
Validity (V1): the calibration is monotone (Spearman(D_m, e_m) >= 0.8) in the window; (V2) braking gives D < 0.
Usage: python audit3/vista_ctrl3.py <pack> <out_vista> <out.csv>"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vista_leak import FR_T, interp, plan_wps  # noqa: E402

PLANS = ("collide", "d3", "d1", "d-2")


def main(pack, out, dst):
    pack, out = Path(pack), Path(out)
    W = {w["token"]: w for w in json.load(open(pack / "windows.json"))}
    toks = (pack / "main_vista.txt").read_text().split()
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    cache = {}

    def feats(p):
        if p not in cache:
            im = cv2.imread(str(p))
            cache[p] = (orb.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None), im.shape[1] / 2, im.shape[0] / 2)
        return cache[p]

    def expansion(pa_, pb_):
        (ka, da), cx, cy = feats(pa_)
        (kb, db), _, _ = feats(pb_)
        if da is None or db is None:
            return np.nan
        m = [p[0] for p in bf.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(m) < 20:
            return np.nan
        A = np.float32([ka[x.queryIdx].pt for x in m])
        B = np.float32([kb[x.trainIdx].pt for x in m])
        _, mask = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, 1.5, 0.99)
        if mask is None:
            return np.nan
        A, B = A[mask.ravel() == 1], B[mask.ravel() == 1]
        ra = np.hypot(A[:, 0] - cx, A[:, 1] - cy)
        rb = np.hypot(B[:, 0] - cx, B[:, 1] - cy)
        ok = ra >= 80
        return float(np.median(np.log(rb[ok] / ra[ok]))) if ok.sum() >= 15 else np.nan
    rows = []
    for tok in toks:
        w = W[tok]
        sl = interp(plan_wps(w, "logged"), FR_T)[:, 0]
        if sl[12] < 2.0 or not (out / tok / "logged_s0" / "f24.jpg").exists():
            continue
        lg = lambda k: out / tok / "logged_s0" / f"f{k:02d}.jpg"  # noqa: E731
        row = {"token": tok, "v0": float(sl[12])}
        per = {pl: [] for pl in PLANS}
        mono = []
        for k in (6, 9, 12):
            cal = [(sl[k + m] - sl[k], expansion(lg(k), lg(k + m))) for m in range(1, 25 - k)]
            cal = [(d, e) for d, e in cal if np.isfinite(e)]
            if len(cal) < 5:
                continue
            D, E = np.array(cal).T
            mono.append(spearmanr(D, E).correlation)
            o = np.argsort(E)
            for pl in PLANS:
                p = out / tok / f"{pl}_s0" / f"f{k:02d}.jpg"
                if not p.exists():
                    continue
                cmd = interp(plan_wps(w, pl), FR_T)[k, 0] - sl[k]
                if abs(cmd) < 0.5:
                    continue
                e = expansion(lg(k), p)
                if not np.isfinite(e):
                    continue
                # negative expansion (camera behind): mirror the calibration (contraction ~ -expansion)
                d_hat = np.interp(e, E[o], D[o]) if e >= 0 else -np.interp(-e, E[o], D[o])
                per[pl].append(d_hat / cmd)
        row["calib_spearman"] = float(np.median(mono)) if mono else np.nan
        for pl in PLANS:
            row[f"ratio_{pl}"] = float(np.median(per[pl])) if per[pl] else np.nan
        rows.append(row)
        print(tok[:8], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k != "token"},
              flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(dst, index=False)
    ok = D[D.calib_spearman >= 0.8]
    print(f"\nmoving-ego windows {len(D)}, with a monotone calibration (V1) {len(ok)}")
    for pl in PLANS:
        r = ok[f"ratio_{pl}"].dropna()
        print(f"  {pl:8s} realized / commanded advance: median {r.median():+.2f} [IQR {r.quantile(.25):+.2f}, "
              f"{r.quantile(.75):+.2f}] (n {len(r)})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
