"""Vista controllability check (registered before output, see prereg): does Vista's camera execute the plan?
Visual localization of a generated frame (+2.0 s) against the REAL 12 Hz CAM_FRONT stream of the same drive: ORB
features + ratio test + RANSAC fundamental-matrix inliers to every real frame from +0.5 to +4.0 s (same 1024 x 576
resize as Vista's input); t* = the real time with the most inliers; camera position = logged progress s_log(t*)
(2 Hz keyframe progress, linear in between, constant speed past +3 s). Realized extra advance of plan P = s_log(t*_P)
- s_log(t*_logged); commanded = s_P(2.0) - s_logged(2.0).
K0: logged-plan t* within 0.5 s of 2.0 s in >= 70 % of windows. K1: median realized / commanded (collide) >= 0.5.
Usage (pod, vista env): python audit3/vista_ctrl.py /workspace/pack /workspace/out_vista /workspace/data/nuscenes out.csv"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vista_leak import FR_T, H_IMG, W_IMG, interp, plan_wps  # noqa: E402

PLANS = ("collide", "d3", "d1", "d-2")
T0_REAL, T1_REAL, K_GEN = 0.5, 4.0, 24


def feats(img, orb):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(g, None)


def inliers(fa, fb, bf):
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return 0
    m = [p[0] for p in bf.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(m) < 15:
        return len(m) // 2
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.99)
    return int(mask.sum()) if mask is not None else 0


def main(pack, out, nusc, dst):
    pack, out, nusc = Path(pack), Path(out), Path(nusc)
    W = {w["token"]: w for w in json.load(open(pack / "windows.json"))}
    toks = (pack / "main_vista.txt").read_text().split()
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    rows = []
    for tok in toks:
        w = W[tok]
        s_log = np.asarray(w["s_log"], float)
        f0 = w["frames"][0]["file"]
        log, ts0 = f0.split("__CAM_FRONT__")[0], int(f0.rsplit("__", 1)[1].split(".")[0])
        real = []
        for d in (nusc / "sweeps" / "CAM_FRONT", nusc / "samples" / "CAM_FRONT"):
            for q in d.glob(f"{log}__CAM_FRONT__*.jpg"):
                t = (int(q.name.rsplit("__", 1)[1].split(".")[0]) - ts0) / 1e6
                if T0_REAL <= t <= T1_REAL:
                    real.append((t, q))
        real.sort()
        if len(real) < 10:
            continue
        RF = [(t, feats(cv2.resize(cv2.imread(str(q)), (W_IMG, H_IMG)), orb)) for t, q in real]

        def s_of(t):
            tk = 0.5 * np.arange(len(s_log))
            if t <= tk[-1]:
                return float(np.interp(t, tk, s_log))
            return float(s_log[-1] + (t - tk[-1]) * (s_log[-1] - s_log[-2]) / 0.5)

        def locate(plan):
            p = out / tok / f"{plan}_s0" / f"f{K_GEN:02d}.jpg"
            if not p.exists():
                return None, None
            fg = feats(cv2.resize(cv2.imread(str(p)), (W_IMG, H_IMG)), orb)
            sc = np.array([inliers(fg, fr, bf) for _, fr in RF])
            k = int(sc.argmax())
            return RF[k][0], int(sc[k])
        tl, nl = locate("logged")
        if tl is None:
            continue
        s_cmd_l = interp(plan_wps(w, "logged"), FR_T)[K_GEN, 0]
        row = {"token": tok, "v0": float(s_log[1] / 0.5), "t_logged": tl, "inl_logged": nl}
        for pl in PLANS:
            tp, npl = locate(pl)
            if tp is None:
                continue
            cmd = interp(plan_wps(w, pl), FR_T)[K_GEN, 0] - s_cmd_l
            real_adv = s_of(tp) - s_of(tl)
            row.update({f"t_{pl}": tp, f"inl_{pl}": npl, f"cmd_{pl}": cmd, f"real_{pl}": real_adv,
                        f"ratio_{pl}": real_adv / cmd if abs(cmd) >= 1.0 else np.nan})
        rows.append(row)
        print(f"{tok[:8]} v0 {row['v0']:.1f} t_logged {tl:.2f} ({nl}) | collide t {row.get('t_collide', np.nan):.2f} "
              f"cmd {row.get('cmd_collide', np.nan):+.1f} real {row.get('real_collide', np.nan):+.1f}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(dst, index=False)
    mv = D[D.v0 >= 2.0]
    print(f"\nwindows {len(D)}, moving-ego (v0 >= 2 m/s) {len(mv)}")
    print(f"K0: logged-plan t* within 0.5 s of 2.0 s: {(abs(mv.t_logged - 2.0) <= 0.5).mean():.2f} (moving windows)")
    for pl in PLANS:
        c = f"ratio_{pl}"
        if c in mv:
            r = mv[c].dropna()
            print(f"  {pl:8s} realized / commanded extra advance: median {r.median():+.2f} (n {len(r)}), "
                  f"commanded median {mv[f'cmd_{pl}'].median():+.1f} m, realized median {mv[f'real_{pl}'].median():+.1f} m")


if __name__ == "__main__":
    main(*sys.argv[1:5])
