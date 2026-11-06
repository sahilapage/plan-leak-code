"""OmniNWM depth vs true camera depth of the real lead's rear face, at frames where the real car is clearly visible."""
import json, sys
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MPath
sys.path.insert(0, str(Path.home() / "work/drive/paper/corl_final"))
from teaser import box_faces, project
OUT = Path.home() / "work/a37_pod/out_omni"
W = {w["token"]: w for w in json.load(open(Path.home() / "work/drive/results/a37_final/windows_merged.json"))}
toks = open(Path.home() / "work/drive/results/a37_final/main18.txt").read().split()
rows = []
for tok in toks:
    w = W[tok]; d = OUT / tok / "logged_s0_n25"
    z = np.load(d / "front3.npz"); M = np.load(d / "rel_poses.npy")
    t = np.array(json.load(open(d / "reward.json"))["t"])
    lt = np.array(w["lead_track_t0frame"])
    for k in (0, 6, 12):
        cx = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 0]); cy = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 1])
        C2E, K = z["cam2ego"][1, k].astype(float), z["cam_k"][1, k].astype(float)
        _, rear = box_faces(cx, cy, *w["lead_LW"])
        uv, dz = project(rear, M[k], C2E, K)
        if (dz < 1).any():
            continue
        H, Wd = z["depth"].shape[2:]
        yy, xx = np.mgrid[0:H, 0:Wd]
        inside = MPath(uv).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(H, Wd)
        car = inside & (z["seg"][1, k] == 3)
        if car.sum() < 50:
            continue
        dep = z["depth"][1, k].astype(float)[car]
        rows.append((tok[:6], k, dz.mean(), np.median(dep), np.percentile(dep, 5), car.sum()))
r = np.array([x[2:] for x in rows])
for x in rows: print("%s k=%2d true_z %.1f  omni_med %.1f  omni_p5 %.1f  n=%d" % x)
a, b = np.polyfit(r[:, 0], r[:, 1], 1); print("median: omni = %.2f * true + %.2f" % (a, b))
a, b = np.polyfit(r[:, 0], r[:, 2], 1); print("p5:     omni = %.2f * true + %.2f" % (a, b))
print("ratio median", np.median(r[:, 1] / r[:, 0]), "ratio p5", np.median(r[:, 2] / r[:, 0]))
