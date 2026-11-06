"""A24 endpoints: CPC vs control (and original), SWTC real-car windows of fold B, paired by window, scene bootstrap.
Usage: python audit3/a24_stats.py <orig_dir> <ctrl_dir> <cpc_dir>   (each with soft.npy, soft_tokens.json, truth.pkl)"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import probe_insert as pi  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, PLANS, TAU, box_in_frame, plan_pose  # noqa: E402


def per_window(d):
    d = Path(d)
    truth = pickle.load(open(d / "truth.pkl", "rb"))
    soft = np.load(d / "soft.npy", mmap_mode="r")
    rows = []
    for i, tok in enumerate(json.load(open(d / "soft_tokens.json"))):
        t = truth[tok]
        L, W = t["LW"]
        r = {"token": tok, "scene": t["scene"]}
        vc, vl = 0.0, 0.0
        for h in (1, 2, 3):
            lx, ly, lyaw = t["track"][2 * h]
            for p_i, plan in enumerate(PLANS):
                fx, fy, fyaw = plan_pose(t["plans"][plan], h)
                m = np.asarray(soft[i, p_i, h - 1], np.float32)
                lead = box_in_frame(fx, fy, fyaw, lx, ly, lyaw, L, W)
                r[f"kept_{plan}_{h}"] = float(pi.q50(m, lead) >= TAU) if lead.any() else 0.0
                if plan in ("collide", "logged"):
                    own = box_in_frame(fx, fy, fyaw, fx + EGO_C * math.cos(fyaw), fy + EGO_C * math.sin(fyaw), fyaw,
                                       EGO_L, EGO_W)
                    v = float(m[own].max() >= TAU)
                    if plan == "collide":
                        vc = max(vc, v)
                    else:
                        vl = max(vl, v)
        r["self_recall"], r["self_fa"] = vc, vl
        r["clearing2"] = r["kept_logged_2"] - r["kept_collide_2"]
        r["clearing3"] = r["kept_logged_3"] - r["kept_collide_3"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("token")


def boot_diff(a, b, col, B=10000):
    m = a[[col, "scene"]].join(b[[col]], rsuffix="_b", how="inner")
    g = m.groupby("scene").agg(x=(col, "sum"), y=(f"{col}_b", "sum"), n=(col, "size")).to_numpy(float)
    idx = np.random.default_rng(0).integers(0, len(g), (B, len(g)))
    reps = (g[idx, 0].sum(1) - g[idx, 1].sum(1)) / g[idx, 2].sum(1)
    d = (g[:, 0].sum() - g[:, 1].sum()) / g[:, 2].sum()
    return f"{d:+.3f} [{np.quantile(reps, 0.025):+.3f}, {np.quantile(reps, 0.975):+.3f}] (n {len(m)}, {len(g)} scenes)"


O, C, P = (per_window(x) for x in sys.argv[1:4])
for name, df in (("original", O), ("control", C), ("CPC", P)):
    print(f"{name:9s}: kept +2 s logged {df.kept_logged_2.mean():.3f} / collide {df.kept_collide_2.mean():.3f} -> clearing "
          f"{df.clearing2.mean():+.3f} (+3 s {df.clearing3.mean():+.3f}) | self recall {df.self_recall.mean():.3f} | "
          f"self false alarm {df.self_fa.mean():.3f}")
print("P1 clearing (+2 s), CPC - control:", boot_diff(P, C, "clearing2"))
print("P2 self recall, CPC - control:", boot_diff(P, C, "self_recall"))
print("P3 self false alarm, CPC - control:", boot_diff(P, C, "self_fa"))
print("descriptive: CPC - original clearing", boot_diff(P, O, "clearing2"), "| recall", boot_diff(P, O, "self_recall"))
