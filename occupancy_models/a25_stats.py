"""A25 endpoints: randomized-plan training (RPT, past-only plan timing) vs logged-anchored CPC (A24) and control, SWTC
real-car windows of fold B, paired by window, scene-cluster bootstrap B = 10,000, seed 0.
Usage: python audit3/a25_stats.py <orig> <ctrl> <cpc> <rpt05> <rpt10>   (dirs with soft.npy, soft_tokens.json, truth.pkl)
Lead beta (per window, collide plan, +2 s): 1 - s / dx_ego, s = the longitudinal shift best aligning the collide
forecast's obstacle profile with the logged forecast's in the lead's lane corridor (|y - lead y| <= 2 m, own ego box
excluded); background beta: the same for |y - lead y| > 4 m, |y| <= 20 m (audit3/plan_leak_bg.py)."""
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

SH = np.arange(-75, 76)


def align_beta(a, b, dx):
    if a.sum() < 3 or b.sum() < 3 or abs(dx) < 3:
        return np.nan
    sc = []
    for s in SH:
        u, v = (a[: 200 - s], b[s:]) if s >= 0 else (a[-s:], b[: 200 + s])
        sc.append(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
    return 1.0 - SH[int(np.argmax(sc))] * 0.4 / dx


def per_window(d):
    d = Path(d)
    truth = pickle.load(open(d / "truth.pkl", "rb"))
    soft = np.load(d / "soft.npy", mmap_mode="r")
    own = (np.abs(pi.GX - EGO_C) <= EGO_L / 2 + 0.5) & (np.abs(pi.GY) <= EGO_W / 2 + 0.3)
    rows = []
    for i, tok in enumerate(json.load(open(d / "soft_tokens.json"))):
        t = truth[tok]
        L, W = t["LW"]
        r = {"token": tok, "scene": t["scene"]}
        vc, vl = 0.0, 0.0
        for h in (1, 2, 3):
            lx, ly, lyaw = t["track"][2 * h]
            prof = {}
            for p_i, plan in enumerate(PLANS):
                fx, fy, fyaw = plan_pose(t["plans"][plan], h)
                m = np.asarray(soft[i, p_i, h - 1], np.float32)
                lead = box_in_frame(fx, fy, fyaw, lx, ly, lyaw, L, W)
                r[f"kept_{plan}_{h}"] = float(pi.q50(m, lead) >= TAU) if lead.any() else 0.0
                if plan in ("collide", "logged"):
                    ob = box_in_frame(fx, fy, fyaw, fx + EGO_C * math.cos(fyaw), fy + EGO_C * math.sin(fyaw), fyaw,
                                      EGO_L, EGO_W)
                    v = float(m[ob].max() >= TAU)
                    if plan == "collide":
                        vc = max(vc, v)
                    else:
                        vl = max(vl, v)
                    by = math.sin(-fyaw) * (lx - fx) + math.cos(-fyaw) * (ly - fy)
                    lane = (np.abs(pi.GY - by) <= 2.0) & ~own
                    bg = (np.abs(pi.GY - by) > 4.0) & (np.abs(pi.GY) <= 20.0)
                    prof[plan] = (fx, (m * lane).sum(1), (m * bg).sum(1))
            dx = prof["collide"][0] - prof["logged"][0]
            r[f"beta_lead_{h}"] = align_beta(prof["collide"][1], prof["logged"][1], dx)
            r[f"beta_bg_{h}"] = align_beta(prof["collide"][2], prof["logged"][2], dx)
        r["self_recall"], r["self_fa"] = vc, vl
        r["clearing2"] = r["kept_logged_2"] - r["kept_collide_2"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("token")


def boot_mean_diff(a, b, col, B=10000):
    m = a[[col, "scene"]].join(b[[col]], rsuffix="_b", how="inner")
    g = m.groupby("scene").agg(x=(col, "sum"), y=(f"{col}_b", "sum"), n=(col, "size")).to_numpy(float)
    idx = np.random.default_rng(0).integers(0, len(g), (B, len(g)))
    reps = (g[idx, 0].sum(1) - g[idx, 1].sum(1)) / g[idx, 2].sum(1)
    dd = (g[:, 0].sum() - g[:, 1].sum()) / g[:, 2].sum()
    return dd, np.quantile(reps, 0.025), np.quantile(reps, 0.975), len(m), len(g)


def boot_median_diff(a, b, col, B=10000):
    m = a[[col, "scene"]].join(b[[col]], rsuffix="_b", how="inner").dropna()
    m["d"] = m[col] - m[f"{col}_b"]
    scenes = m.scene.unique()
    g = {s: m[m.scene == s].d.to_numpy() for s in scenes}
    rng = np.random.default_rng(0)
    reps = [np.median(np.concatenate([g[s] for s in rng.choice(scenes, len(scenes))])) for _ in range(B)]
    return float(m.d.median()), np.quantile(reps, 0.025), np.quantile(reps, 0.975), len(m), len(scenes)


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}] (n {t[3]}, {t[4]} scenes)"


names = ("original", "control", "CPC", "RPT-0.5", "RPT-1.0")
D = dict(zip(names, (per_window(x) for x in sys.argv[1:6])))
for n, df in D.items():
    print(f"{n:9s}: kept +2 s logged {df.kept_logged_2.mean():.3f} / collide {df.kept_collide_2.mean():.3f} -> clearing "
          f"{df.clearing2.mean():+.3f} | self recall {df.self_recall.mean():.3f} | self FA {df.self_fa.mean():.3f} | "
          f"lead beta +2 s median {df.beta_lead_2.median():+.2f} (+3 s {df.beta_lead_3.median():+.2f}) | "
          f"background beta +2 s {df.beta_bg_2.median():+.2f}")
R1, P, C, R5 = D["RPT-1.0"], D["CPC"], D["control"], D["RPT-0.5"]
p1 = boot_median_diff(R1, P, "beta_lead_2")
p2 = boot_mean_diff(R1, P, "clearing2")
p3 = boot_mean_diff(R1, P, "self_recall")
p4 = boot_mean_diff(R1, C, "self_fa")
print("P1 lead beta +2 s (median paired diff), RPT-1.0 - CPC:", fmt(p1), "->", "SUPPORTED" if p1[2] < 0 else "NOT SUPPORTED")
print("P2 clearing +2 s, RPT-1.0 - CPC:", fmt(p2), "->", "SUPPORTED" if p2[2] < 0 else "NOT SUPPORTED")
print("P3 self recall, RPT-1.0 - CPC:", fmt(p3), "->", "SUPPORTED" if p3[1] > 0 else "NOT SUPPORTED")
print("P4 self false alarm, RPT-1.0 - control (upper <= +0.05):", fmt(p4), "->", "SUPPORTED" if p4[2] <= 0.05 else "NOT SUPPORTED")
print("descriptive: RPT-0.5 - CPC (anchoring alone): lead beta", fmt(boot_median_diff(R5, P, "beta_lead_2")),
      "| clearing", fmt(boot_mean_diff(R5, P, "clearing2")), "| recall", fmt(boot_mean_diff(R5, P, "self_recall")),
      "| FA", fmt(boot_mean_diff(R5, P, "self_fa")))
print("descriptive: RPT-1.0 - control: lead beta", fmt(boot_median_diff(R1, C, "beta_lead_2")), "| clearing",
      fmt(boot_mean_diff(R1, C, "clearing2")), "| recall", fmt(boot_mean_diff(R1, C, "self_recall")))
print("descriptive: background beta RPT-1.0 - CPC", fmt(boot_median_diff(R1, P, "beta_bg_2")))
