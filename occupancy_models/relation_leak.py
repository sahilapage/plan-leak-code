"""Plan-Leak Index by agent relation (exploratory; risk (a) of docs/PRIOR_ART_AUDIT.md). For every annotated vehicle in
the SWTC real-car windows (src/sw_realhazard.py dirs, original model, plans logged / collide / brake / still), class at
t0 (t0 ego frame, 3 s annotated track):
  lead      moving, same direction (|yaw| <= 30 deg), |y| <= 1.5 m, 4 <= x <= 35 m   (true causal effect of our plan ~0)
  follower  moving, same direction, |y| <= 1.5 m, -35 <= x <= -4 m                  (does react to us: effect > 0)
  adjacent  moving, same direction, 2.5 <= |y| <= 8 m                              (~0 unless it merges)
  oncoming  moving, |yaw| >= 150 deg                                                (~0)
  parked    3 s displacement < 1 m                                                  (0)
Isolated agents only: no other annotated vehicle within |dy| <= 2.5 m, |dx| <= 25 m at any step (the corridor
profile must belong to this agent). Imagined displacement: the logged plan's obstacle profile along x in the agent's
corridor (|y - agent y| <= 1.5 m, own ego footprint excluded), +-10 m around the agent's TRUE position (world-anchored,
in each plan's own frame), slid over the plan's profile (+-30 m); delta = best cosine shift (m, world).
beta = delta / dx_ego (dx_ego = plan ego x - logged ego x at h, |dx_ego| >= 2 m). 0 = world-consistent, 1 = keeps the
gap to the ego. Data side: corr / slope of the agent's real x-acceleration on the ego's (observational).
Scene-cluster bootstrap of the median, B = 2000, seed 0. Usage (server, vla env): python audit3/relation_leak.py <d> ..."""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import probe_insert as pi  # noqa: E402
import realhazard as rh  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, PLANS, plan_pose  # noqa: E402

RES = 0.4
EST = "align"                                         # or "centroid" (--centroid: audit estimator)
LW, PW = 25, 75                                       # half-widths in cells: logged 10 m, plan 30 m
T = 0.5 * np.arange(7)


def accel(xs):
    A = np.stack([np.ones_like(T), T, 0.5 * T ** 2], 1)
    return float(np.linalg.lstsq(A, xs, rcond=None)[0][2])


def classify(tr):
    x0, y0, yw = tr[0]
    if np.hypot(*(tr[6, :2] - tr[0, :2])) < 1.0:
        return "parked" if abs(x0) <= 35 and abs(y0) <= 15 else None
    if abs(yw) <= math.radians(30):
        if abs(y0) <= 1.5 and 4 <= x0 <= 35:
            return "lead"
        if abs(y0) <= 1.5 and -35 <= x0 <= -4:
            return "follower"
        if 2.5 <= abs(y0) <= 8 and -30 <= x0 <= 35:
            return "adjacent"
    if abs(yw) >= math.radians(150) and abs(x0) <= 35 and abs(y0) <= 12:
        return "oncoming"
    return None


def main(dirs):
    from nuscenes.nuscenes import NuScenes
    import dome_infer as di
    nusc = NuScenes("v1.0-trainval", dataroot=str(rh.DATA / "nuscenes"), verbose=False)
    own = (np.abs(pi.GX - EGO_C) <= EGO_L / 2 + 0.5) & (np.abs(pi.GY) <= EGO_W / 2 + 0.3)
    ctr = pi.GX[:, 0]
    rows, data = [], []
    for d in map(Path, dirs):
        truth = pickle.load(open(d / "truth.pkl", "rb"))
        soft = np.load(d / "soft.npy", mmap_mode="r")
        for i, tok in enumerate(json.load(open(d / "soft_tokens.json"))):
            t = truth[tok]
            sample = nusc.get("sample", tok)
            P0 = di.pose_of({"token": tok, **_info(nusc, sample)})
            lg = np.vstack([[0.0, 0.0], np.asarray(t["plans"]["logged"], float)])
            agents = []
            for at in sample["anns"]:
                ann = nusc.get("sample_annotation", at)
                if ann["category_name"].startswith(rh.VEH_CATS):
                    tr = rh.lead_track(nusc, ann, P0)
                    if tr is not None:
                        agents.append(tr)
            for k, tr in enumerate(agents):
                rel = classify(tr)
                if rel is None:
                    continue
                iso = all(not ((np.abs(o[:, 1] - tr[:, 1]) <= 2.5) & (np.abs(o[:, 0] - tr[:, 0]) <= 25)).any()
                          for j, o in enumerate(agents) if j != k)
                data.append({"scene": t["scene"], "rel": rel, "iso": iso, "a_ag": accel(tr[:, 0]),
                             "a_ego": accel(lg[:, 0])})
                if not iso:
                    continue
                for h in (1, 2, 3):
                    ax, ay, _ = tr[2 * h]
                    prof = {}
                    for p_i, plan in enumerate(PLANS):
                        fx, fy, fyaw = plan_pose(t["plans"][plan], h)
                        c, s_ = math.cos(-fyaw), math.sin(-fyaw)
                        bx, by = c * (ax - fx) - s_ * (ay - fy), s_ * (ax - fx) + c * (ay - fy)
                        m = np.asarray(soft[i, p_i, h - 1], np.float32)
                        p1 = (m * ((np.abs(pi.GY - by) <= 1.5) & ~own)).sum(1)
                        prof[plan] = (fx, p1, int(np.argmin(np.abs(ctr - bx))))
                    fx0, p0, c0 = prof["logged"]
                    if not (LW <= c0 < len(ctr) - LW):
                        continue
                    ref = p0[c0 - LW: c0 + LW + 1]
                    if ref.sum() < 3:
                        continue
                    for plan in ("collide", "brake", "still"):
                        fx, p1, c1 = prof[plan]
                        dx = fx - fx0
                        if abs(dx) < 2.0:
                            continue
                        if EST == "centroid":                    # audit estimator: world-anchored mass centroids
                            CW = 50                              # +-20 m
                            if not (CW <= c0 < len(ctr) - CW and CW <= c1 < len(ctr) - CW):
                                continue
                            r0, r1 = p0[c0 - CW: c0 + CW + 1], p1[c1 - CW: c1 + CW + 1]
                            if r0.sum() < 3 or r1.sum() < 3:
                                continue
                            u = np.arange(-CW, CW + 1) * RES
                            delta = float((u * r1).sum() / r1.sum() - (u * r0).sum() / r0.sum())
                            rows.append({"scene": t["scene"], "rel": rel, "h": h, "plan": plan, "dx": dx,
                                         "beta": delta / dx, "score": np.nan})
                            continue
                        best, arg = -1.0, None
                        for sh in range(-PW + LW, PW - LW + 1):
                            lo = c1 + sh - LW
                            if lo < 0 or lo + 2 * LW + 1 > len(ctr):
                                continue
                            seg = p1[lo: lo + 2 * LW + 1]
                            if seg.sum() < 1.0:
                                continue
                            sc = float(seg @ ref / (np.linalg.norm(seg) * np.linalg.norm(ref) + 1e-9))
                            if sc > best:
                                best, arg = sc, sh
                        if arg is None:
                            continue
                        rows.append({"scene": t["scene"], "rel": rel, "h": h, "plan": plan, "dx": dx,
                                     "beta": arg * RES / dx, "score": best})
    df, dd = pd.DataFrame(rows), pd.DataFrame(data)
    df.to_parquet(Path(dirs[0]).parent / f"relation_leak_{EST}.parquet")
    rng = np.random.default_rng(0)

    def bmed(f, col="beta", B=2000):
        sc = f.scene.unique()
        g = {s: f[f.scene == s][col].to_numpy() for s in sc}
        v = [np.median(np.concatenate([g[s] for s in rng.choice(sc, len(sc))])) for _ in range(B)]
        q = np.quantile(v, [0.025, 0.975])
        return f"{np.median(f[col]):+.2f} [{q[0]:+.2f}, {q[1]:+.2f}]"
    print("agents (all / isolated):", dd.groupby("rel").iso.agg(["size", "sum"]).to_dict("index"))
    for rel in ("lead", "follower", "adjacent", "oncoming", "parked"):
        for h in (2, 3):
            f = df[(df.rel == rel) & (df.h == h)]
            if len(f) >= 8:
                per = " | ".join(f"{p} {np.median(f[f.plan == p].beta):+.2f} (n {(f.plan == p).sum()})"
                                 for p in ("collide", "brake", "still"))
                print(f"{rel:9s} +{h} s beta pooled {bmed(f)} n {len(f)} ({f.scene.nunique()} scenes) | {per}")
        e = dd[dd.rel == rel]
        if len(e) > 10:
            x = e.a_ego - e.a_ego.mean()
            print(f"{'':9s} data: corr(a_agent, a_ego) {np.corrcoef(e.a_ego, e.a_ag)[0, 1]:+.2f}, slope "
                  f"{float((x * (e.a_ag - e.a_ag.mean())).sum() / (x * x).sum()):+.2f} (n {len(e)})")


def _info(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return {"ego2global_translation": ep["translation"], "ego2global_rotation": ep["rotation"]}


if __name__ == "__main__":
    if "--centroid" in sys.argv:
        EST = "centroid"
        sys.argv.remove("--centroid")
    main(sys.argv[1:])
