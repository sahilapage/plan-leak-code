"""Path clearing on REAL cars, camera-based model (SparseWorld-TC; no insertion). Windows of src/realhazard.py (real moving
lead car ahead in the ego lane, detected, tracked 3 s), without the 36 m range limit (SWTC's grid moves with each plan).
Four plans for the same images (src/swtc_verifier.py, one camera feature pass, the occupancy head conditioned per plan):
  logged     the recorded path (never collides)
  collide    the logged path plus the smallest extra constant acceleration putting the ego >= 2 m into the real lead car
  brake      the reference braking plan (4 m/s^2 from the t0 speed; src/planswap.py)
  still      a stationary plan (all waypoints at the origin)
Measures (SWTC obstacle probability, cells in each plan's own future ego frame):
  kept       the real lead car's true box at h, placed in that plan's frame: >= 50 % at p >= tau
  veto       the COLLIDE candidate's ego box at h, placed in that plan's frame: max p >= tau at any of +1 / +2 / +3 s
Path clearing = the real car is kept less when the plan drives into it (collide) than under logged / brake / still.
Usage: python src/sw_realhazard.py prep <fold> <n> <dir>   (vla env, CPU: candidates jsonl + ids + truth pkl)
       python src/sw_realhazard.py score <dir>             (after swtc_verifier.py --soft-all on <dir>/cands.jsonl)
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_hazard as ph  # noqa: E402
import probe_insert as pi  # noqa: E402
import realhazard as rh  # noqa: E402
from planswap import ref_poses  # noqa: E402

PLANS = ("logged", "collide", "brake", "still")
TAU = 0.2
EGO_L, EGO_W, EGO_C = 4.084, 1.73, 1.3          # src/swtc_verifier.py footprint


def wp_from_poses(P0, poses):
    Ti = np.linalg.inv(P0)
    return [[float(v) for v in (Ti @ np.array([*P[:2, 3], 0.0, 1.0]))[:2]] for P in poses]


def prep(fold, n, out):
    import json
    rh.X_MAX = 1e9
    infos, sel = rh.select(fold, n)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    with open(out / "cands.jsonl", "w") as f:
        for s in sel:
            fr = infos[s["scene"]][s["idx"]: s["idx"] + 10]
            lp = [di.pose_of(x) for x in fr]
            P0 = lp[3]
            logged = wp_from_poses(P0, lp[4:10])
            collide = [[x + 0.5 * s["a"] * (0.5 * (k + 1)) ** 2, y] for k, (x, y) in enumerate(logged)]
            brake = wp_from_poses(P0, ref_poses(lp, 4.0))
            still = [[0.0, 0.0]] * 6
            f.write(json.dumps({"id": s["token"], "cands": [logged, collide, brake, still]}) + "\n")
            truth[s["token"]] = {"scene": s["scene"], "track": s["lead_track"], "LW": s["lead_LW"], "a": s["a"],
                                 "plans": {"logged": logged, "collide": collide, "brake": brake, "still": still}}
    (out / "ids.txt").write_text("\n".join(truth))
    pickle.dump(truth, open(out / "truth.pkl", "wb"))
    print(f"prepared {len(truth)} windows, {len({t['scene'] for t in truth.values()})} scenes", flush=True)


def plan_pose(wp, h):
    """t0-frame pose (x, y, yaw) of a plan at h s, as src/swtc_verifier.py cand_poses (heading from the local step)."""
    wp = np.asarray(wp, float)
    prev = np.vstack([[0, 0], wp[:-1]])
    yaw = 0.0
    for i in (1, 3, 5):
        d = wp[i] - prev[i]
        yaw = math.atan2(d[1], d[0]) if np.hypot(*d) > 0.2 else yaw
        if i == 2 * h - 1:
            return wp[i, 0], wp[i, 1], yaw


def box_in_frame(fx, fy, fyaw, x, y, yaw, L, W):
    """(200, 200) mask of a box (t0-frame pose x, y, yaw) in the ego frame at (fx, fy, fyaw)."""
    c, s_ = math.cos(-fyaw), math.sin(-fyaw)
    bx, by = c * (x - fx) - s_ * (y - fy), s_ * (x - fx) + c * (y - fy)
    byaw = yaw - fyaw
    cb, sb = math.cos(byaw), math.sin(byaw)
    u = (pi.GX - bx) * cb + (pi.GY - by) * sb
    w = -(pi.GX - bx) * sb + (pi.GY - by) * cb
    return (np.abs(u) <= L / 2) & (np.abs(w) <= W / 2)


def score(d):
    import json
    import pandas as pd
    d = Path(d)
    truth = pickle.load(open(d / "truth.pkl", "rb"))
    soft = np.load(d / "soft.npy", mmap_mode="r")                     # (N, 4 plans, 3, 200, 200)
    toks = json.load(open(d / "soft_tokens.json"))
    rows = []
    for i, tok in enumerate(toks):
        t = truth[tok]
        L, W = t["LW"]
        for h in (1, 2, 3):
            lx, ly, lyaw = t["track"][2 * h]
            cx, cy, cyaw = plan_pose(t["plans"]["collide"], h)
            row = {"token": tok, "scene": t["scene"], "h": h}
            for p_i, plan in enumerate(PLANS):
                fx, fy, fyaw = plan_pose(t["plans"][plan], h)
                m = np.asarray(soft[i, p_i, h - 1], np.float32)
                lead = box_in_frame(fx, fy, fyaw, lx, ly, lyaw, L, W)
                ego = box_in_frame(fx, fy, fyaw, cx + EGO_C * math.cos(cyaw), cy + EGO_C * math.sin(cyaw), cyaw,
                                   EGO_L, EGO_W)
                row[f"{plan}_kept"] = pi.q50(m, lead) >= TAU if lead.any() else np.nan
                row[f"{plan}_check"] = float(m[ego].max()) if ego.any() else 0.0
                if plan == "logged":                                   # false alarm: the logged path in its own frame
                    own = box_in_frame(fx, fy, fyaw, fx + EGO_C * math.cos(fyaw), fy + EGO_C * math.sin(fyaw), fyaw,
                                       EGO_L, EGO_W)
                    row["logged_fa_check"] = float(m[own].max())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(d / "scored.parquet")
    print(f"{df.token.nunique()} windows, {df.scene.nunique()} scenes; real lead car kept (q50 >= {TAU}) per plan:")
    print(df.groupby("h")[[f"{p}_kept" for p in PLANS]].mean().round(3).to_string())
    v = df.groupby("token")[[f"{p}_check" for p in PLANS] + ["logged_fa_check"]].max() >= TAU
    print("veto of the colliding candidate at any of +1/+2/+3 s, by the forecast it is judged in:")
    print(v[[f"{p}_check" for p in PLANS]].mean().round(3).to_string())
    print(f"false alarm on the logged path (self): {v.logged_fa_check.mean():.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        score(sys.argv[2])
