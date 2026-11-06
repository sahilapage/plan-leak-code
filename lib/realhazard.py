"""Real-agent hazards: do the judges, alone and composed, veto a candidate that drives into a REAL moving car, and do
they leave the (collision-free) logged path alone? No insertion: the hazard is a real vehicle with its real future.
Windows (fold, selection seed 0): near-straight logged path (heading change < 15 deg, lateral drift < 2 m), mean ego speed
>= 3 m/s; a real moving vehicle (annotation speed > 1 m/s, vehicle category) ahead in the ego's lane at t0 (4-40 m ahead,
|lateral| <= 1.5 m, heading within 30 deg), present in all six future keyframes, and detected at t0 by the camera
detector (SparseDrive, score >= 0.3, centre within 2 m).
Candidates:
  logged    the recorded path (never collides: it is what happened)
  collide   the logged path plus the smallest constant extra acceleration a in 0.25 .. 4 m/s^2 (0.25 steps) for which the
            ego's front is >= OV_MIN = 2 m into the lead's true box at one of +1 / +2 / +3 s (a clear collision, not a
            graze; lead positions from its annotations; the ego stays on the logged path's line; poses = logged poses
            shifted along the t0 heading). The contact point must lie <= X_MAX = 36 m ahead of the t0 position, so that
            it is inside every judge's 40 m grid (the blind judge forecasts in the t0 frame).
Judges (veto = check >= tau at ANY of +1 / +2 / +3 s, the candidate's ego box placed in the forecast's frame):
  self     DOME conditioned on the candidate itself (one forecast per candidate)
  swap     DOME conditioned on the reference braking plan (src/planswap.py), one forecast for both candidates
  blind    DOME on the ego-neutral world (src/blind.py), one forecast for both candidates
  objects  the registered permanence layer's objects only (A11: moving detections at their best-scored predicted
           trajectory, heading along velocity), geometric overlap with the candidate's ego box
and compositions (union of vetoes).
Truth: collide = collision (recall); logged = no collision (false alarm).
Usage: python src/realhazard.py select <fold> <n>            (CPU: selection summary)
       python src/realhazard.py run <fold> <n> <out.parquet> (GPU)
"""
import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_hazard as ph  # noqa: E402
import probe_insert as pi  # noqa: E402
from blind import to_frame  # noqa: E402
from common import DATA, DROOT  # noqa: E402
from planswap import ego_box_in, ref_poses  # noqa: E402

A_GRID = np.arange(0.25, 4.01, 0.25)
OV_MIN, X_MAX = 2.0, 36.0
REQUIRE_DET = True                      # False (audit3 fair-physics test): keep leads the camera detector missed
FRONT0 = ph.EGO[0] + ph.EGO[2] / 2
EGO_W = ph.EGO[3]
VEH_CATS = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.construction", "vehicle.trailer", "vehicle.emergency")
SD_VEH = (0, 1, 2, 3, 4)


def shift_pose(P, P0, ds):
    Q = P.copy()
    yaw0 = pi.yaw_of(P0)
    Q[:2, 3] = P[:2, 3] + ds * np.array([math.cos(yaw0), math.sin(yaw0)])
    return Q


def lead_track(nusc, ann0, P0):
    """t0-ego-frame (x, y, yaw) of the annotated vehicle at t = 0, 0.5, .., 3 s, or None if it leaves the annotations."""
    from pyquaternion import Quaternion
    Ti = np.linalg.inv(P0)
    out, a = [], ann0
    for k in range(7):
        if a is None:
            return None
        g = Ti @ np.array([*a["translation"][:2], 0.0, 1.0])
        yaw = Quaternion(a["rotation"]).yaw_pitch_roll[0] - pi.yaw_of(P0)
        out.append((g[0], g[1], (yaw + math.pi) % (2 * math.pi) - math.pi))
        a = nusc.get("sample_annotation", a["next"]) if a["next"] else None
    return np.array(out)


def detections(sd, tok):
    r = sd.get(tok)
    if r is None:
        return []
    m = np.isin(r["labels"], SD_VEH) & (r["scores"] >= 0.3)
    b = r["boxes"][m].astype(np.float64)
    T = r["lidar2ego"].astype(np.float64)
    ctr = (np.c_[b[:, :3], np.ones(len(b))] @ T.T)[:, :2]
    vel = (np.c_[b[:, 7:9], np.zeros(len(b))] @ T[:3, :3].T)[:, :2]
    out = []
    for k in range(len(b)):
        t_ = r["trajs"][m][k].astype(np.float64)
        tr = (np.c_[t_.reshape(-1, 2), np.full(72, b[k, 2]), np.ones(72)] @ T.T)[:, :2].reshape(6, 12, 2)
        out.append({"x": ctr[k, 0], "y": ctr[k, 1], "v": vel[k], "L": float(max(b[k, 3], b[k, 4])),
                    "W": float(min(b[k, 3], b[k, 4])), "tr": tr, "ts": r["traj_scores"][m][k]})
    return out


def select(fold, n, seed=0):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-trainval", dataroot=str(DATA / "nuscenes"), verbose=False)
    sd = pickle.load(open(DROOT / "runs" / "sparsedrive" / "sd_clean.pkl", "rb"))
    infos, wins = di.windows({fold})
    rng = np.random.default_rng(seed)
    out = []
    for j in rng.permutation(len(wins)):
        scene, idx = wins[j]
        fr = infos[scene][idx: idx + 10]
        ps = [di.pose_of(f) for f in fr]
        if abs((pi.yaw_of(ps[9]) - pi.yaw_of(ps[0]) + math.pi) % (2 * math.pi) - math.pi) > math.radians(15):
            continue
        if max(abs(ph.along(ps[3], ps[k])[1]) for k in range(4, 10)) > 2.0:
            continue
        s_log = np.array([0.0] + [ph.along(ps[3], ps[k])[0] for k in range(4, 10)])     # t = 0, 0.5, .., 3 s
        if s_log[-1] / 3.0 < 3.0:
            continue
        tok = fr[3]["token"]
        dets = detections(sd, tok)
        best = None
        for at in nusc.get("sample", tok)["anns"]:
            ann = nusc.get("sample_annotation", at)
            if not ann["category_name"].startswith(VEH_CATS):
                continue
            v = nusc.box_velocity(at)[:2]
            if not np.all(np.isfinite(v)) or np.hypot(*v) < 1.0:
                continue
            tr = lead_track(nusc, ann, ps[3])
            if tr is None:
                continue
            x0, y0, yw0 = tr[0]
            if not (4 <= x0 <= 40 and abs(y0) <= 1.5 and abs(yw0) <= math.radians(30)):
                continue
            if REQUIRE_DET and not any(math.hypot(d["x"] - x0, d["y"] - y0) <= 2.0 for d in dets):
                continue
            if best is None or x0 < best[1][0, 0]:
                best = (ann, tr)
        if best is None:
            continue
        ann, tr = best
        L, W = ann["size"][1], ann["size"][0]
        t = np.arange(7) * 0.5

        def overlap(a_):
            s = s_log + 0.5 * a_ * t ** 2
            ov = s + FRONT0 - (tr[:, 0] - L / 2)
            lat = np.abs(tr[:, 1]) <= (EGO_W + W) / 2
            return ov, lat
        ov0, lat0 = overlap(0.0)
        if (ov0 > 0)[lat0].any():                         # the logged path itself touches the lead box: annotation noise
            continue
        a_star = None
        for a_ in A_GRID:
            ov, lat = overlap(a_)
            hit = [(ov[2 * h] >= OV_MIN) and lat[2 * h] and (tr[2 * h, 0] - L / 2 <= X_MAX) for h in (1, 2, 3)]
            if any(hit):
                a_star, h_c = a_, 1 + hit.index(True)
                break
        if a_star is None:
            continue
        out.append({"scene": scene, "idx": idx, "token": tok, "a": float(a_star), "h_contact": h_c,
                    "lead_x0": float(tr[0, 0]), "lead_speed": float(np.hypot(*nusc.box_velocity(ann["token"])[:2])),
                    "v_ego": float(s_log[-1] / 3.0), "s_log": s_log, "lead_track": tr, "lead_LW": (L, W),
                    "dets": dets,
                    "lead_det_err": float(min([math.hypot(d["x"] - tr[0, 0], d["y"] - tr[0, 1]) for d in dets],
                                              default=np.inf))})
        if len(out) == n:
            break
    return infos, out


def object_veto(dets, P0, Pc_h, h):
    """the registered layer's objects (moving detections at the best-scored mode) vs the candidate's ego box at h, both in
    the t0 frame."""
    em = ego_box_in(P0, Pc_h)
    for d in dets:
        if math.hypot(*d["v"]) < 1.0:
            continue
        px, py = d["tr"][int(np.argmax(d["ts"])), 2 * h - 1]
        yaw = math.atan2(d["v"][1], d["v"][0])
        c, s_ = math.cos(yaw), math.sin(yaw)
        u = (pi.GX - px) * c + (pi.GY - py) * s_
        w = -(pi.GX - px) * s_ + (pi.GY - py) * c
        if ((np.abs(u) <= d["L"] / 2) & (np.abs(w) <= d["W"] / 2) & em).any():
            return True
    return False


def run(a):
    import pandas as pd
    infos, sel = select(a.fold, a.n)
    print(f"{len(sel)} windows", flush=True)
    net = pi.Dome()
    rows = []
    for b0 in range(0, len(sel), a.bs):
        batch = sel[b0: b0 + a.bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s in batch]
        logged = [[di.pose_of(f) for f in fr] for fr in frs]
        cand = [lp[:4] + [shift_pose(lp[k], lp[3], 0.5 * s["a"] * (0.5 * (k - 3)) ** 2) for k in range(4, 10)]
                for s, lp in zip(batch, logged)]
        refp = [lp[:4] + ref_poses(lp, 4.0) for lp in logged]
        still = [[lp[3]] * 10 for lp in logged]
        occ = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for s, fr in zip(batch, frs)]
        neutral = [np.stack([to_frame(o[k], lp[k], lp[3]) for k in range(4)]) for o, lp in zip(occ, logged)]
        F = {"self_c": net(occ, cand, seed=b0), "self_l": net(occ, logged, seed=b0),
             "swap": net(occ, refp, seed=b0), "blind": net(neutral, still, seed=b0)}
        for i, (s, lp, cp, rp) in enumerate(zip(batch, logged, cand, refp)):
            for h in (1, 2, 3):
                row = {"token": s["token"], "scene": s["scene"], "h": h, "a": s["a"], "h_contact": s["h_contact"],
                       "lead_x0": s["lead_x0"], "lead_speed": s["lead_speed"], "v_ego": s["v_ego"]}
                for name, poses in (("collide", cp), ("logged", lp)):
                    Ph = poses[3 + 2 * h]
                    row[f"{name}_self"] = float(F["self_c" if name == "collide" else "self_l"][i][h]
                                                [ego_box_in(Ph, Ph)].max())
                    row[f"{name}_swap"] = float(F["swap"][i][h][ego_box_in(rp[3 + 2 * h], Ph)].max())
                    row[f"{name}_blind"] = float(F["blind"][i][h][ego_box_in(lp[3], Ph)].max())
                    row[f"{name}_objects"] = object_veto(s["dets"], lp[3], Ph, h)
                rows.append(row)
        print(f"{b0 + len(batch)}/{len(sel)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    summary(df)


def summary(df, tau=0.2):
    v = df.groupby("token").agg(**{f"{c}_{j}": (f"{c}_{j}", (lambda x: bool(x.any())) if j == "objects" else
                                                (lambda x: bool((x >= tau).any())))
                                   for c in ("collide", "logged") for j in ("self", "swap", "blind", "objects")})
    for c in ("collide", "logged"):
        for j in ("swap", "blind"):
            v[f"{c}_{j}+objects"] = v[f"{c}_{j}"] | v[f"{c}_objects"]
    judges = ["self", "swap", "blind", "objects", "swap+objects", "blind+objects"]
    print(f"{len(v)} real-agent hazards; veto at any of +1/+2/+3 s")
    for j in judges:
        print(f"  {j:14s} recall (collide) {v[f'collide_{j}'].mean():.3f} | false alarm (logged) {v[f'logged_{j}'].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("fold", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "select":
        infos, sel = select(a.fold, a.n)
        import pandas as pd
        d = pd.DataFrame([{k: s[k] for k in ("scene", "a", "h_contact", "lead_x0", "lead_speed", "v_ego")} for s in sel])
        print(f"{len(d)} windows, {d.scene.nunique()} scenes")
        print(d.describe().round(2).to_string())
        print("objects alone on these windows (CPU):", end=" ")
        rec = fa = 0
        for s in sel:
            fr = infos[s["scene"]][s["idx"]: s["idx"] + 10]
            lp = [di.pose_of(f) for f in fr]
            cp = lp[:4] + [shift_pose(lp[k], lp[3], 0.5 * s["a"] * (0.5 * (k - 3)) ** 2) for k in range(4, 10)]
            rec += any(object_veto(s["dets"], lp[3], cp[3 + 2 * h], h) for h in (1, 2, 3))
            fa += any(object_veto(s["dets"], lp[3], lp[3 + 2 * h], h) for h in (1, 2, 3))
        print(f"recall {rec / len(sel):.3f}, false alarm {fa / len(sel):.3f}")
    else:
        run(a)


if __name__ == "__main__":
    main()
