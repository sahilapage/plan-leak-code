"""A37 window selection (rented GPU; shared by the OmniNWM and Vista arms). Same rules as src/realhazard.py select (real
lead in the ego lane, near-straight logged path, collide = logged path + smallest constant extra acceleration) with the
horizon shortened to the video models' clips: contact (ego front >= OV_MIN = 1 m into the lead's annotated box, the A35 crash rule) at
+1.0 .. +2.5 s (OmniNWM: 33 frames at 12 Hz = 2.67 s; the Vista arm (25 frames ~ 2.0 s) uses the windows with contact
<= +2.0 s), and the lead within 25 m at t0 (OmniNWM's own reward ignores depth >= 25 m), bumper gap >= 2 m. The lead may be moving or stopped (queued traffic), and so
may the ego (any speed; a stopped ego's collide plan creeps into the car ahead); speeds are recorded. No detector requirement (the video models see pixels).
Only val scenes whose CAM_FRONT keyframe images are on disk (blob part 1, camera only) are used.
Writes <out>/windows.json (the src/vista_pack.py format + v0_past + t0 sample/timestamp) and <out>/imgs/ symlinks.
Usage (pod, omni env): python src/a37_windows.py /workspace/data/nuscenes /workspace/pack"""
import json
import math
import sys
from pathlib import Path

import numpy as np

EGO_CX, EGO_L, EGO_W = 1.3, 4.1, 1.75          # src/probe_hazard.py EGO
FRONT0 = EGO_CX + EGO_L / 2
A_GRID = np.arange(0.25, 4.01, 0.25)
OV_MIN, H_CONTACT = 1.0, (2, 3, 4, 5)            # half-second indices: +1.0 .. +2.5 s
VEH_CATS = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.construction", "vehicle.trailer", "vehicle.emergency")


def pose(ep):
    from pyquaternion import Quaternion
    P = np.eye(4)
    P[:3, :3] = Quaternion(ep["rotation"]).rotation_matrix
    P[:3, 3] = ep["translation"]
    return P


def yaw_of(P):
    return math.atan2(P[1, 0], P[0, 0])


def along(P0, P):
    y0 = yaw_of(P0)
    d = P[:2, 3] - P0[:2, 3]
    return d[0] * math.cos(y0) + d[1] * math.sin(y0), -d[0] * math.sin(y0) + d[1] * math.cos(y0)


def main(root, out):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import val as VAL
    from pyquaternion import Quaternion
    root, out = Path(root), Path(out)
    nusc = NuScenes("v1.0-trainval", dataroot=str(root), verbose=False)
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    wins, n_scene = [], 0
    for sc in nusc.scene:
        if sc["name"] not in VAL:
            continue
        s0 = nusc.get("sample", sc["first_sample_token"])
        if not (root / nusc.get("sample_data", s0["data"]["CAM_FRONT"])["filename"]).exists():
            continue
        n_scene += 1
        samples, cur = [], s0
        while cur:
            samples.append(cur)
            cur = nusc.get("sample", cur["next"]) if cur["next"] else None
        for i in range(1, len(samples) - 6):          # need 0.5 s of past (v0_past) and 3 s of future
            sds = [nusc.get("sample_data", samples[i + k]["data"]["CAM_FRONT"]) for k in range(-1, 7)]
            ps = [pose(nusc.get("ego_pose", sd["ego_pose_token"])) for sd in sds]
            P0 = ps[1]
            if abs((yaw_of(ps[-1]) - yaw_of(P0) + math.pi) % (2 * math.pi) - math.pi) > math.radians(15):
                continue
            al = [along(P0, P) for P in ps[2:]]
            if max(abs(a[1]) for a in al) > 2.0:
                continue
            s_log = np.r_[0.0, [a[0] for a in al]]                          # t = 0, 0.5, .., 3 s
            dt_past = (sds[1]["timestamp"] - sds[0]["timestamp"]) / 1e6
            v0_past = -along(P0, ps[0])[0] / dt_past
            if not (-0.5 <= v0_past <= 20.0):              # any ego speed, stopped included
                continue
            best = None
            for at in samples[i]["anns"]:
                ann = nusc.get("sample_annotation", at)
                if not ann["category_name"].startswith(VEH_CATS):
                    continue
                tr, a = [], ann
                Ti = np.linalg.inv(P0)
                for k in range(7):
                    if a is None:
                        break
                    g = Ti @ np.array([*a["translation"][:2], 0.0, 1.0])
                    yw = Quaternion(a["rotation"]).yaw_pitch_roll[0] - yaw_of(P0)
                    tr.append((g[0], g[1], (yw + math.pi) % (2 * math.pi) - math.pi))
                    a = nusc.get("sample_annotation", a["next"]) if a["next"] else None
                if len(tr) < 7:
                    continue
                tr = np.array(tr)
                x0, y0, yw0 = tr[0]
                L = ann["size"][1]
                if not (2 <= x0 - L / 2 and x0 <= 25 and abs(y0) <= 1.0 and abs(yw0) <= math.radians(30)):
                    continue
                if best is None or x0 < best[1][0, 0]:
                    best = (ann, tr)
            if best is None:
                continue
            ann, tr = best
            L, W = ann["size"][1], ann["size"][0]
            t = np.arange(7) * 0.5
            lat = np.abs(tr[:, 1]) <= (EGO_W + W) / 2
            if ((s_log + FRONT0 - (tr[:, 0] - L / 2) > 0) & lat).any():
                continue                                     # logged path touches the lead box: annotation noise
            a_star = None
            for a_ in A_GRID:
                ov = s_log + 0.5 * a_ * t ** 2 + FRONT0 - (tr[:, 0] - L / 2)
                hit = [ov[h] >= OV_MIN and lat[h] for h in H_CONTACT]
                if any(hit):
                    a_star, h_c = float(a_), H_CONTACT[hit.index(True)] / 2
                    break
            if a_star is None:
                continue
            y0 = yaw_of(P0)
            u = np.array([math.cos(y0), math.sin(y0)])
            plans = {}
            for name, ax in (("logged", 0.0), ("collide", a_star)):
                wp = []
                for k in range(1, 7):
                    Q = ps[1 + k].copy()
                    Q[:2, 3] += 0.5 * ax * (0.5 * k) ** 2 * u
                    g = np.linalg.inv(P0) @ np.array([*Q[:2, 3], 0.0, 1.0])
                    wp.append([float(g[0]), float(g[1]), float(yaw_of(Q) - y0)])
                plans[name] = wp
            frames = []
            for k in range(7):
                sd = sds[1 + k]
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                ep = nusc.get("ego_pose", sd["ego_pose_token"])
                fn = Path(sd["filename"]).name
                if not (out / "imgs" / fn).exists():
                    (out / "imgs" / fn).symlink_to(root / sd["filename"])
                frames.append({"file": fn, "t": sd["timestamp"], "ego_t": ep["translation"], "ego_q": ep["rotation"],
                               "cam_t": cs["translation"], "cam_q": cs["rotation"], "K": cs["camera_intrinsic"],
                               "sample": samples[i + k]["token"]})
            vehicles = []
            for at in samples[i]["anns"]:
                a = nusc.get("sample_annotation", at)
                if a["category_name"].startswith(VEH_CATS):
                    track, aa = [], a
                    for k in range(7):
                        track.append({"t": aa["translation"], "q": aa["rotation"], "size": aa["size"]})
                        if not aa["next"]:
                            break
                        aa = nusc.get("sample_annotation", aa["next"])
                    vehicles.append({"instance": a["instance_token"], "track": track})
            wins.append({"token": samples[i]["token"], "scene": sc["name"], "t0": sds[1]["timestamp"],
                         "frames": frames, "lead_instance": ann["instance_token"],
                         "lead_track_t0frame": tr.tolist(), "lead_LW": [L, W], "plans": plans, "a_star": a_star,
                         "h_contact": h_c, "v0_past": float(v0_past), "s_log": s_log.tolist(),
                         "lead_speed": float(np.nan_to_num(np.hypot(*nusc.box_velocity(ann["token"])[:2]))),
                         "vehicles": vehicles})
    json.dump(wins, open(out / "windows.json", "w"))
    sc = sorted({w["scene"] for w in wins})
    print(f"{n_scene} val scenes on disk; {len(wins)} windows in {len(sc)} scenes")
    for w in wins:
        print(f"  {w['scene']} {w['token'][:8]} lead_x0 {w['lead_track_t0frame'][0][0]:5.1f} v_lead {w['lead_speed']:4.1f} "
              f"v0 {w['v0_past']:4.1f} a* {w['a_star']:.2f} contact +{w['h_contact']} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
