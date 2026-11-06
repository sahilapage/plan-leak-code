"""A37 'seen, then deleted' in OmniNWM (descriptive, amendment 2+): do vehicles the model was shown at t0 vanish from
its generated video, and does a plan that drives into the lead make the world model clear its path (flee or delete)?
Per window and plan in {logged, collide, pm0, pm-1.5}: each annotated vehicle whose 3D box projects >= 50 % inside one
of the front-left / front / front-right views at t0 within 30 m is followed with the detector + IoU tracker of
src/a37_control2.py from frame 0 (re-acquisition allowed for 2 missed frames). 'Expected visible until' = the last frame
at which its TRUE future box (annotation, the vehicle does not react) projected into that plan's camera (commanded
pose) still has its centre in the view, 1-40 m ahead. Vanished = the track is lost >= 4 frames (0.33 s) before that.
Reports vanish rates by plan and group (lead / other), with how far the generated lead sits from the ego at the end.
Detections are cached per (token, plan, view) in <cache_dir>.
Usage: python src/a37_vanish.py <windows.json> <out_dir> <tokens.txt> <cache_dir> <out.csv>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_control2 as C2  # noqa: E402
from a37_control import quat_R  # noqa: E402

PLANS = ("logged", "collide", "pm0", "pm-1.5")


def main(windows, out, toks, cache, dst):
    import imageio.v3 as iio
    W = {w["token"]: w for w in json.load(open(windows))}
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    det = None
    rows = []
    for tok in open(toks).read().split():
        w, d = W[tok], Path(out) / tok
        if not all((d / f"{p}_s0_n25" / "front3.npz").exists() for p in PLANS):
            continue
        t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
        f0 = w["frames"][0]
        R0, T0 = quat_R(f0["ego_q"]), np.array(f0["ego_t"])
        yaw0 = np.arctan2(R0[1, 0], R0[0, 0])
        Z = {p: np.load(d / f"{p}_s0_n25" / "front3.npz") for p in PLANS}
        M = {p: np.load(d / f"{p}_s0_n25" / "rel_poses.npy") for p in PLANS}
        vids = {}
        for veh in w["vehicles"]:
            tr = np.array([R0.T @ (np.array(b["t"]) - T0) for b in veh["track"]])
            if len(tr) < 2:
                continue
            b = veh["track"][0]
            q = b["q"]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)) - yaw0
            best = None
            for v in range(3):
                pr = C2.project(tr[0], b["size"], yaw, Z["logged"]["cam_k"][v, 0], Z["logged"]["cam2ego"][v, 0])
                if pr is not None and pr[1] >= 0.5 and np.hypot(*tr[0, :2]) <= 30 and (best is None or pr[1] > best[2]):
                    best = (v, pr[0], pr[1])
            if best is None:
                continue
            v, box0, _ = best
            # true future centre per generated frame (2 Hz annotations, linear interpolation; held after the last one)
            tt = np.arange(len(tr)) * 0.5
            ctr = np.stack([np.interp(t, tt, tr[:, i]) for i in range(3)], 1)
            lead = veh["instance"] == w["lead_instance"]
            for p in PLANS:
                key = cache / f"{tok[:8]}_{p}_v{v}.npz"
                if key.exists():
                    dets = list(np.load(key, allow_pickle=True)["d"])
                else:
                    if det is None:
                        det = C2.detector()
                    if p not in vids:
                        vids[p] = iio.imread(d / f"{p}_s0_n25" / "rgb.mp4")
                    dets = det([vids[p][k, :448, 800 * v:800 * (v + 1)] for k in range(len(t))])
                    np.savez(key, d=np.array(dets, dtype=object))
                trk = C2.track(dets, box0)
                last = -1 if trk is None else max(k for k, bb in enumerate(trk) if bb is not None)
                vis = []
                for k in range(len(t)):
                    P = np.linalg.inv(M[p][k]) @ np.array([*ctr[k], 1.0])            # ego_t(plan)
                    pc = np.linalg.inv(Z[p]["cam2ego"][v, k]) @ P
                    if 1.0 < pc[2] < 40:
                        uv = (Z[p]["cam_k"][v, k] @ pc[:3])[:2] / pc[2]
                        vis.append(0 <= uv[0] < 800 and 0 <= uv[1] < 448)
                    else:
                        vis.append(False)
                vis = np.array(vis)
                until = int(np.nonzero(vis)[0].max()) if vis.any() else -1
                rows.append({"token": tok, "scene": w["scene"], "plan": p, "lead": lead, "view": v,
                             "x0": tr[0, 0], "y0": tr[0, 1], "speed": np.hypot(*(tr[1, :2] - tr[0, :2])) / 0.5,
                             "acquired": trk is not None, "last_tracked": last, "visible_until": until,
                             "vanished": trk is not None and until - last >= 4})
        print(f"{tok[:8]} done ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(dst, index=False)
    A = df[df.acquired]
    print(f"vehicles acquired at t0: {A.groupby('plan').size().to_dict()} (of {df.groupby('plan').size().to_dict()})")
    for g, m in (("lead", A.lead), ("other", ~A.lead)):
        print(f"  {g}: " + " | ".join(f"{p} vanish {A[m & (A.plan == p)].vanished.mean():.2f} (n {int((m & (A.plan == p)).sum())})"
                                    for p in PLANS))


if __name__ == "__main__":
    main(*sys.argv[1:])
