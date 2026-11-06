"""Package the SWTC real-car windows (src/sw_realhazard.py dirs, both folds) for the Vista leak test on a rented GPU:
for every window, CAM_FRONT keyframes t0 .. t0+3 s (2 Hz; t0 = Vista's conditioning frame, later ones = real-frame
detector calibration), the camera calibration, ego poses (global) and the lead's annotated 3D box at each keyframe,
plus the logged / collide plans (t0 ego frame, 0.5 s waypoints) and all annotated vehicles at t0 (relations).
Output: <out>/imgs/<file>.jpg, <out>/windows.json. Usage (server, vla env): python src/vista_pack.py <out> <dir> ..."""
import json
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realhazard as rh  # noqa: E402


def main(out, dirs):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-trainval", dataroot=str(rh.DATA / "nuscenes"), verbose=False)
    out = Path(out)
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    wins = []
    for d in map(Path, dirs):
        T = pickle.load(open(d / "truth.pkl", "rb"))
        for tok, t in T.items():
            s = nusc.get("sample", tok)
            frames, cur = [], s
            for k in range(7):
                sd = nusc.get("sample_data", cur["data"]["CAM_FRONT"])
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                ep = nusc.get("ego_pose", sd["ego_pose_token"])
                fn = Path(sd["filename"]).name
                if not (out / "imgs" / fn).exists():
                    shutil.copy(rh.DATA / "nuscenes" / sd["filename"], out / "imgs" / fn)
                frames.append({"file": fn, "t": sd["timestamp"], "ego_t": ep["translation"], "ego_q": ep["rotation"],
                               "cam_t": cs["translation"], "cam_q": cs["rotation"], "K": cs["camera_intrinsic"],
                               "sample": cur["token"]})
                if not cur["next"]:
                    break
                cur = nusc.get("sample", cur["next"])
            vehicles = []
            for at in s["anns"]:
                a = nusc.get("sample_annotation", at)
                if a["category_name"].startswith(rh.VEH_CATS):
                    track, aa = [], a
                    for k in range(7):
                        track.append({"t": aa["translation"], "q": aa["rotation"], "size": aa["size"]})
                        if not aa["next"]:
                            break
                        aa = nusc.get("sample_annotation", aa["next"])
                    vehicles.append({"instance": a["instance_token"], "track": track})
            wins.append({"token": tok, "scene": t["scene"], "fold_dir": d.name, "frames": frames,
                         "lead_track_t0frame": [list(map(float, r)) for r in t["track"]], "lead_LW": list(t["LW"]),
                         "plans": {k: v for k, v in t["plans"].items()}, "a_star": t["a"], "vehicles": vehicles})
    json.dump(wins, open(out / "windows.json", "w"))
    print(f"packed {len(wins)} windows, {len(list((out / 'imgs').iterdir()))} images")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
