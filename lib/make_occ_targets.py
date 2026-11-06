"""Pseudo future-occupancy targets for fine-tuning SparseWorld-TC's imagination.  (CPU)

For a keyframe t the model is supervised at t, t+1 s, t+2 s, t+3 s (keyframe offsets 0/2/4/6, as in its training),
each grid in the ego frame of that moment, with fut2cur = logged ego motion (CAM_FRONT ego pose, as in the original
LoadOccFromFile). Target at moment tau =
  static : the frozen model's own perception of tau (wm_now_pass.py) with all moving-agent classes removed
  agents : the annotated 3D box at tau of every moving agent the frozen model PERCEIVED at t (an obstacle voxel
           within 0.4 m of its footprint in the now-map of t). Agents it could not see at t are left out, so
           the model is taught to keep and move what it saw, never to hallucinate unseen agents.
The same rule is used for real nuScenes and Adv-nuSc (whose generated agents carry no visibility labels).
Output pkl {token: {"fut2cur": (4,4,4), "static": [tau tokens], "boxes": [ (n,8) x,y,z,w,l,h,yaw,cls per moment ],
                     "scene", "drive", "adv_inst", "seen_adv": bool}}
"""
import argparse
import json
import pickle
import re

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from common import ADV, DATA, NUSC, RUNS

OFFSETS = (0, 2, 4, 6)
DYN_CLS = (2, 3, 4, 5, 6, 7, 9, 10)  # bicycle bus car construction_vehicle motorcycle pedestrian trailer truck
OBST = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)
CAT2CLS = [("vehicle.car", 4), ("vehicle.emergency", 4), ("vehicle.truck", 10), ("vehicle.bus", 3),
           ("vehicle.trailer", 9), ("vehicle.construction", 5), ("vehicle.motorcycle", 6), ("vehicle.bicycle", 2),
           ("human.pedestrian", 7)]
C = -40.0 + 0.4 * (np.arange(200) + 0.5)
CZ = -1.0 + 0.4 * (np.arange(16) + 0.5)


def load_now(split):
    m = np.load(RUNS / f"wm_now_{split}_meta.npz")
    return {"tokens": m["tokens"], "offsets": m["offsets"],
            "loc": np.load(RUNS / f"wm_now_{split}_loc.npy", mmap_mode="r"),
            "sem": np.load(RUNS / f"wm_now_{split}_sem.npy", mmap_mode="r")}


def cls_of(cat):
    for p, c in CAT2CLS:
        if cat.startswith(p):
            return c
    return None


def ego_pose(nusc, sample, chan):
    return nusc.get("ego_pose", nusc.get("sample_data", sample["data"][chan])["ego_pose_token"])


def fut2cur(cur, fut):
    """SparseWorld-TC loaders/pipelines/loading.py T_fut2cur."""
    Rc, Rf = Quaternion(cur["rotation"]).rotation_matrix, Quaternion(fut["rotation"]).rotation_matrix
    T = np.eye(4)
    T[:3, :3] = Rc.T @ Rf
    T[:3, 3] = Rc.T @ (np.array(fut["translation"]) - np.array(cur["translation"]))
    return T.astype(np.float32)


def box_in_ego(ann, pose):
    q = Quaternion(pose["rotation"])
    p = q.inverse.rotate(np.array(ann["translation"]) - np.array(pose["translation"]))
    yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[0] - q.yaw_pitch_roll[0]
    w, l, h = ann["size"]
    return np.array([p[0], p[1], p[2], w, l, h, (yaw + np.pi) % (2 * np.pi) - np.pi], np.float32)


def voxelize(boxes):
    """(n,8) boxes -> (m,3) voxel indices and (m,) classes of voxel centres inside each box."""
    locs, cls = [], []
    for x, y, z, w, l, h, yaw, c in boxes:
        r = np.hypot(w, l) / 2 + 0.3
        ix = np.where(np.abs(C - x) <= r)[0]
        iy = np.where(np.abs(C - y) <= r)[0]
        iz = np.where(np.abs(CZ - z) <= h / 2)[0]
        if not len(ix) or not len(iy) or not len(iz):
            continue
        X, Y = np.meshgrid(C[ix] - x, C[iy] - y, indexing="ij")
        u, v = X * np.cos(yaw) + Y * np.sin(yaw), -X * np.sin(yaw) + Y * np.cos(yaw)
        inside = (np.abs(u) <= l / 2) & (np.abs(v) <= w / 2)
        if not inside.any():  # thin agents (pedestrians): keep the voxel under the centre
            inside[np.argmin(np.abs(C[ix] - x)), np.argmin(np.abs(C[iy] - y))] = True
        a, b = np.nonzero(inside)
        g = np.stack(np.broadcast_arrays(ix[a][:, None], iy[b][:, None], iz[None, :]), -1).reshape(-1, 3)
        locs.append(g)
        cls.append(np.full(len(g), int(c), np.uint8))
    if not locs:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.uint8)
    return np.concatenate(locs), np.concatenate(cls)


def seen(box, now_bev):
    """True if the now-map has an obstacle cell within 0.4 m of the box footprint."""
    x, y, _, w, l, _, yaw, _ = box
    r = np.hypot(w, l) / 2 + 0.8
    ix, iy = np.where(np.abs(C - x) <= r)[0], np.where(np.abs(C - y) <= r)[0]
    if not len(ix) or not len(iy):
        return False
    X, Y = np.meshgrid(C[ix] - x, C[iy] - y, indexing="ij")
    u, v = X * np.cos(yaw) + Y * np.sin(yaw), -X * np.sin(yaw) + Y * np.cos(yaw)
    near = (np.abs(u) <= l / 2 + 0.4) & (np.abs(v) <= w / 2 + 0.4)
    return bool((near & now_bev[np.ix_(ix, iy)]).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["clean", "adv"], required=True)
    args = ap.parse_args()
    root = str(NUSC if args.split == "clean" else ADV / "advnusc")
    nusc = NuScenes("v1.0-trainval", root, verbose=False)
    now = load_now(args.split)
    off = now["offsets"]
    idx = {t: i for i, t in enumerate(now["tokens"])}
    # json copy of adv_meta.pkl (that pickle needs numpy 2; this runs in the py3.8 swtc env so the output loads there)
    meta = json.load(open(DATA / "gt" / "adv_meta.json")) if args.split == "adv" else {}

    def now_bev(tok):
        i = idx[tok]
        loc, sem = now["loc"][off[i]:off[i + 1]], now["sem"][off[i]:off[i + 1]]
        keep = np.isin(sem, OBST)
        m = np.zeros((200, 200), bool)
        m[loc[keep, 0], loc[keep, 1]] = True
        return m

    out, n_seen, n_all = {}, 0, 0
    for tok in now["tokens"]:
        s0 = nusc.get("sample", tok)
        chain = [s0]
        for _ in range(OFFSETS[-1]):
            nxt = chain[-1]["next"]
            if not nxt:
                break
            chain.append(nusc.get("sample", nxt))
        if len(chain) <= OFFSETS[-1] or any(chain[k]["token"] not in idx for k in OFFSETS):
            continue
        cam0, lid0 = ego_pose(nusc, s0, "CAM_FRONT"), ego_pose(nusc, s0, "LIDAR_TOP")
        bev0 = now_bev(tok)
        seen_inst = set()
        for a in s0["anns"]:
            ann = nusc.get("sample_annotation", a)
            c = cls_of(ann["category_name"])
            if c is None:
                continue
            n_all += 1
            if seen(np.append(box_in_ego(ann, lid0), c), bev0):
                seen_inst.add(ann["instance_token"])
        n_seen += len(seen_inst)
        boxes, T = [], []
        for k in OFFSETS:
            s = chain[k]
            lid = ego_pose(nusc, s, "LIDAR_TOP")
            b = []
            for a in s["anns"]:
                ann = nusc.get("sample_annotation", a)
                if ann["instance_token"] in seen_inst:
                    b.append(np.append(box_in_ego(ann, lid), cls_of(ann["category_name"])))
            boxes.append(np.array(b, np.float32).reshape(-1, 8))
            T.append(fut2cur(cam0, ego_pose(nusc, s, "CAM_FRONT")))
        scene = nusc.get("scene", s0["scene_token"])["name"]
        m = meta.get(tok, {})
        out[tok] = {"fut2cur": np.stack(T), "static": [chain[k]["token"] for k in OFFSETS], "boxes": boxes,
                    "scene": scene, "drive": re.sub(r"-obj\d+$", "", scene), "adv_inst": m.get("adv_inst"),
                    "seen_adv": m.get("adv_inst") in seen_inst}
    pickle.dump(out, open(DATA / f"occ_targets_{args.split}.pkl", "wb"))
    sa = [v["seen_adv"] for v in out.values() if v["adv_inst"]]
    print(f"{args.split}: {len(out)} training frames; moving agents perceived at t: {n_seen}/{n_all} "
          f"({100 * n_seen / max(n_all, 1):.0f}%)" + (f"; attacker perceived in {100 * np.mean(sa):.0f}%" if sa else ""))


if __name__ == "__main__":
    main()
