"""Learned world-model verifier: SparseWorld-TC (CVPR'26) imagines future 3D occupancy for each candidate path.

Run inside the `swtc` env with cwd = third_party/SparseWorld-TC. For each frame:
  1. camera features are extracted once (6 cameras x 8 frames over the last 2 s),
  2. for every candidate path, the future ego poses at 1/2/3 s (from waypoints 2/4/6) condition the
     occupancy head, which predicts a 200x200x16 semantic grid (0.4 m voxels) for each future moment,
  3. a candidate is predicted to collide if an obstacle voxel falls inside our car's footprint
     (the grid is centred on the ego pose of that future moment).
Output jsonl rows: {"id", "collide": [K bools], "clearance": [K floats], "occ_ms": time per frame}
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SWTC = HERE.parent / "third_party" / "SparseWorld-TC"
sys.path.insert(0, str(SWTC))

from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel, collate, scatter  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet.datasets.builder import PIPELINES  # noqa: E402

import importlib  # noqa: E402

importlib.import_module("models")
importlib.import_module("loaders")
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402

OCC_NAMES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
             'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
             'manmade', 'vegetation']
OBSTACLE = [OCC_NAMES.index(n) for n in ('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                                         'motorcycle', 'pedestrian', 'trailer', 'truck', 'others')]
VOX, RANGE_MIN, Z_MIN = 0.4, -40.0, -1.0
EGO_L, EGO_W, EGO_CENTER_FWD = 4.084, 1.73, 1.3
FUT_WP = (1, 3, 5)          # waypoint indices for 1 s, 2 s, 3 s
Z_BAND = (-0.5, 2.0)        # heights (m) that count as an obstacle for our car


@PIPELINES.register_module()
class SetFuturePoses:
    """Replaces LoadOccFromFile at inference: identity poses; the real ones are set per candidate."""
    def __init__(self, future_frames=(0, 2, 4, 6)):
        self.future_frames = list(future_frames)

    def __call__(self, results):
        results["fut_list"] = self.future_frames
        results["fut2cur"] = [np.eye(4, dtype=np.float32) for _ in self.future_frames]
        return results


def pose_mat(x, y, yaw):
    T = np.eye(4, dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3] = x, y
    return T


def cand_poses(wp):
    """(6,2) waypoints -> fut2cur list [I, T(1s), T(2s), T(3s)] with heading from the local direction."""
    wp = np.asarray(wp, float)
    prev = np.vstack([[0, 0], wp[:-1]])
    out = [np.eye(4, dtype=np.float32)]
    last_yaw = 0.0
    for i in FUT_WP:
        d = wp[i] - prev[i]
        yaw = math.atan2(d[1], d[0]) if np.hypot(*d) > 0.2 else last_yaw
        last_yaw = yaw
        out.append(pose_mat(wp[i, 0], wp[i, 1], yaw))
    return out


def footprint_mask():
    """Boolean (200,200) mask of voxels under our car in the (future) ego frame."""
    c = RANGE_MIN + VOX * (np.arange(200) + 0.5)
    X, Y = np.meshgrid(c, c, indexing="ij")
    return (np.abs(X - EGO_CENTER_FWD) <= EGO_L / 2) & (np.abs(Y) <= EGO_W / 2)


FOOT = footprint_mask()
FOOT_IDX = np.argwhere(FOOT)
FOOT_XY = RANGE_MIN + VOX * (FOOT_IDX + 0.5)


def obstacle_bev(r):
    """Packed (200*200/8,) bitmap of obstacle voxels (in the height band) from one get_occ result."""
    zlo, zhi = int((Z_BAND[0] - Z_MIN) / VOX), int((Z_BAND[1] - Z_MIN) / VOX)
    loc, sem = r["occ_loc"], r["sem_pred"]
    keep = np.isin(sem, OBSTACLE) & (loc[:, 2] >= zlo) & (loc[:, 2] <= zhi)
    m = np.zeros((200, 200), bool)
    m[loc[keep][:, 0], loc[keep][:, 1]] = True
    return np.packbits(m.reshape(-1))


def judge(occ_list):
    """occ_list: get_occ results for [t0, 1s, 2s, 3s]. Returns (collide, clearance in m)."""
    zlo = int((Z_BAND[0] - Z_MIN) / VOX)
    zhi = int((Z_BAND[1] - Z_MIN) / VOX)
    collide, clearance = False, 50.0
    for r in occ_list[1:]:
        loc, sem = r["occ_loc"], r["sem_pred"]
        keep = np.isin(sem, OBSTACLE) & (loc[:, 2] >= zlo) & (loc[:, 2] <= zhi)
        pts = loc[keep][:, :2]
        if len(pts) == 0:
            continue
        inside = FOOT[pts[:, 0], pts[:, 1]]
        if inside.any():
            collide = True
            clearance = 0.0
            break
        xy = RANGE_MIN + VOX * (pts + 0.5)
        # distance from each obstacle voxel to the nearest footprint voxel centre (coarse, fine for ranking)
        d = np.sqrt(((xy[:, None, :] - FOOT_XY[None, ::7, :]) ** 2).sum(-1)).min()
        clearance = min(clearance, float(d))
    return collide, clearance


def danger_score(head, outs, margin=0.2):
    """Raw (pre-threshold) confidence that an obstacle overlaps our footprint at 1/2/3 s: max over the head's points
    that fall inside the footprint (+margin, obstacle height band) of their max obstacle-class probability.
    Uses the same point filter as get_occ (distance to the query centre < ctr_dist_thr), but no score threshold."""
    from models.bbox.utils import decode_points
    cls = outs["all_cls_scores"][-1].sigmoid()           # [FT, Q, P, 17]
    pts = decode_points(outs["all_refine_pts"][-1], head.pc_range)   # [FT, Q, P, 3], metres, that moment's ego frame
    ctr = pts.mean(dim=2, keepdim=True)
    ok = torch.norm(pts - ctr, dim=-1) < head.test_cfg.get("ctr_dist_thr", 3.0)
    inside = ((pts[..., 0] - EGO_CENTER_FWD).abs() <= EGO_L / 2 + margin) & (pts[..., 1].abs() <= EGO_W / 2 + margin) \
        & (pts[..., 2] >= Z_BAND[0]) & (pts[..., 2] <= Z_BAND[1]) & ok
    obst = cls[..., OBSTACLE].max(dim=-1).values
    d = torch.where(inside, obst, torch.zeros_like(obst)).flatten(1).max(dim=1).values   # per moment
    return float(d[1:].max())


VEHICLE = [OCC_NAMES.index(n) for n in ('bus', 'car', 'construction_vehicle', 'trailer', 'truck')]


def soft_bev(head, outs):
    """Pre-threshold BEV probability maps for every moment: (2, FT, 200, 200) = [any obstacle, vehicle], max over the
    head's points in each 0.4 m cell within the obstacle height band (same point filter as get_occ, no 0.5 cut-off)."""
    from models.bbox.utils import decode_points
    cls = outs["all_cls_scores"][-1].sigmoid()
    pts = decode_points(outs["all_refine_pts"][-1], head.pc_range)
    ctr = pts.mean(dim=2, keepdim=True)
    ok = (torch.norm(pts - ctr, dim=-1) < head.test_cfg.get("ctr_dist_thr", 3.0)) \
        & (pts[..., 2] >= Z_BAND[0]) & (pts[..., 2] <= Z_BAND[1])
    ix = ((pts[..., 0] - RANGE_MIN) / VOX).floor().long()
    iy = ((pts[..., 1] - RANGE_MIN) / VOX).floor().long()
    ok &= (ix >= 0) & (ix < 200) & (iy >= 0) & (iy < 200)
    FT = cls.shape[0]
    out = torch.zeros(2, FT, 40000, device=cls.device)
    for c, idx in enumerate((OBSTACLE, VEHICLE)):
        prob = cls[..., idx].max(dim=-1).values
        for f in range(FT):
            m = ok[f]
            out[c, f].scatter_reduce_(0, (ix[f][m] * 200 + iy[f][m]), prob[f][m], reduce="amax", include_self=True)
    return out.reshape(2, FT, 200, 200).half().cpu().numpy()


def online_features(net, img, img_metas):
    """SparseWorld.simple_test_online up to (not including) the occupancy head: per-frame feature
    extraction with the model's own memory cache, then the same reorganisation of features and metas."""
    from mmcv.runner.fp16_utils import cast_tensor_type
    net.fp16_enabled = False
    B, N, C, H, W = img.shape
    img = img.reshape(B, N // 6, 6, C, H, W)
    names = img_metas[0]["filename"]
    num_frames = len(names) // 6
    shape = (H, W, C)
    for k in ("img_shape", "ori_shape", "pad_shape"):
        img_metas[0][k] = [shape for _ in range(len(names))]
    feats_list, metas_list = [], []
    for i in range(num_frames):
        ids = list(range(i * 6, (i + 1) * 6))
        cur = [{}]
        for k, item in img_metas[0].items():
            cur[0][k] = [item[j] for j in ids] if isinstance(item, list) and len(item) == 6 * num_frames else item
        key = names[ids[0]]
        if key in net.memory:
            f = net.memory[key]
        else:
            f = net.extract_feat(img[:, i], cur)
            net.memory[key] = f
            net.queue.put(key)
            while net.queue.qsize() >= 16:
                net.memory.pop(net.queue.get())
        feats_list.append(f)
        metas_list.append(cur)
    feats = []
    for j in range(len(feats_list[0])):
        fl = torch.cat([feats_list[i][j] for i in range(len(feats_list))], dim=0)
        feats.append(fl.flatten(0, 1)[None, ...])
    metas = metas_list[0]
    for i in range(1, len(metas_list)):
        for k, v in metas_list[i][0].items():
            if isinstance(v, list):
                metas[0][k].extend(v)
    return cast_tensor_type(feats, torch.half, torch.float32), metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="VLA candidates jsonl (sample_vla.py output)")
    ap.add_argument("--infos", required=True, help="mmdet3d-style info pkl for the same frames")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default=None, help="optional file with sample tokens to process")
    ap.add_argument("--dump", default=None, help="save raw occupancy for probe paths (npz per frame) here")
    ap.add_argument("--soft-dump", default=None,
                    help="save pre-threshold BEV probability maps (obstacle, vehicle) x (t0, 1, 2, 3 s) for candidate 0 "
                         "of every target frame: <path>.npy (N, 2, 4, 200, 200) float16 + <path>_tokens.json")
    ap.add_argument("--soft-all", action="store_true",
                    help="with --soft-dump: store the obstacle-probability BEV at 1/2/3 s for EVERY candidate instead: "
                         "<path>.npy (N, K, 3, 200, 200) float16 (stage-wise attribution / PLCG)")
    ap.add_argument("--danger", action="store_true", help="also output the raw-confidence danger score per candidate")
    ap.add_argument("--dump-cands", default=None,
                    help="json {token: [candidate indices]}: save packed obstacle maps at 1/2/3 s for these paths")
    ap.add_argument("--mln-ref", default="plan", choices=("plan", "cv"),
                    help="cv: geometry-only plan conditioning (src/swtc_mlnref.py); use for models trained with it")
    args = ap.parse_args()

    cfg = Config.fromfile(str(SWTC / "configs" / "sw-tc.py"))
    test = cfg.data.test
    test.ann_file = args.infos
    test.data_root = args.data_root
    test.test_mode = True
    test.pipeline = [p if p["type"] != "LoadOccFromFile" else dict(type="SetFuturePoses")
                     for p in cfg.test_pipeline]
    dataset = build_dataset(test)
    # the dataset sorts frames by timestamp across scenes; Adv-nuSc has several scenes built from the same
    # drive (identical timestamps), so re-order by (scene, time) to keep each frame's history in its scene
    dataset.data_infos = sorted(dataset.data_infos, key=lambda e: (e["scene_name"], e["timestamp"]))

    model = build_model(cfg.model)
    load_checkpoint(model, args.weights, map_location="cpu", strict=False)
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    net = model.module
    mln, v0 = None, {}
    if args.mln_ref == "cv":
        import swtc_mlnref
        mln = swtc_mlnref.install(net)
        v0 = swtc_mlnref.past_speed(dataset.data_infos)
        print(f"geometry-only plan conditioning: past speeds for {len(v0)} frames", flush=True)

    cands = {}
    for line in open(args.samples):
        s = json.loads(line)
        cands[s["id"]] = s["cands"][: args.k]
    dump_cands = json.load(open(args.dump_cands)) if args.dump_cands else None
    wanted = set(open(args.ids).read().split()) if args.ids else (set(dump_cands) if dump_cands else None)
    cand_maps = {}  # "token:ci" -> (3, 5000) packed obstacle maps at 1/2/3 s
    targets = [i for i, info in enumerate(dataset.data_infos)
               if info["token"] in cands and (wanted is None or info["token"] in wanted)]
    if args.limit:
        targets = targets[: args.limit]
    # the model reads earlier keyframes' features from a cache filled in time order, so also run (features
    # only) every earlier frame of the same scene that precedes a target
    first = {}
    for i, info in enumerate(dataset.data_infos):
        first.setdefault(info["scene_name"], i)
    need = set()
    for i in targets:
        need.update(range(first[dataset.data_infos[i]["scene_name"]], i + 1))
    idx, targets = sorted(need), set(targets)
    print(f"{len(targets)} frames to imagine ({len(idx) - len(targets)} extra history frames)", flush=True)

    soft = None
    if args.soft_dump:
        shape = (len(targets), args.k, 3, 200, 200) if args.soft_all else (len(targets), 2, 4, 200, 200)
        soft = np.lib.format.open_memmap(args.soft_dump + ".npy", mode="w+", dtype=np.float16, shape=shape)
        soft_tokens = []
    out = open(args.out, "w")
    now_maps = {}  # token -> packed 200x200 obstacle map of the model's "now" prediction
    t_all = time.time()
    for n, i in enumerate(idx):
        t0 = time.time()
        data = scatter(collate([dataset[i]], samples_per_gpu=1), [0])[0]
        img_metas = data["img_metas"][0]
        with torch.no_grad():
            img = data["img"][0]
            feats, img_metas = online_features(net, img, img_metas)
            token = dataset.data_infos[i]["token"]
            if i not in targets:
                continue
            col, clr, dng = [], [], []
            fut_list = data["fut_list"][0]
            if mln is not None:
                mln._ref = swtc_mlnref.cv_ref(v0.get(token, 0.0))
            for ci, c in enumerate(cands[token]):
                if c is None:
                    col.append(True)
                    clr.append(0.0)
                    dng.append(1.0)
                    continue
                fut2cur = [torch.from_numpy(T)[None].cuda() for T in cand_poses(c)]
                # the head re-assigns entries of the feature list in place: give each candidate a fresh list
                outs = net.pts_bbox_head(list(feats), img_metas, fut2cur, fut_list)
                if args.danger:
                    dng.append(danger_score(net.pts_bbox_head, outs))
                if soft is not None and args.soft_all:
                    soft[len(soft_tokens), ci] = soft_bev(net.pts_bbox_head, outs)[0, 1:]
                    if ci == len(cands[token]) - 1:
                        soft_tokens.append(token)
                elif soft is not None and ci == 0:
                    soft[len(soft_tokens)] = soft_bev(net.pts_bbox_head, outs)
                    soft_tokens.append(token)
                occ = net.pts_bbox_head.get_occ(outs, img_metas[0])
                if token not in now_maps:
                    now_maps[token] = obstacle_bev(occ[0])
                if dump_cands and ci in dump_cands.get(token, ()):
                    cand_maps[f"{token}:{ci}"] = np.stack([obstacle_bev(r) for r in occ[1:]])
                cc, cl = judge(occ)
                col.append(bool(cc))
                clr.append(cl)
            if args.dump:
                probes = {"vla": cands[token][0], "brake": [[0.0, 0.0]] * 6,
                          "swerve": [[2.5 * (j + 1), 0.6 * (j + 1) ** 1.5] for j in range(6)]}
                dump = {}
                for name, c in probes.items():
                    fut2cur = [torch.from_numpy(T)[None].cuda() for T in cand_poses(c)]
                    occ = net.pts_bbox_head.get_occ(net.pts_bbox_head(list(feats), img_metas, fut2cur, fut_list),
                                                    img_metas[0])
                    for t, r in enumerate(occ):
                        dump[f"{name}_{t}_loc"], dump[f"{name}_{t}_sem"] = r["occ_loc"], r["sem_pred"]
                    dump[f"{name}_path"] = np.asarray(c, float)
                Path(args.dump).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(args.dump) / f"{token}.npz", **dump)
        row = {"id": token, "collide": col, "clearance": clr, "occ_ms": 1000 * (time.time() - t0)}
        if args.danger:
            row["danger"] = dng
        out.write(json.dumps(row) + "\n")
        if n % 50 == 0:
            print(f"{n + 1}/{len(idx)}  {time.time() - t_all:.0f}s  last frame {1000 * (time.time() - t0):.0f} ms",
                  flush=True)
    out.close()
    if soft is not None:
        soft.flush()
        json.dump(soft_tokens, open(args.soft_dump + "_tokens.json", "w"))
    if dump_cands:
        keys = sorted(cand_maps)
        np.savez_compressed(str(args.out).replace(".jsonl", "_cands.npz"), keys=np.array(keys),
                            maps=np.stack([cand_maps[k] for k in keys]) if keys else np.zeros((0, 3, 5000), np.uint8))
    toks = sorted(now_maps)
    np.savez_compressed(str(args.out).replace(".jsonl", "_now.npz"), tokens=np.array(toks),
                        bits=np.stack([now_maps[t] for t in toks]) if toks else np.zeros((0, 5000), np.uint8))
    print("SWTC_DONE", flush=True)


if __name__ == "__main__":
    main()
