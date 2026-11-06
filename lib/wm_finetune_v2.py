"""Confirmatory fine-tuning of SWTC (pre-registered arms FT-real / FT-mask / FT-full; prereg/prereg.md section 4).
(GPU, swtc env; run from third_party/SparseWorld-TC)

Differences from the pilot (wm_finetune.py, kept unchanged for the record):
- targets v2 (make_occ_targets_v2.py): Occ3D occupancy (all agents) for real frames; original-scene Occ3D with the
  replaced / dropped agents removed and the adversary's conditioning box added for Adv frames; Occ3D mask_camera weights.
- FT-mask: the adversary is removed from BOTH loss terms at every moment: its target voxels (conditioning box +0.5 m)
  are set to free, and predicted points inside that box are dropped before point matching / classification
  (masked_loss_single replicates SparseWorldHead.loss_single with only this filter added).
- stopping rule: every 1,000 frames, vehicle forecast recall at +3 s on 200 held-out REAL frames (share of Occ3D vehicle
  voxels with a predicted vehicle voxel within one voxel, imagining along the logged ego motion, no augmentation); the
  run stops if it is > 3 points below the frozen model at two consecutive checkpoints.
- seeds: --seed sets the data order and torch RNG.
Frozen: image backbone (ResNet-50) + FPN neck. Trained: the spatiotemporal decoder head incl. image-feature sampling.
"""
import argparse
import json
import math
import os
import pickle
import queue
import random
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from swtc_verifier import SWTC, Config, build_dataset, build_model, collate, load_checkpoint, online_features, \
    scatter
from make_occ_targets import voxelize
from common import DATA

VEH = (2, 3, 4, 5, 6, 7, 9, 10)


def make_dataset(infos, root, augment):
    cfg = Config.fromfile(str(SWTC / "configs" / "sw-tc.py"))
    test = cfg.data.test
    test.ann_file, test.data_root, test.test_mode = infos, root, True
    pipe = []
    for p in cfg.test_pipeline:
        if p["type"] == "LoadOccFromFile":
            p = dict(type="SetFuturePoses")
        elif p["type"] == "LoadMultiViewImageFromMultiSweeps":
            p = dict(p, force_offline=True)
        elif p["type"] == "RandomTransformImage":
            p = dict(p, training=augment)
        pipe.append(p)
    test.pipeline = pipe
    ds = build_dataset(test)
    ds.data_infos = sorted(ds.data_infos, key=lambda e: (e["scene_name"], e["timestamp"]))
    return cfg, ds


def inside_boxes(pts, boxes):
    """pts (N, 3) tensor, boxes (n, 7) [x, y, z, w, l, h, yaw] -> (N,) bool, inside any box."""
    if boxes is None or len(boxes) == 0:
        return torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
    b = torch.as_tensor(boxes, dtype=pts.dtype, device=pts.device)
    dx, dy = pts[:, None, 0] - b[None, :, 0], pts[:, None, 1] - b[None, :, 1]
    c, s = torch.cos(b[None, :, 6]), torch.sin(b[None, :, 6])
    u, v = dx * c + dy * s, -dx * s + dy * c
    inside = (u.abs() <= b[None, :, 4] / 2) & (v.abs() <= b[None, :, 3] / 2) & \
        ((pts[:, None, 2] - b[None, :, 2]).abs() <= b[None, :, 5] / 2)
    return inside.any(1)


def existence_loss(refine_pts_list, cls_scores_list, gt_points_list, gt_labels_list, radius, sigma, excl_w=0.0, far=3.0):
    """Mode-seeking existence loss (2026-10-03). For every dynamic-object instance in a frame's ground truth (connected
    component of its dynamic-class GT points in the 0.4 m BEV grid, >= 5 points), require that SOME predicted points within
    `radius` metres of it carry confident dynamic-class probability: score = mean of the top-k of w_j * q_j over predicted
    points j, with w_j = sigmoid((radius - d_j) / sigma) (d_j = BEV distance to the instance's nearest GT point), q_j = max
    sigmoid score over dynamic classes, k = half the instance's GT point count; loss = -log(score). Under positional
    uncertainty a voxel-wise target is minimised by smearing (mean of the futures, below threshold); a max-type existence
    target is minimised by committing to one confident blob near the truth (winner-takes-all / multiple-choice learning)."""
    from scipy.ndimage import distance_transform_edt, label as cc_label
    losses, xl = [], []
    for P, S, G, L in zip(refine_pts_list, cls_scores_list, gt_points_list, gt_labels_list):
        q_all = S[:, list(VEH)].sigmoid().max(-1).values
        dyn = torch.zeros_like(L, dtype=torch.bool)
        for c in VEH:
            dyn |= L == c
        if len(P) == 0:
            continue
        g = G[dyn][:, :2]
        ij = ((g + 40.0) / 0.4).long().clamp(0, 199).cpu().numpy()
        bev = np.zeros((200, 200), bool)
        bev[ij[:, 0], ij[:, 1]] = True
        if excl_w > 0:                    # exclusivity: no confident dynamic mass > `far` m from any real dynamic object
            dist = distance_transform_edt(~bev) * 0.4 if bev.any() else np.full((200, 200), 99.0)
            pij = ((P[:, :2] + 40.0) / 0.4).long().clamp(0, 199)
            dfar = torch.as_tensor(dist, device=P.device)[pij[:, 0], pij[:, 1]] > far
            if dfar.any():            # hard negatives: the k most confident far points (k = GT dynamic points, >= 20)
                qf = q_all[dfar]
                kx = min(len(qf), max(20, int(dyn.sum())))
                xl.append(excl_w * -torch.log(1 - torch.topk(qf, kx).values.clamp(max=0.999)).mean())
        if dyn.sum() < 5:
            continue
        lab, n = cc_label(bev, structure=np.ones((3, 3), bool))
        comp = torch.as_tensor(lab[ij[:, 0], ij[:, 1]], device=G.device)
        q = q_all
        for c in range(1, n + 1):
            gc = g[comp == c]
            if len(gc) < 5:
                continue
            lo, hi = gc.min(0).values - radius - 3 * sigma, gc.max(0).values + radius + 3 * sigma
            near = ((P[:, :2] >= lo) & (P[:, :2] <= hi)).all(-1)
            k = max(3, len(gc) // 2)
            if near.sum() == 0:                       # nothing predicted anywhere near: maximal loss, no gradient
                losses.append(P.new_tensor(-math.log(1e-4)))
                continue
            d = torch.cdist(P[near][:, :2].float(), gc.float()).min(-1).values
            w = torch.sigmoid((radius - d) / sigma)
            sq = w * q[near]
            score = torch.topk(sq, min(k, len(sq))).values.sum() / k
            losses.append(-torch.log(score + 1e-4))
    out = [torch.stack(v).mean() for v in (losses, xl) if v]
    return sum(out) if out else None


def masked_loss_single(self, cls_scores, refine_pts, gt_points_list, gt_masks_list, gt_labels_list):
    """SparseWorldHead.loss_single with predicted points inside self._ignore[i] removed (FT-mask)."""
    from mmdet.core import multi_apply
    from models.bbox.utils import decode_points
    num_imgs = cls_scores.size(0)
    cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
    refine_pts = decode_points(refine_pts.reshape(num_imgs, -1, 3), self.pc_range)
    cls_scores_list, refine_pts_list = [], []
    for i in range(num_imgs):
        ign = self._ignore[i] if getattr(self, "_ignore", None) is not None else None
        keep = ~inside_boxes(refine_pts[i], ign)
        cls_scores_list.append(cls_scores[i][keep])
        refine_pts_list.append(refine_pts[i][keep])
    (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights, gt_pts_weights) = multi_apply(
        self._get_target_single, refine_pts_list, gt_points_list, gt_masks_list, gt_labels_list)
    gt_paired_pts, pred_paired_pts = [], []
    for i in range(num_imgs):
        gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
        pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])
    cls_scores = torch.cat(cls_scores_list)
    labels = torch.cat(labels_list)
    cls_weights = torch.cat(cls_weights)
    gt_pts = torch.cat(gt_points_list)
    gt_paired_pts = torch.cat(gt_paired_pts)
    gt_pts_weights = torch.cat(gt_pts_weights)
    dyn_w = getattr(self, "_dyn_w", 1.0)
    if dyn_w != 1.0:                                  # permanence: every dynamic-object GT point counts dyn_w times
        gl = torch.cat(gt_labels_list)
        is_dyn = torch.zeros_like(gl, dtype=torch.bool)
        for c in VEH:
            is_dyn |= gl == c
        gt_pts_weights = gt_pts_weights * torch.where(is_dyn, dyn_w, 1.0).to(gt_pts_weights.dtype)
    pred_pts = torch.cat(refine_pts_list)
    pred_paired_pts = torch.cat(pred_paired_pts)
    loss_cls = self.loss_cls(cls_scores, labels, weight=cls_weights, avg_factor=cls_scores.shape[0])
    loss_pts = pred_pts.new_tensor(0)
    loss_pts += self.loss_pts(gt_pts, gt_paired_pts, weight=gt_pts_weights[..., None], avg_factor=gt_pts.shape[0])
    loss_pts += self.loss_pts(pred_pts, pred_paired_pts, avg_factor=pred_pts.shape[0])
    ex_w = getattr(self, "_exist_w", 0.0)
    if ex_w > 0:
        le = existence_loss(refine_pts_list, cls_scores_list, gt_points_list, gt_labels_list,
                            getattr(self, "_exist_r", 2.0), getattr(self, "_exist_s", 0.5), getattr(self, "_excl_w", 0.0))
        if le is not None:
            loss_pts = loss_pts + ex_w * le
    return loss_cls, loss_pts


class TargetsV2:
    def __init__(self, split):
        self.split = split
        self.tokens = set(json.load(open(DATA / f"occ_targets_v2_{split}_index.json")))
        self.drive = {t: e["drive"] for t, e in pickle.load(open(DATA / f"occ_targets_{split}.pkl", "rb")).items()}

    def load(self, tok, mask_adversary):
        z = np.load(DATA / f"occ_targets_v2_{self.split}" / f"{tok}.npz")
        sem, mask, T, ign = z["sem"].copy(), z["mask"], z["fut2cur"], z["ignore"]
        ignore = []
        for k in range(sem.shape[0]):
            bx = ign[:, k][np.abs(ign[:, k]).sum(-1) > 0] if len(ign) else np.zeros((0, 7), np.float32)
            if mask_adversary and len(bx):
                vl, _ = voxelize(np.c_[bx, np.full(len(bx), 4)])
                sem[k, vl[:, 0], vl[:, 1], vl[:, 2]] = 17
            ignore.append(bx if mask_adversary else np.zeros((0, 7), np.float32))
        return sem, mask, T, ignore


class Frames(Dataset):
    def __init__(self, items, dsets, targets, mask_adversary):
        self.items, self.dsets, self.targets, self.mask = items, dsets, targets, mask_adversary

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        split, i, tok = self.items[n]
        sem, mask, T, ign = self.targets[split].load(tok, self.mask and split == "adv")
        return self.dsets[split][i], torch.from_numpy(sem), torch.from_numpy(mask), torch.from_numpy(T), ign


def collate_one(batch):
    (d, sem, mask, T, ign), = batch
    return collate([d], samples_per_gpu=1), sem, mask, T, ign


def head_outputs(net, data, T, grad):
    data = scatter(data, [0])[0]
    img, img_metas, fut_list = data["img"][0], data["img_metas"][0], data["fut_list"][0]
    net.memory, net.queue = {}, queue.Queue()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        feats, img_metas = online_features(net, img, img_metas)
    feats = [f.float() for f in feats]
    fut2cur = [T[k][None].cuda() for k in range(T.shape[0])]
    with torch.set_grad_enabled(grad):
        outs = net.pts_bbox_head(list(feats), img_metas, fut2cur, fut_list)
    return outs, img_metas


def step_loss(net, data, sem, mask, T, ign):
    outs, _ = head_outputs(net, data, T, True)
    net.pts_bbox_head._ignore = ign
    losses = net.pts_bbox_head.loss(sem.cuda().long(), mask.cuda().bool(), outs)
    net.pts_bbox_head._ignore = None
    return sum(v for v in losses.values())


@torch.no_grad()
def teacher_targets(teacher, data, sem, T):
    """teacher-anchored targets: the frozen model's own prediction for every NON-dynamic class (no drift on the static
    scene / perception), with the TRUE dynamic voxels of every agent (Occ3D) pasted in; teacher-predicted dynamic voxels
    are dropped (free) so only real agents are taught."""
    outs, metas = head_outputs(teacher, data, T, False)
    occ = teacher.pts_bbox_head.get_occ(outs, metas[0])
    new = torch.full_like(sem, 17)
    for k in range(sem.shape[0]):
        loc, cls = occ[k]["occ_loc"], np.asarray(occ[k]["sem_pred"])
        keep = ~np.isin(cls, VEH)
        if keep.any():
            l = torch.as_tensor(loc[keep], dtype=torch.long)
            new[k, l[:, 0], l[:, 1], l[:, 2]] = torch.as_tensor(cls[keep], dtype=new.dtype)
        dyn = torch.zeros_like(sem[k], dtype=torch.bool)
        for c in VEH:
            dyn |= sem[k] == c
        new[k][dyn] = sem[k][dyn]
    return new


@torch.no_grad()
def vehicle_recall(net, loader):
    """share of Occ3D vehicle voxels at +3 s with a predicted vehicle voxel within one voxel."""
    net.pts_bbox_head.eval()
    hit, tot = 0, 0
    for d, sem, mask, T, ign in loader:
        outs, img_metas = head_outputs(net, d, T, False)
        occ = net.pts_bbox_head.get_occ(outs, img_metas[0])[3]
        pred = torch.zeros((200, 200, 16), device="cuda")
        v = occ["occ_loc"][np.isin(occ["sem_pred"], VEH)]
        if len(v):
            pred[v[:, 0], v[:, 1], v[:, 2]] = 1
        pred = F.max_pool3d(pred[None, None], 3, stride=1, padding=1)[0, 0].bool().cpu().numpy()
        tgt = np.isin(sem[3].numpy(), VEH)
        hit += int((pred & tgt).sum())
        tot += int(tgt.sum())
    net.pts_bbox_head.train()
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["real", "mask", "full"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--check-every", type=int, default=1000)
    ap.add_argument("--check-frames", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher-anchor", action="store_true",
                    help="targets = frozen teacher's own prediction for non-dynamic classes + true dynamic voxels")
    ap.add_argument("--dyn-weight", type=float, default=1.0,
                    help="weight of dynamic-object GT points in the GT->prediction point term (permanence loss)")
    ap.add_argument("--exist-weight", type=float, default=0.0, help="weight of the mode-seeking existence loss (0 = off)")
    ap.add_argument("--exist-radius", type=float, default=2.0, help="existence tolerance radius (m)")
    ap.add_argument("--exist-sigma", type=float, default=0.5)
    ap.add_argument("--excl-weight", type=float, default=0.0,
                    help="exclusivity: relative weight of -log(1-q) on predicted points > 3 m from any real dynamic object")
    ap.add_argument("--check-from-train", action="store_true",
                    help="stop-rule frames from 20%% of the TRAINING fold's clusters (excluded from training), not the held-out fold")
    args = ap.parse_args()
    random.seed(1000 + 10 * args.seed + args.fold)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    folds = json.load(open(DATA / "dpo" / "folds.json"))
    dsets, dsets_eval, targets, train, held_real = {}, {}, {}, [], []
    for split in ("clean", "adv"):
        root = str(DATA / ("nuscenes" if split == "clean" else "advnusc/advnusc")) + "/"
        cfg, dsets[split] = make_dataset(str(DATA / f"swtc_infos_{split}.pkl"), root, True)
        _, dsets_eval[split] = make_dataset(str(DATA / f"swtc_infos_{split}.pkl"), root, False)
        targets[split] = TargetsV2(split)
        pos = {e["token"]: i for i, e in enumerate(dsets[split].data_infos)}
        items = [(split, pos[t], t) for t in sorted(targets[split].tokens)
                 if t in pos and targets[split].drive.get(t) in folds]
        if split == "clean":
            held_real = [x for x in items if folds[targets[split].drive[x[2]]] == args.fold]
        if args.arm != "real" or split == "clean":
            train += [x for x in items if folds[targets[split].drive[x[2]]] != args.fold]
    if args.check_from_train:
        drv = sorted({targets["clean"].drive[x[2]] for x in train if x[0] == "clean"})
        val_drv = set(random.Random(0).sample(drv, max(1, len(drv) // 5)))
        held_real = [x for x in train if x[0] == "clean" and targets["clean"].drive[x[2]] in val_drv]
        train = [x for x in train if targets[x[0]].drive[x[2]] not in val_drv]
    check = random.Random(0).sample(held_real, min(args.check_frames, len(held_real)))
    check_loader = DataLoader(Frames(check[: 8 if args.smoke else None], dsets_eval, targets, False), batch_size=1,
                              num_workers=6, collate_fn=collate_one)
    print(f"arm {args.arm} fold {args.fold} seed {args.seed}: {len(train)} training frames "
          f"({sum(x[0] == 'adv' for x in train)} attacked), {args.steps} steps, stop-rule frames {len(check)}", flush=True)

    model = build_model(cfg.model)
    load_checkpoint(model, args.weights, map_location="cpu", strict=False)
    net = model.cuda()
    net.pts_bbox_head.loss_single = types.MethodType(masked_loss_single, net.pts_bbox_head)
    net.pts_bbox_head._dyn_w = args.dyn_weight
    net.pts_bbox_head._exist_w, net.pts_bbox_head._exist_r, net.pts_bbox_head._exist_s = \
        args.exist_weight, args.exist_radius, args.exist_sigma
    net.pts_bbox_head._excl_w = args.excl_weight
    teacher = None
    if args.teacher_anchor:
        teacher = build_model(cfg.model)
        load_checkpoint(teacher, args.weights, map_location="cpu", strict=False)
        teacher = teacher.cuda().eval()
        teacher.requires_grad_(False)
    for m in (net.img_backbone, net.img_neck):
        m.requires_grad_(False)
    net.train()
    net.img_backbone.eval()
    net.img_neck.eval()
    head = [(n, p) for n, p in net.pts_bbox_head.named_parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in head if "sampling_offset" not in n], "lr": args.lr},
              {"params": [p for n, p in head if "sampling_offset" in n], "lr": args.lr * 0.1}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lr = [g["lr"] for g in groups]
    n_upd = args.steps // args.accum

    def set_lr(u):
        w = 100
        f = (1 / 3 + (2 / 3) * u / w) if u < w else 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * (u - w) / max(1, n_upd - w)))
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * f

    rng, order = random.Random(1000 + 10 * args.seed + args.fold), []
    while len(order) < args.steps:
        order += rng.sample(train, len(train))
    state_path = str(args.out).replace(".pth", "_state.pth")
    n0, u = 0, 0
    try:
        st = torch.load(state_path, map_location="cpu")
        net.pts_bbox_head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        n0, u, log = st["n"], st["u"], st["log"]
        print(f"RESUMED from {state_path} at frame {n0}", flush=True)
    except FileNotFoundError:
        r0 = vehicle_recall(net, check_loader)
        log = {"args": vars(args), "frozen_recall": r0, "checks": [], "train": [], "below": 0, "stopped": False}
        print(f"CHECK frozen vehicle recall @3s {r0:.4f}", flush=True)
    loader = DataLoader(Frames(order[n0: args.steps], dsets, targets, args.arm == "mask"), batch_size=1,
                        shuffle=False, num_workers=8, collate_fn=collate_one, prefetch_factor=4)
    t0, run = time.time(), []
    set_lr(u)
    for n, (d, sem, mask, T, ign) in enumerate(loader, start=n0):
        if teacher is not None:
            sem = teacher_targets(teacher, d, sem, T)
        loss = step_loss(net, d, sem, mask, T, ign)
        (loss / args.accum).backward()
        run.append(float(loss))
        if (n + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in head], 35)
            opt.step()
            opt.zero_grad(set_to_none=True)
            u += 1
            set_lr(u)
        if n % 50 == 0:
            print(f"{n}/{args.steps} loss {np.mean(run[-50:]):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{(time.time() - t0) / (n - n0 + 1):.2f}s/it mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G",
                  flush=True)
            log["train"].append((n, float(np.mean(run[-50:]))))
        if args.smoke and n >= n0 + 12:
            break
        if (n + 1) % args.check_every == 0:
            r = vehicle_recall(net, check_loader)
            log["checks"].append((n + 1, r))
            log["below"] = log["below"] + 1 if r < log["frozen_recall"] - 0.03 else 0
            print(f"CHECK frame {n + 1}: vehicle recall @3s {r:.4f} (frozen {log['frozen_recall']:.4f}, "
                  f"consecutive below {log['below']})", flush=True)
            torch.save({"head": net.pts_bbox_head.state_dict(), "opt": opt.state_dict(), "n": n + 1, "u": u,
                        "log": log}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
            if log["below"] >= 2:
                log["stopped"] = True
                print("STOP RULE TRIGGERED", flush=True)
                break
    torch.save({"state_dict": net.state_dict()}, args.out)
    json.dump(log, open(str(args.out).replace(".pth", ".json"), "w"), indent=1)
    print("WM_FINETUNE_V2_DONE stopped" if log["stopped"] else "WM_FINETUNE_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
