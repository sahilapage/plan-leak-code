"""Counterfactual-plan consistency (CPC) fine-tuning of SparseWorld-TC: teach an action-conditioned world model that other
agents do not vanish because of the plan it is conditioned on (path clearing, A19 / real-car tests), without collision
data or labels.
Per training frame (fold != --fold; swtc env, run from third_party/SparseWorld-TC):
  with probability --p-cf the logged future poses (fut2cur at +1 / +2 / +3 s) are replaced by a COUNTERFACTUAL plan:
  x += 0.5 a t^2, y += c t^2 (current frame), a ~ U(-4, 4) m/s^2, c ~ U(-0.4, 0.4) m/s^2, heading kept;
  target = the REAL future occupancy (Occ3D targets v2) re-expressed from the logged future frame into the counterfactual
  frame (nearest voxel; cells not covered by the source grid ignored via the mask), teacher-anchored: the frozen model's
  own prediction under the SAME plan for every non-vehicle class, the true vehicle voxels pasted in (src/wm_finetune_v2.py
  teacher_targets). Real agents are taken as non-reactive to the ego plan (the safe assumption for a judge).
  otherwise: the same teacher-anchored target on the logged plan (--p-cf 0 = control arm).
--cf-mode retime (A25, randomized-plan training): the counterfactual plan's TIMING is drawn from the past only, so it
  carries no information about the future: arc length s(t) = v0 t + a t^2 / 2 (a ~ U(-4, 4), stopping at 0), v0 = the
  ego speed over the last 0.5 s (previous keyframe pose), placed along the logged path's SHAPE (straight ahead if the
  path is shorter than 0.5 m, extrapolated along its last segment), heading = path tangent, plus the same lateral c t^2.
  The default (--cf-mode offset, A24) adds a t^2 / 2 to the LOGGED timing, which still encodes the logged future.
--cf-mode cv (PF, plan-free): every frame (p_cf 1) is conditioned on the straight constant-velocity plan at the past
  speed, targets re-expressed into it: the model can only forecast agents from history; candidates are judged by
  geometry in that forecast (the reference judge).
--mln-ref cv (geometry-only plan conditioning, the leak law's fix by construction): the learned pose embedding reads
  a past-only constant-velocity pose; the true plan enters through the sampling geometry only (src/swtc_mlnref.py).
Usage: python src/swtc_cpc.py --fold 1 --weights <sw-tc-small.pth> --out <ckpt.pth> [--p-cf 0.5] [--steps 6000]
       [--cf-mode offset|retime]
"""
import argparse
import json
import math
import os
import random
import sys
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_finetune_v2 as wf  # noqa: E402
from swtc_verifier import build_model, load_checkpoint  # noqa: E402
from common import DATA  # noqa: E402

CTR = -40.0 + 0.4 * (torch.arange(200, dtype=torch.float32) + 0.5)


def counterfactual(T, rng):
    """T: (4, 4, 4) fut2cur (index 0 = now). Returns a perturbed copy (+1 / +2 / +3 s)."""
    a, c = rng.uniform(-4.0, 4.0), rng.uniform(-0.4, 0.4)
    Tc = T.clone()
    for k, t in ((1, 1.0), (2, 2.0), (3, 3.0)):
        Tc[k, 0, 3] += 0.5 * a * t * t
        Tc[k, 1, 3] += c * t * t
    return Tc


def _quat_mat(t, q):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def past_speed(infos):
    """token -> forward ego speed (m/s) over the last keyframe interval (previous pose -> current), from the past only."""
    by = {e["token"]: e for e in infos}
    out = {}
    for e in infos:
        p = by.get(e.get("prev") or "")
        if p is None:
            continue
        dt = (e["timestamp"] - p["timestamp"]) / 1e6
        if not 0.2 < dt < 1.0:
            continue
        rel = np.linalg.inv(_quat_mat(e["ego2global_translation"], e["ego2global_rotation"])) @ \
            _quat_mat(p["ego2global_translation"], p["ego2global_rotation"])
        out[e["token"]] = max(0.0, -float(rel[0, 3]) / dt)
    return out


def along(path, s):
    """(x, y, yaw) at arc length s along the polyline path (N, 2) starting at the origin."""
    seg = np.diff(path, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    if ln.sum() < 0.5:
        return s, 0.0, 0.0
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    good = np.nonzero(ln > 1e-3)[0]
    for i in good:
        if s <= cum[i + 1] or i == good[-1]:
            u = seg[i] / ln[i]
            p = path[i] + u * (s - cum[i])
            return float(p[0]), float(p[1]), math.atan2(u[1], u[0])


def retime(T, v0, rng):
    """randomized-plan counterfactual: past-only timing along the logged path shape (see the module docstring)."""
    a, c = rng.uniform(-4.0, 4.0), rng.uniform(-0.4, 0.4)
    path = np.vstack([[0.0, 0.0]] + [T[k, :2, 3].double().numpy() for k in (1, 2, 3)])
    Tc = T.clone()
    for k, t in ((1, 1.0), (2, 2.0), (3, 3.0)):
        s = v0 * t + 0.5 * a * t * t
        if a < 0 and v0 + a * t < 0:
            s = v0 * v0 / (2 * -a)
        x, y, yaw = along(path, max(0.0, s))
        R = torch.eye(4, dtype=T.dtype)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(yaw), -math.sin(yaw), math.sin(yaw), math.cos(yaw)
        R[0, 3], R[1, 3], R[2, 3] = x, y + c * t * t, T[k, 2, 3]
        Tc[k] = R
    return Tc


def cv_plan(T, v0):
    """plan-free (PF) conditioning: a straight constant-velocity plan at the past speed v0 (no future information)."""
    Tc = T.clone()
    for k, t in ((1, 1.0), (2, 2.0), (3, 3.0)):
        R = torch.eye(4, dtype=T.dtype)
        R[0, 3], R[2, 3] = v0 * t, T[k, 2, 3]
        Tc[k] = R
    return Tc


def reexpress(sem, mask, T, Tc):
    """real targets (4, 200, 200, 16) in the logged future frames -> the counterfactual frames (top-down rigid map)."""
    X, Y = torch.meshgrid(CTR, CTR, indexing="ij")
    pts = torch.stack([X, Y, torch.zeros_like(X), torch.ones_like(X)], -1).reshape(-1, 4)          # (40000, 4)
    s2, m2 = sem.clone(), mask.clone()
    for k in (1, 2, 3):
        M = torch.linalg.inv(T[k].double()) @ Tc[k].double()                                     # cf frame -> logged frame
        p = (pts.double() @ M.T)[:, :2]
        ij = torch.floor((p + 40.0) / 0.4).long()
        ok = (ij >= 0).all(1) & (ij < 200).all(1)
        ns = torch.full((40000, 16), 17, dtype=sem.dtype)
        nm = torch.zeros((40000, 16), dtype=mask.dtype)
        ns[ok] = sem[k][ij[ok, 0], ij[ok, 1]]
        nm[ok] = mask[k][ij[ok, 0], ij[ok, 1]]
        s2[k], m2[k] = ns.reshape(200, 200, 16), nm.reshape(200, 200, 16)
    return s2, m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-cf", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cf-mode", default="offset", choices=("offset", "retime", "cv"))
    ap.add_argument("--offset-lr-mult", type=float, default=0.1)
    ap.add_argument("--mln-ref", default="plan", choices=("plan", "cv"),
                    help="cv: the pose embedding reads a past-only constant-velocity pose (src/swtc_mlnref.py)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(1000 + a.seed)
    folds = json.load(open(DATA / "dpo" / "folds.json"))
    cfg, ds = wf.make_dataset(str(DATA / "swtc_infos_clean.pkl"), str(DATA / "nuscenes") + "/", True)
    targets = {"clean": wf.TargetsV2("clean")}
    pos = {e["token"]: i for i, e in enumerate(ds.data_infos)}
    train = [("clean", pos[t], t) for t in sorted(targets["clean"].tokens)
             if t in pos and targets["clean"].drive.get(t) in folds and folds[targets["clean"].drive[t]] != a.fold]
    print(f"CPC p_cf {a.p_cf} mode {a.cf_mode}: {len(train)} training frames (fold != {a.fold}), {a.steps} steps", flush=True)
    v0 = past_speed(ds.data_infos) if a.cf_mode in ("retime", "cv") or a.mln_ref == "cv" else {}
    model = build_model(cfg.model)
    load_checkpoint(model, a.weights, map_location="cpu", strict=False)
    net = model.cuda()
    mln = None
    if a.mln_ref == "cv":
        import swtc_mlnref
        mln = swtc_mlnref.install(net)
    net.pts_bbox_head.loss_single = types.MethodType(wf.masked_loss_single, net.pts_bbox_head)
    net.pts_bbox_head._dyn_w, net.pts_bbox_head._exist_w, net.pts_bbox_head._excl_w = 1.0, 0.0, 0.0
    teacher = build_model(cfg.model)
    load_checkpoint(teacher, a.weights, map_location="cpu", strict=False)
    teacher = teacher.cuda().eval()
    teacher.requires_grad_(False)
    for m in (net.img_backbone, net.img_neck):
        m.requires_grad_(False)
    net.train()
    net.img_backbone.eval()
    net.img_neck.eval()
    head = [(n, p) for n, p in net.pts_bbox_head.named_parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in head if "sampling_offset" not in n], "lr": a.lr},
              {"params": [p for n, p in head if "sampling_offset" in n], "lr": a.lr * a.offset_lr_mult}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lr = [g["lr"] for g in groups]
    n_upd = a.steps // a.accum

    def set_lr(u):
        f = (1 / 3 + (2 / 3) * u / 100) if u < 100 else 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * (u - 100) / max(1, n_upd - 100)))
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * f

    order = []
    r2 = random.Random(2000 + a.seed)
    while len(order) < a.steps:
        order += r2.sample(train, len(train))
    loader = DataLoader(wf.Frames(order[: a.steps], {"clean": ds}, targets, False), batch_size=1, shuffle=False,
                        num_workers=8, collate_fn=wf.collate_one, prefetch_factor=4)
    t0, run, n_cf, u = time.time(), [], 0, 0
    set_lr(u)
    for n, (d, sem, mask, T, ign) in enumerate(loader):
        tok = order[n][2]
        if rng.random() < a.p_cf and (a.cf_mode == "offset" or tok in v0):
            Tc = (counterfactual(T, rng) if a.cf_mode == "offset" else
                  retime(T, v0[tok], rng) if a.cf_mode == "retime" else cv_plan(T, v0[tok]))
            sem, mask = reexpress(sem, mask, T, Tc)
            T = Tc
            n_cf += 1
        sem = wf.teacher_targets(teacher, d, sem, T)
        if mln is not None:
            import swtc_mlnref
            mln._ref = swtc_mlnref.cv_ref(v0.get(tok, 0.0))
        loss = wf.step_loss(net, d, sem, mask, T, ign)
        (loss / a.accum).backward()
        run.append(float(loss))
        if (n + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in head], 35)
            opt.step()
            opt.zero_grad(set_to_none=True)
            u += 1
            set_lr(u)
        if n % 50 == 0:
            print(f"{n}/{a.steps} loss {np.mean(run[-50:]):.4f} cf {n_cf} {(time.time() - t0) / (n + 1):.2f}s/it "
                  f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if a.smoke and n >= 12:
            break
    torch.save({"state_dict": net.state_dict()}, a.out)
    print("CPC_DONE", flush=True)


if __name__ == "__main__":
    main()
