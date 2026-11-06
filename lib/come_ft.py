"""Fine-tune COME (released checkpoint, deterministic forecasting branch) with COUNTERFACTUAL MOTION AUGMENTATION.
The intervention test (A14) showed that the models keep a car they saw only if it does not move. This fine-tune teaches
the missing skill directly: in a share of training samples, real-shape cars (voxels of real, well-observed Occ3D cars from
TRAINING scenes) are inserted into all 10 frames of the window -- the 4 input frames and the 6 target frames -- moving at a
constant velocity (or parked), so the targets contain the car's continued path. Everything else is COME's own training:
its loss (class-weighted cross-entropy; targets aligned into the current frame by the model), its input format.
Control: the same run with --p-aug 0 (plain fine-tuning on the same data for the same number of steps).
Data: nuScenes TRAIN scenes only (Occ3D GT); the validation folds A/B are never seen.
Usage: python src/come_ft.py <out_dir> [--p-aug 0.7] [--iters 6000] [--bs 4] [--lr 1e-4] [--amp none] [--w-motion 1] [--seed 0] [--save-every 2000]
"""
import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_insert as pi  # noqa: E402

TRAIN_PKL = di.DOME / "data" / "nuscenes_infos_train_ours.pkl"
FT_ALL = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]          # 4 inputs + 6 targets (2 Hz)
EGO = (np.abs(pi.GX - 1.3) <= 2.05 + 1.0) & (np.abs(pi.GY) <= 0.87 + 1.0)  # ego box + 1 m margin, in every frame


def train_windows():
    d = pickle.load(open(TRAIN_PKL, "rb"))
    infos = d["infos"] if "infos" in d else d
    wins = [(s, i) for s, fr in infos.items() for i in range(0, len(fr) - 9)]
    return infos, wins


def train_bank(infos, n, seed):
    """real cars from TRAINING scenes (same filters as the test's bank, which comes from validation scenes)."""
    from common import DATA
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    nu = NuScenes("v1.0-trainval", str(DATA / "nuscenes"), verbose=False)
    rng = np.random.default_rng(seed)
    scenes = list(infos)
    bank = []
    for s in rng.permutation(len(scenes)):
        fr = infos[scenes[s]]
        f = fr[rng.integers(0, len(fr))]
        P = di.pose_of(f)
        occ = None
        for at in nu.get("sample", f["token"])["anns"]:
            a = nu.get("sample_annotation", at)
            L, W = a["size"][1], a["size"][0]
            if a["category_name"] != "vehicle.car" or not (3.8 <= L <= 5.2) or a["visibility_token"] != "4" or a["num_lidar_pts"] < 50:
                continue
            xy, yaw = np.array(a["translation"][:2]), Quaternion(a["rotation"]).yaw_pitch_roll[0]
            m, c = pi.footprint(P, xy, yaw, dims=(L, W))
            if not m.any() or np.hypot(*c) > 25:
                continue
            occ = di.load_occ(scenes[s], f["token"]) if occ is None else occ
            if np.isin(occ[m], di.VEHICLE).any(1).mean() < 0.7:
                continue
            g = pi.ground_top(occ, m)
            yl = yaw - pi.yaw_of(P)
            u_, w_, dz_, cl_ = [], [], [], []
            for ii, jj in np.argwhere(m):
                for zz in np.where(np.isin(occ[ii, jj], di.VEHICLE))[0]:
                    dx, dy = pi.CTR[ii] - c[0], pi.CTR[jj] - c[1]
                    u_.append(dx * math.cos(yl) + dy * math.sin(yl))
                    w_.append(-dx * math.sin(yl) + dy * math.cos(yl))
                    dz_.append(zz - g)
                    cl_.append(occ[ii, jj, zz])
            bank.append({"u": np.array(u_), "w": np.array(w_), "dz": np.array(dz_), "cls": np.array(cl_, np.uint8),
                         "dims": (L, W)})
            break
        if len(bank) == n:
            break
    return bank


def sample_motion(rng, P0):
    """a random car: start (ahead, lateral), heading, speed; 25 % parked, 25 % of moving ones oncoming."""
    yaw0 = pi.yaw_of(P0)
    ahead, lat = rng.uniform(-15, 25), rng.choice([-7.2, -3.6, 3.6, 7.2]) + rng.normal(0, 0.3)
    v = 0.0 if rng.random() < 0.25 else rng.uniform(2.0, 14.0)
    head = yaw0 + rng.normal(0, math.radians(4)) + (math.pi if (v > 0 and rng.random() < 0.25) else 0.0)
    xy0 = P0[:2, 3] + ahead * np.array([math.cos(yaw0), math.sin(yaw0)]) + lat * np.array([-math.sin(yaw0), math.cos(yaw0)])
    return xy0, head, v


def augment(occs, poses, bank, rng, kmax=3, tries=24):
    """insert up to kmax valid moving / parked cars into all 10 frames; returns new occs and the number inserted."""
    n_ins = 0
    k_target = rng.integers(1, kmax + 1)
    for _ in range(tries):
        if n_ins == k_target:
            break
        pa = bank[rng.integers(0, len(bank))]
        xy0, head, v = sample_motion(rng, poses[3])
        e = np.array([math.cos(head), math.sin(head)])
        ok = True
        for k, t in enumerate(FT_ALL):
            m, c = pi.footprint(poses[k], xy0 + v * t * e, head, dims=pa["dims"])
            if not m.any() or abs(c[0]) > 38 or abs(c[1]) > 38 or (m & EGO).any():   # never on top of the ego itself
                ok = False
                break
            col = occs[k][m]
            if np.isin(col, pi.OBST).any() or (col == pi.DRIVABLE).any(1).mean() < 0.8:
                ok = False
                break
            mg, _ = pi.footprint(poses[k], xy0 + v * t * e, head, dims=pa["dims"], grow=0.6)
            if np.isin(occs[k][mg], pi.OBST).any():
                ok = False
                break
        if not ok:
            continue
        occs = [pi.insert(occs[k], poses[k], xy0 + v * t * e, head, patch=pa) for k, t in enumerate(FT_ALL)]
        n_ins += 1
    return occs, n_ins


class Data(torch.utils.data.Dataset):
    def __init__(self, infos, wins, bank, p_aug, seed):
        self.infos, self.wins, self.bank, self.p_aug, self.seed = infos, wins, bank, p_aug, seed

    def __len__(self):
        return 10 ** 9

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        s, j = self.wins[rng.integers(0, len(self.wins))]
        fr = self.infos[s][j: j + 10]
        poses = [di.pose_of(f) for f in fr]
        occs = [di.load_occ(s, f["token"]) for f in fr]
        n_ins = 0
        if rng.random() < self.p_aug:
            occs, n_ins = augment(occs, poses, self.bank, rng)
        occ = np.stack(occs).transpose(0, 3, 2, 1).astype(np.int64)           # (10, Z, Y, X) as COME's dataset
        return {"src": occ[:4], "tgt": occ[4:], "poses": np.stack(poses), "n_ins": n_ins}


def build_come():
    import types
    import come_infer as ci
    from common import DROOT
    sys.path.insert(0, str(ci.COME / "occforecasting"))
    sys.path.insert(0, str(DROOT / "third_party" / "pylibs"))
    ops = types.ModuleType("mmcv.ops")
    ops.sigmoid_focal_loss = None
    sys.modules["mmcv.ops"] = ops
    mu = types.ModuleType("mmdet.utils")
    mu.ConfigType = mu.OptConfigType = dict
    sys.modules["mmdet"] = types.ModuleType("mmdet")
    sys.modules["mmdet.utils"] = mu
    from mmengine import Config
    from occforecasting.registry import MODELS
    import occforecasting  # noqa: F401
    cfg = Config.fromfile(str(ci.COME / "occforecasting" / "configs" / "unet" / "unet_aligned_past2s_future_3s.py"))
    net = MODELS.build(cfg.model).cuda()
    ck = torch.load(ci.COME / "work_dir" / "unet" / "unet_past2s_future3s.pth", map_location="cpu", weights_only=False)
    print(net.load_state_dict(ck.get("state_dict", ck.get("model", ck)), strict=False), flush=True)
    return net, ci


VEH_IDS = torch.tensor([3, 4, 5, 9, 10])


def weighted_loss(net, inputs, w_motion):
    """COME's own objective (class-weighted cross-entropy over the aligned targets, normalised by the number of non-free
    target voxels) with one change: vehicle voxels a vehicle has MOVED INTO -- vehicle in the target frame but not in the
    same voxel of the current frame (targets are aligned into the current frame, so parked vehicles stay put and are not
    affected) -- are weighted w_motion. Label-free; covers real moving agents and inserted cars alike."""
    import torch.nn.functional as F
    x, targets = net.pre_process(inputs)
    B, T, C, Z, H, W = x.shape
    x = x.reshape(B, -1, H, W)
    enc = []
    for e in net.encoder:
        x = e(x)
        enc.append(x)
    for i in reversed(range(len(net.decoder))):
        x = net.decoder[i](enc[i], x)
    tT, tZ, tH, tW = net.target_seq_size
    x = net.temporal_stem(x).reshape(B * tT, -1, H, W)
    sem = net.occ_sem_conv(net.temporal_shared_branch(x)).reshape(B, tT, -1, tZ, H, W)[..., :tH, :tW]
    logits = sem.permute(0, 1, 3, 4, 5, 2).reshape(-1, sem.shape[2]).float()
    tgt = targets.reshape(-1)
    cw = torch.tensor(net.occ_losses["ce_loss"].class_weight, device=logits.device, dtype=torch.float32)
    ce = F.cross_entropy(logits, tgt, weight=cw, ignore_index=-1, reduction="none")
    vid = VEH_IDS.to(targets.device)
    cur = inputs["source_occs"][:, -1].to(targets.device)                     # current frame (Z, H, W), already aligned
    moved = torch.isin(targets, vid) & ~torch.isin(cur, vid)[:, None]
    w = torch.ones_like(ce)
    w[moved.reshape(-1)] = w_motion
    avg = ((tgt != -1) & (tgt != net.num_classes - 1)).sum().clamp(min=1)
    return (ce * w).sum() / avg, float(moved.float().mean())


def freeze_bn(net):
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--p-aug", type=float, default=0.7)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--amp", default="none", choices=["bf16", "none"])
    ap.add_argument("--w-motion", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    infos, wins = train_windows()
    bank = train_bank(infos, 120, seed=7)
    print(f"{len(wins)} training windows from {len(infos)} scenes; bank {len(bank)} cars; p_aug {a.p_aug}", flush=True)
    net, ci = build_come()
    net.train()
    freeze_bn(net)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    dl = torch.utils.data.DataLoader(Data(infos, wins, bank, a.p_aug, a.seed), batch_size=a.bs, num_workers=a.workers,
                                     persistent_workers=True, prefetch_factor=4)
    it, t0, run, n_aug = 0, time.time(), 0.0, 0
    for batch in dl:
        inputs = {"source_occs": batch["src"], "target_occs": batch["tgt"],
                  "source_metas": [{"ego2global": list(p[:4].numpy())} for p in batch["poses"]],
                  "target_metas": [{"ego2global": list(p[4:].numpy())} for p in batch["poses"]],
                  "metas": [{"scene_range": ci.SCENE_RANGE}] * len(batch["src"])}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.amp == "bf16")):
            loss, frac = weighted_loss(net, inputs, a.w_motion)
            if it == 0:                                                       # must equal COME's own loss at w = 1
                with torch.no_grad():
                    ref = sum(net(inputs).values())
                    l1, _ = weighted_loss(net, inputs, 1.0)
                print(f"loss check: COME {float(ref):.5f} | reimplemented (w=1) {float(l1):.5f} | moved-into voxel share "
                      f"{frac:.2e}", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        it += 1
        run = 0.98 * run + 0.02 * float(loss) if it > 1 else float(loss)
        n_aug += int((batch["n_ins"] > 0).sum())
        if it % 50 == 0:
            print(f"it {it} loss {run:.4f} aug-samples {n_aug} {(time.time() - t0) / it:.2f} s/it", flush=True)
        if it % a.save_every == 0 or it == a.iters:
            torch.save({"state_dict": net.state_dict(), "iter": it, "args": vars(a)}, out / f"it{it}.pth")
            print("saved", out / f"it{it}.pth", flush=True)
        if it == a.iters:
            break
    print("TRAIN DONE")


if __name__ == "__main__":
    main()
