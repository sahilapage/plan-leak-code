"""Third world model: COME's scene-centric UNet occupancy forecaster (synsin0/COME, stage 2,
work_dir/unet/unet_past2s_future3s.pth) along the LOGGED ego path.  (GPU)

Deterministic, dense forecaster: 4 past ground-truth Occ3D frames (t0 - 1.5 ... t0) are aligned into the t0 ego frame
with the logged ego poses, a 2D UNet over (time x class-embedding x height) channels predicts 6 future frames (0.5 s
apart) in the t0 frame, and each future frame is warped into that moment's ego frame with the logged pose (COME's own
recover_target_coors). Like DOME it receives PERFECT perception. We keep the softmax (COME's test-time path takes the
argmax): vehicle probability = softmax summed over Occ3D vehicle classes, warped with COME's coordinate transform
(nearest, same math, float instead of integer labels), max over the obstacle height band.
Output (same layout as dome_infer / swtc_soft): runs/soft/come_<tag>_clean_f<folds>.npy (N, 2, 4, 200, 200) fp16 =
[obstacle, vehicle] at [t0, +1, +2, +3 s] (t0 = the ground-truth input frame), + _tokens.json.
Usage: python src/come_infer.py <folds> [--tag frozen] [--limit N] [--bs 8]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DROOT, RUNS  # noqa: E402
from dome_infer import OBSTACLE, VEHICLE, ZHI, ZLO, load_occ, pose_of, windows  # noqa: E402

COME = DROOT / "third_party" / "COME"
SCENE_RANGE = (-40, -40, -1, 40, 40, 5.4)


def warp(vol, src, dst):
    """COME BaseForecaster.transform_coor_system on float volumes (Z, H, W): resample from frame src into frame dst."""
    sr = vol.new_tensor(SCENE_RANGE)
    origin, size = (sr[:3] + sr[3:]) / 2, sr[3:] - sr[:3]
    Z, H, W = vol.shape[-3:]
    x = (torch.arange(W, device=vol.device) + 0.5) / W * size[0] + sr[0]
    y = (torch.arange(H, device=vol.device) + 0.5) / H * size[1] + sr[1]
    z = (torch.arange(Z, device=vol.device) + 0.5) / Z * size[2] + sr[2]
    coors = torch.stack([x[None, None, :].expand(Z, H, W), y[None, :, None].expand(Z, H, W), z[:, None, None].expand(Z, H, W)], -1)
    c = F.pad(coors.reshape(-1, 3), (0, 1), value=1.0) @ vol.new_tensor(dst).T @ torch.inverse(vol.new_tensor(src)).T
    off = ((c[:, :3] - origin) / size * 2).reshape(1, Z, H, W, 3)
    out = F.grid_sample(vol[None, None], off, mode="nearest", align_corners=False)[0, 0]
    return out.masked_fill((off.abs() > 1).any(-1)[0], 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folds")
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    sys.path.insert(0, str(COME / "occforecasting"))
    sys.path.insert(0, str(DROOT / "third_party" / "pylibs"))      # mmcv-lite 2.2.0 (pure python), isolated
    import types
    ops = types.ModuleType("mmcv.ops")                               # CUDA ops only used by an unused focal loss
    ops.sigmoid_focal_loss = None
    sys.modules["mmcv.ops"] = ops
    mu = types.ModuleType("mmdet.utils")                             # type aliases only (unused transformer module)
    mu.ConfigType = mu.OptConfigType = dict
    sys.modules["mmdet"] = types.ModuleType("mmdet")
    sys.modules["mmdet.utils"] = mu
    from mmengine import Config
    from occforecasting.registry import MODELS
    import occforecasting  # noqa: F401  (registers COME modules)
    cfg = Config.fromfile(str(COME / "occforecasting" / "configs" / "unet" / "unet_aligned_past2s_future_3s.py"))
    net = MODELS.build(cfg.model).cuda().eval()
    ck = torch.load(COME / "work_dir" / "unet" / "unet_past2s_future3s.pth", map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    print(net.load_state_dict(sd, strict=False))
    folds = {int(f) for f in args.folds}
    infos, wins = windows(folds)
    if args.limit:
        wins = wins[: args.limit]
    pre = RUNS / "soft" / f"come_{args.tag}_clean_f{args.folds}"
    out = np.lib.format.open_memmap(str(pre) + ".npy", "w+", np.float16, (len(wins), 2, 4, 200, 200))
    toks = []
    vh = torch.tensor(VEHICLE, device="cuda")
    ob = torch.tensor(OBSTACLE, device="cuda")
    with torch.no_grad():
        for b0 in range(0, len(wins), args.bs):
            batch = wins[b0: b0 + args.bs]
            src, smetas, poses_all = [], [], []
            for scene, idx in batch:
                fr = infos[scene][idx: idx + 10]
                src.append(np.stack([load_occ(scene, f["token"]) for f in fr[:4]]).transpose(0, 3, 2, 1))   # (T, Z, Y, X)
                poses = [pose_of(f) for f in fr]
                poses_all.append(poses)
                smetas.append({"ego2global": poses[:4]})
                toks.append(fr[3]["token"])
            inputs = {"source_occs": torch.from_numpy(np.stack(src).astype(np.int64)), "source_metas": smetas,
                      "metas": [{"scene_range": SCENE_RANGE}] * len(batch)}
            x, _ = net.pre_process(inputs)
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
            logits = net.occ_sem_conv(net.temporal_shared_branch(x)).reshape(B, tT, -1, tZ, H, W)[..., :tH, :tW]
            p = logits.float().softmax(dim=2)                                                     # (B, 6, C, Z, Y, X)
            pv, po = p[:, :, vh].sum(2), p[:, :, ob].sum(2)                                       # (B, 6, Z, Y, X)
            for b in range(B):
                poses = poses_all[b]
                res = np.zeros((2, 4, 200, 200), np.float32)
                cur = torch.from_numpy(src[b][3]).cuda()                                          # t0 ground truth input
                res[0, 0] = torch.isin(cur, ob)[ZLO:ZHI + 1].any(0).float().T.cpu().numpy()
                res[1, 0] = torch.isin(cur, vh)[ZLO:ZHI + 1].any(0).float().T.cpu().numpy()
                for k, f in enumerate((1, 3, 5)):                                                 # +1, +2, +3 s
                    for ch, vol in ((0, po[b, f]), (1, pv[b, f])):
                        w_ = warp(vol, poses[3], poses[4 + f])
                        res[ch, k + 1] = w_[ZLO:ZHI + 1].amax(0).T.cpu().numpy()
                out[b0 + b] = res.astype(np.float16)
            if b0 % 400 == 0:
                print(f"{b0}/{len(wins)}", flush=True)
    out.flush()
    json.dump(toks, open(str(pre) + "_tokens.json", "w"))
    print("DONE", len(toks), "frames ->", pre.name)


if __name__ == "__main__":
    main()
