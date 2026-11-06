"""Second world model: DOME (diffusion occupancy world model) imagining along the LOGGED ego path.  (GPU)

DOME (gusongen/DOME, released checkpoints dome_latest.pth / occvae_latest.pth) forecasts 6 future Occ3D frames (0.5 s
apart, 3 s) from 4 past frames (t0 - 1.5 ... t0) of ground-truth occupancy (official Occ3D-nuScenes labels), conditioned
on the ego motion (per-frame relative poses, as in DOME's own get_meta_data). Unlike SWTC it receives PERFECT perception,
so any lost agent is lost by the forecast alone. Sampling: DDPM, 20 steps, DOME's config (guidance as configured), one
sample per keyframe with a fixed seed per keyframe. The VAE input is the 4 past frames followed by six copies of the t0
frame (DOME's sampler needs a full-length conditioning tensor and uses only the first 4; the copies avoid feeding any
future ground truth into the encoder).
Output (same layout as runs/soft/swtc_soft_<cond>): runs/soft/dome_<tag>_clean_f<folds>.npy (N, 2, 4, 200, 200) fp16 =
[obstacle, vehicle] probability (softmax over DOME's 18 classes, summed over the class group, max over the Z band) at
[t0, +1, +2, +3 s], + _tokens.json. --samples K: element-wise max over K diffusion samples (sample 0 = the K=1 output). Usage: python src/dome_infer.py <folds> [--ckpt path] [--tag frozen] [--limit N]
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DROOT, RUNS  # noqa: E402

DOME = DROOT / "third_party" / "DOME"
sys.path.insert(0, str(DOME))
VEHICLE = [3, 4, 5, 9, 10]
OBSTACLE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ZLO, ZHI = int((-0.5 + 1.0) / 0.4), int((2.0 + 1.0) / 0.4)


def get_meta_data(poses):
    """identical to DOME dataset.get_meta_data (copied: DOME's dataset module imports mmdet3d)."""
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    rel_pose = np.concatenate([rel_pose, rel_pose[-1:]], axis=0)
    xys = rel_pose[:, :3, 3][:, :2]
    rel_yaws = Rotation.from_matrix(rel_pose[:, :3, :3]).as_euler("zyx", degrees=False)[:, 0]
    rr = np.array([[0, -1], [1, 0]])
    xys = xys @ rr.T
    return {"rel_poses": xys, "rel_poses_yaws": np.concatenate([xys, rel_yaws[:, None]], axis=1)}


def pose_of(info):
    p = np.eye(4)
    p[:3, 3] = info["ego2global_translation"]
    p[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    return p


def windows(folds, scenes=None):
    d = pickle.load(open(DOME / "data" / "nuscenes_infos_val_temporal_v3_scene.pkl", "rb"))
    infos = d["infos"]
    fold_of = json.load(open(DATA / "dpo" / "folds.json"))
    out = []
    for scene, fr in infos.items():
        if fold_of.get(scene, -1) not in folds or (scenes is not None and scene not in scenes):
            continue
        for i in range(3, len(fr) - 6):
            out.append((scene, i - 3))
    return infos, out


def load_occ(scene, token):
    return np.load(DOME / "data" / "nuscenes" / "gts" / scene / token / "labels.npz")["semantics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folds")
    ap.add_argument("--ckpt", default=str(DOME / "ckpts" / "dome_latest.pth"))
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1, help="K diffusion samples per window; output = max over samples")
    ap.add_argument("--heldout-of", default=None, help="checkpoint whose saved held_out scene list restricts the windows")
    args = ap.parse_args()
    from mmengine import Config
    from mmengine.registry import MODELS
    import model  # noqa: F401  (registers DOME modules)
    from diffusion import create_diffusion
    cfg = Config.fromfile(str(DOME / "config" / "train_dome.py"))
    cfg.model.world_model.attention_mode = "math"   # identical computation to xformers (not installed); DOME's 'flash' branch mis-reshapes heads
    net = MODELS.build(cfg.model.world_model).cuda().eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    print(net.load_state_dict(ck["ema"] if "ema" in ck else ck["state_dict"], strict=False))
    vae = MODELS.build(cfg.model.vae).cuda().eval()
    print(vae.load_state_dict(torch.load(DOME / "ckpts" / "occvae_latest.pth", map_location="cpu")["state_dict"]))
    diffusion = create_diffusion(timestep_respacing=str(cfg.sample.num_sampling_steps), beta_start=cfg.schedule.beta_start,
                                 beta_end=cfg.schedule.beta_end, replace_cond_frames=cfg.replace_cond_frames,
                                 cond_frames_choices=cfg.cond_frames_choices,
                                 predict_xstart=cfg.schedule.get("predict_xstart", False))
    import os
    os.environ["eval"] = "true"
    folds = {int(f) for f in args.folds}
    scenes = None
    if args.heldout_of:
        scenes = set(torch.load(args.heldout_of, map_location="cpu", weights_only=False)["held_out"])
    infos, wins = windows(folds, scenes)
    if args.limit:
        wins = wins[: args.limit]
    pre = RUNS / "soft" / f"dome_{args.tag}_clean_f{args.folds}"
    out = np.lib.format.open_memmap(str(pre) + ".npy", "w+", np.float16, (len(wins), 2, 4, 200, 200))
    toks = []
    end = cfg.end_frame
    vh = torch.tensor(VEHICLE, device="cuda")
    ob = torch.tensor(OBSTACLE, device="cuda")
    with torch.no_grad():
        for b0 in range(0, len(wins), args.bs):
            batch = wins[b0: b0 + args.bs]
            occs, metas = [], []
            for scene, idx in batch:
                fr = infos[scene][idx: idx + end]
                past = [load_occ(scene, f["token"]) for f in fr[: cfg.mid_frame]]
                occs.append(np.stack(past + [past[-1]] * (end - cfg.mid_frame)))   # future slots = t0 (no leak)
                metas.append(get_meta_data(np.stack([pose_of(f) for f in fr])))
                toks.append(fr[cfg.mid_frame - 1]["token"])
            occ = torch.from_numpy(np.stack(occs)).long().cuda()
            bs = occ.shape[0]
            torch.manual_seed(b0)
            lat, _ = vae.forward_encoder(occ)
            lat, _, _ = vae.sample_z(lat)
            lat = lat * cfg.model.vae.scaling_factor
            lat = rearrange(lat, "(b f) c h w -> b f c h w", b=bs) if lat.dim() == 4 else rearrange(lat, "b c f h w -> b f c h w")
            w = cfg.model.vae.encoder_cfg.resolution // 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
            veh = obs = None
            for k in range(args.samples):
                if k:
                    torch.manual_seed(b0 * 1000 + k)
                z = diffusion.p_sample_loop(net, (bs, end, cfg.base_channel, w, w), None, clip_denoised=False,
                                            model_kwargs={"metas": metas}, progress=False, device="cuda",
                                            initial_cond_indices=list(range(cfg.mid_frame)), initial_cond_frames=lat)
                z = rearrange(z / cfg.model.vae.scaling_factor, "b f c h w -> b c f h w")
                logits = vae.forward_decoder(z, shapes=[[200, 200], [100, 100], [50, 50]], input_shape=[bs, end, 200, 200, cfg._dim_])
                keep = [cfg.mid_frame - 1, cfg.mid_frame + 1, cfg.mid_frame + 3, cfg.mid_frame + 5]      # t0, +1, +2, +3 s
                p = torch.softmax(logits[:, keep].float(), dim=-1)                                       # (b, 4, X, Y, Z, C)
                v = p[..., vh].sum(-1)[..., ZLO:ZHI + 1].amax(-1)
                o = p[..., ob].sum(-1)[..., ZLO:ZHI + 1].amax(-1)
                veh = v if veh is None else torch.maximum(veh, v)
                obs = o if obs is None else torch.maximum(obs, o)
                del logits, p
            out[b0: b0 + bs] = torch.stack([obs, veh], 1).half().cpu().numpy()
            if b0 % 200 == 0:
                print(f"{b0}/{len(wins)}", flush=True)
    out.flush()
    json.dump(toks, open(str(pre) + "_tokens.json", "w"))
    print("DONE", len(toks), "frames ->", pre.name)


if __name__ == "__main__":
    main()
