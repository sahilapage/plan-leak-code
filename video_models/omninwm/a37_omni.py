"""A37 OmniNWM arm (rented GPU): a frozen 4D world model (OmniNWM, panoramic RGB + depth + semantics,
trajectory-conditioned through per-frame camera ray maps), conditioned on counterfactual ego plans. Everything except
the ego plan is OmniNWM's own released inference path (configs/inference/infer.py, 33 frames at 12 Hz, 448 x 800,
6 cameras, 50 steps); the plan enters exactly where the logged trajectory does: the per-frame relative ego pose
(_calculate_relative_poses) from which the dataset builds the ray maps and traj_gt.
Plans (t = 12 Hz frame times from the 12 Hz info timestamps, t0 frame = the window's keyframe):
  logged     the recorded poses
  collide    the recorded poses shifted along the t0 heading by a* t^2 / 2 (a* from src/a37_windows.py)
  pmA        past-only: straight along the t0 heading from the past speed v0 with constant acceleration A (stops at 0)
  dA         (A37b dose) the recorded poses shifted along the t0 heading by A t^2 / 2, never moving backwards
  swK / wsK  (A37d ray-map schedule) candidate = collide; the first K denoising steps see the ray maps of the past-only
             plan pm-1.5 (sw) or of collide (ws), the remaining steps the other one. Normal plans use the stock denoiser.
  h-logged / h-collide / s-collide  (A37f motion history) the plan is the logged one through frame HIST-1 (0.33 s), then
             logged (h-logged) or collide's extra accel a* (t - t_HIST-1)^2 / 2 (h-collide, s-collide). h- plans see the
             REAL frames 0..HIST-1 (2 latent frames): cond mask on both latents + RGB-latent replacement at every
             denoising step (noised to the current level, exact at the end); s-collide = stock single-frame control.
Per (window, plan, seed) saves <out>/<token>/<plan>_s<seed>/: front3.npz (generated metric depth float16 and semantic
ids uint8 of the three front cameras, 33 frames), rel_poses.npy (the 33 conditioning poses), reward.json (OmniNWM's
released calculate_rewards on this generation with traj_gt = the plan, as shipped), rgb.mp4 (6-camera panorama),
vis.mp4 (OmniNWM's own composite: RGB, depth, semantics, plan + reward), rgb_keyframes.jpg (frames 0/12/24/32), timing. Resumable (skips finished dirs).
Usage (pod): cd /workspace/src/OmniNWM && torchrun --nproc-per-node 1 /workspace/drive/src/a37_omni.py \
    /workspace/pack/windows.json /workspace/out_omni logged,collide,pm-1.5,pm0,pm+1.5 0 [windows.txt]"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CFG = "configs/inference/infer.py"
FAKE = bool(__import__("os").environ.get("A37_FAKE"))      # smoke-test the post-processing on random tensors
DIT = "/workspace/pretrained/omninwm/ckpt/dit"          # the loader appends /model
HIST, HIST_LAT = 5, 2                                   # A37f: 5 real frames = 2 latent frames of the causal VAE


def make_cond(stock, HI):
    """prepare_inference_condition that also marks / encodes the first HIST_LAT latents (real frames 0..HIST-1)."""
    def cond(z, batch=None, causal=True, model_ae=None):
        masks, mz = stock(z, batch=batch, causal=causal, model_ae=model_ae)
        if HI:
            for i in range(z.shape[0]):
                masks[i, :, :HIST_LAT] = 1
                mz[i, :, :HIST_LAT] = model_ae.encode(batch["video"][i, :, :HIST].unsqueeze(0)).squeeze(0)
            HI["ref"] = mz[:, :, :HIST_LAT].clone()
        return masks, mz
    return cond


def make_denoise(SM, SW, HI):
    """I2VDenoiser.denoise with (A37d) the ray maps of steps < K swapped for SW['alt'] (sw) or of steps >= K (ws) and
    (A37f) the RGB latents of the history frames replaced by the real ones noised to the current level."""
    def denoise(self, model, **kwargs):
        img, timesteps = kwargs.pop("img"), kwargs.pop("timesteps")
        patch_size = kwargs.pop("patch_size", 2)
        masks, masked_ref = kwargs.pop("masks"), kwargs.pop("masked_ref")
        kwargs["cond"] = SM.pack(torch.cat((masks, masks * masked_ref), dim=1), patch_size=patch_size)
        kwargs.pop("sigma_min")
        rm = kwargs.pop("raymap")
        if HI:
            B, C, T, H, W = masked_ref.shape
            full = torch.zeros(B, img.shape[-1] // patch_size ** 2, T, H, W, device=img.device, dtype=img.dtype)
            m = torch.zeros_like(full)
            full[:, :C, :HIST_LAT] = HI["ref"].to(img.dtype)
            m[:, :C, :HIST_LAT] = 1
            g = torch.Generator(device=img.device).manual_seed(HI["seed"])
            x0, mt = SM.pack(full, patch_size=patch_size), SM.pack(m, patch_size=patch_size)
            eps = torch.randn(x0.shape, generator=g, device=img.device, dtype=torch.float32).to(img.dtype)
        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            if HI:
                img = mt * ((1 - t_curr) * x0 + t_curr * eps) + (1 - mt) * img
            alt = bool(SW) and ((i < SW["k"]) == SW["alt_first"])
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = model(img=img.detach(), raymap=SW["alt"] if alt else rm, **kwargs, timesteps=t_vec)
            img = img + (t_next - t_curr) * pred
        if HI:
            img = mt * x0 + (1 - mt) * img
        return img
    return denoise


def plan_mats(L, t, name, a_star, v0):
    """33 x 4 x 4 relative ego poses (t0 ego frame) for a plan; L = logged."""
    M = np.array(L, float).copy()
    if name in ("logged", "h-logged"):
        return M
    if name in ("h-collide", "s-collide"):                   # A37f: logged through the history frames, then collide
        M[:, 0, 3] += 0.5 * a_star * np.maximum(t - t[HIST - 1], 0.0) ** 2
        return M
    if name == "collide" or name[:2] in ("sw", "ws"):
        M[:, 0, 3] += 0.5 * a_star * t ** 2
        return M
    if name.startswith("d"):                               # A37b dose: logged path + constant extra accel A (never reverses)
        M[:, 0, 3] = np.maximum.accumulate(np.maximum(M[:, 0, 3] + 0.5 * float(name[1:]) * t ** 2, 0.0))
        return M
    A = float(name[2:])
    x = v0 * t + 0.5 * A * t ** 2
    if A < 0:
        ts = v0 / -A
        x = np.where(t > ts, v0 * ts + 0.5 * A * ts ** 2, x)
    P = np.repeat(np.eye(4)[None], len(t), 0)
    P[:, 0, 3] = x
    return P


@torch.inference_mode()
def main(windows, out, plans, seeds, wfile=None):
    import torch.distributed as dist
    from colossalai.utils import set_seed
    from omninwm.registry import DATASETS, build_module
    from omninwm.utils.cai import get_booster, init_inference_environment
    from omninwm.utils.config import parse_alias, parse_configs
    from omninwm.datasets.utils import save_sample
    from omninwm.utils.inference import (calculate_rewards, concat_6_views_pt, normalize, process_depth,
                                         process_multi_modal, process_seg)
    from omninwm.utils.misc import log_cuda_max_memory, to_torch_dtype
    from omninwm.utils.sampling import SamplingOption, prepare_api, prepare_models, sanitize_sampling_option
    from PIL import Image

    torch.set_grad_enabled(False)
    sys.argv = [sys.argv[0], CFG]
    cfg = parse_alias(parse_configs())
    cfg.model["from_pretrained"] = DIT
    cfg.plugin_config["sp_size"] = 1
    cfg.plugin_config["enable_sequence_parallelism"] = False
    cfg["calculate_reward"] = True
    device, dtype = "cuda", to_torch_dtype(cfg.get("dtype", "bf16"))
    init_inference_environment()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    booster = get_booster(cfg)
    booster_ae = get_booster(cfg, ae=True)
    ds = build_module(cfg.dataset, DATASETS)
    tok2idx = {inf["token"]: i for i, inf in enumerate(ds.data_infos)}
    REPOST = bool(os.environ.get("A37_REPOST"))
    if REPOST or FAKE:                                             # post-processing only: no diffusion model on the GPU
        model = model_ae = model_occ = model_vla = model_vla_proc = None
        booster = booster_ae = None
    else:
        model, model_ae, model_occ, model_vla, model_vla_proc = prepare_models(cfg, device, dtype)
    if booster:
        model = booster.boost(model=model)[0].unwrap()
    if booster_ae:
        model_ae = booster_ae.boost(model=model_ae)[0].unwrap()
    log_cuda_max_memory("build model")
    api_fn = None if (REPOST or FAKE) else prepare_api(model, model_ae, model_occ, model_vla, model_vla_proc)
    from concurrent.futures import ThreadPoolExecutor
    pool, pending = ThreadPoolExecutor(max_workers=1), None   # post-processing overlaps the next generation
    SW, HI = {}, {}                                             # A37d ray-map schedule / A37f history (empty = stock)
    if any(p[:2] in ("sw", "ws") or p.startswith("h-") for p in plans):
        import omninwm.utils.sampling as SM
        SM.I2VDenoiser.denoise = make_denoise(SM, SW, HI)
        SM.prepare_inference_condition = make_cond(SM.prepare_inference_condition, HI)

    def post_safe(x, batch, d, M, t, t_gen, pname, sd, t1, tag):
        try:
            post(x, batch, d, M, t, t_gen, pname, sd, t1)
            (d / "raw.pt").unlink()
            print(f"{tag} posted, total {time.time() - t1:.0f} s", flush=True)
        except Exception as e:                             # keep raw.pt; re-post offline (A37_REPOST=1)
            import traceback
            traceback.print_exc()
            print("POST FAILED, raw kept:", d, repr(e), flush=True)
    so = dict(cfg.sampling_option)
    so["num_steps"] = int(os.environ.get("A37_STEPS", so["num_steps"]))   # 50 shipped; budget runs set 25 (registered)
    opt = sanitize_sampling_option(SamplingOption(**so))
    n = opt.num_frames
    orig_rel = ds._calculate_relative_poses

    def post(x, batch, d, M, t, t_gen, pname, sd, t1):
        x = {k: x[k].to(device) for k in ("rgb_video", "depth_video", "seg_video")}
        n_ = x["rgb_video"].shape[2]
        rgb = concat_6_views_pt(normalize(x["rgb_video"].clone().float())).float().cpu()   # (C, T, 2H, 3W) in [0, 1]
        depth = process_depth(x["depth_video"].float(), ds.max_depth)[:, 0]               # (views, T, H, W) metric
        _, seg = process_seg(x["seg_video"].float(), ds.seg_color_map, devices=device)    # (views, T, H, W) ids
        np.savez_compressed(d / "front3.npz", depth=depth[:3].numpy().astype(np.float16),
                            seg=seg[:3].cpu().numpy().astype(np.uint8),
                            cam2ego=batch["ori_cam2ego_all"][0, :3].float().cpu().numpy(),
                            cam_k=batch["cam_k"][0, :3].float().cpu().numpy())
        np.save(d / "rel_poses.npy", M)
        save_sample(rgb.clone(), save_path=str(d / "rgb"), fps=12, normalize=False, verbose=False)
        fr = rgb[:, [0, 12, 24, n_ - 1]].permute(1, 2, 3, 0).numpy()
        Image.fromarray((np.concatenate(list(fr), 0) * 255).clip(0, 255).astype(np.uint8)).save(
            d / "rgb_keyframes.jpg", quality=90)
        rew = calculate_rewards(cfg, depth.clone()[:, None], seg, None, batch)
        try:                                               # OmniNWM's own composite video (cosmetic; never fatal)
            vis = process_multi_modal(cfg=cfg, x={k: v.float().cpu() for k, v in x.items()}, batch=batch, dataset=ds,
                                      devices=device)[0]
            save_sample(vis.clone(), save_path=str(d / "vis"), fps=12, normalize=False, verbose=False)
        except Exception as e:
            print("vis failed (non-fatal):", repr(e), flush=True)
        json.dump({"reward_released": [float(r) for r in rew], "t": np.asarray(t).tolist(), "t_gen_s": t_gen,
                   "num_steps": opt.num_steps, "t_total_s": time.time() - t1, "plan": pname, "seed": sd},
                  open(d / "reward.json", "w"))

    if REPOST:                                             # finish generations whose post-processing failed
        for rp in sorted(Path(out).glob("*/*/raw.pt")):
            R = torch.load(rp, weights_only=False)
            tok, (pname, sd) = rp.parent.parent.name, (rp.parent.name.split("_s")[0], 0)
            w = next(w for w in json.load(open(windows)) if w["token"] == tok)
            s = tok2idx[tok]
            ds._calculate_relative_poses = lambda data, frame_idx, num_frames, data_infos: R["M"][frame_idx]
            ds.data = [list(range(s, s + n))]
            batch = torch.utils.data.default_collate([ds.get_data_info(0, n, opt.height, opt.width)])
            ds._calculate_relative_poses = orig_rel
            batch["traj_gt"] = R["traj"]
            post(R["x"], batch, rp.parent, R["M"], R["t"], R["t_gen"], pname, sd, time.time())
            rp.unlink()
            print("reposted", rp.parent, flush=True)
        return

    W = json.load(open(windows))
    if wfile:
        keep = open(wfile).read().split()
        W = sorted([w for w in W if w["token"] in keep], key=lambda w: keep.index(w["token"]))
    out = Path(out)
    t_start, n_done = time.time(), 0
    for w in W:
        if w["token"] not in tok2idx:
            print("SKIP not in 12 Hz infos", w["token"], flush=True)
            continue
        s = tok2idx[w["token"]]
        infos = ds.data_infos[s:s + n]
        ts = np.array([f["timestamp"] for f in infos], float)
        t = (ts - ts[0]) / 1e6
        if len(infos) < n or np.diff(t).max() > 0.15 or not infos[0].get("is_key_frame", False):
            print("SKIP clip", w["token"], len(infos), float(np.diff(t).max()) if len(t) > 1 else -1, flush=True)
            continue
        L = np.stack([orig_rel(f, k, n, infos) for k, f in enumerate(infos)])
        for pname in plans:
            M = plan_mats(L, t, pname, w["a_star"], w["v0_past"])
            for sd in seeds:
                d = out / w["token"] / f"{pname}_s{sd}" if opt.num_steps == 50 else \
                    out / w["token"] / f"{pname}_s{sd}_n{opt.num_steps}"
                if (d / "reward.json").exists() or (d / "raw.pt").exists():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                ds._calculate_relative_poses = lambda data, frame_idx, num_frames, data_infos: M[frame_idx]
                ds.data = [list(range(s, s + n))]
                ret = ds.get_data_info(0, n, opt.height, opt.width)
                ds._calculate_relative_poses = orig_rel
                batch = torch.utils.data.default_collate([ret])
                SW.clear()
                HI.clear()
                if pname.startswith("h-"):
                    HI["seed"] = 1000 + sd
                if pname[:2] in ("sw", "ws"):                  # A37d: ray maps of the past-only plan pm-1.5
                    from einops import rearrange
                    Ma = plan_mats(L, t, "pm-1.5", w["a_star"], w["v0_past"])
                    ds._calculate_relative_poses = lambda data, frame_idx, num_frames, data_infos: Ma[frame_idx]
                    ds.data = [list(range(s, s + n))]
                    ra = torch.utils.data.default_collate([ds.get_data_info(0, n, opt.height, opt.width)])["ray_map"]
                    ds._calculate_relative_poses = orig_rel
                    if cfg.get("is_multi_view", False):
                        ra = rearrange(ra, "b n ... -> (b n) ...")
                    SW.update(alt=ra.to(device, dtype), k=int(pname[2:]), alt_first=pname.startswith("sw"))
                t1 = time.time()
                if FAKE:                                   # post-processing smoke test, no diffusion
                    x = {k: torch.rand(len(cfg.view_order), 3, n, opt.height, opt.width) * 2 - 1
                         for k in ("rgb_video", "depth_video", "seg_video")}
                    traj = batch["traj_gt"].float().numpy()
                else:
                    set_seed(sd)
                    x, traj = api_fn(opt, cfg=cfg, seed=sd, patch_size=cfg.get("patch_size", 2),
                                     channel=cfg["model"]["in_channels"], num_samples=len(cfg.view_order), dataset=ds,
                                     **{k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()})
                t_gen = time.time() - t1
                batch["traj_gt"] = traj
                raw = {k: x[k].to("cpu", torch.float16) for k in ("rgb_video", "depth_video", "seg_video")}
                torch.save({"x": raw, "traj": traj, "M": M, "t": t, "t_gen": t_gen}, d / "raw.pt")   # never lose a gen
                del x
                if pending is not None:
                    pending.result()                       # at most one post-processing in flight
                n_done += 1
                tag = f"[{n_done}] {w['scene']} {w['token'][:8]} {pname} s{sd}"
                pending = pool.submit(post_safe, raw, batch, d, M, t, t_gen, pname, sd, t1, tag)
                print(f"{tag}: gen {t_gen:.0f} s, elapsed {(time.time() - t_start) / 3600:.2f} h", flush=True)
                if n_done == 1:
                    log_cuda_max_memory("first generation")
    if pending is not None:
        pending.result()
    pool.shutdown()
    dist.barrier()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2].split(","), [int(x) for x in a[3].split(",")], a[4] if len(a) > 4 else None)
