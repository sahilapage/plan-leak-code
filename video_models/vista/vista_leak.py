"""A37 (rented GPU): the plan leak in a video world model (Vista, NeurIPS'24; front camera, nuScenes-trained,
trajectory-conditioned). Same question as A31 / relation_leak: when only the ego plan changes, does the real lead car
keep its gap to the ego (beta = 1) or stay where the world puts it (beta = 0)?
Vista trajectory input (verified on its nuScenes_val annotations vs nuScenes ego poses, median max error 0.21 m):
4 waypoints at frames 6 / 12 / 18 / 24 of the 12 Hz CAM_FRONT stream (~0.5 / 1.0 / 1.5 / 2.0 s), each (-lateral,
forward) in metres in the t0 ego frame, prefixed by the origin; generated frame k ~ k / 12 s.
Stages:
  gen  <pack> <out> <vista_dir> <plans> <seeds> <n_steps> [<windows.txt>]
       plans: comma list of logged, collide, brake, pm-1.5, pm0, pm+1.5 (pmA = past-only constant accel A from the
       past speed, straight); writes <out>/<token>/<plan>_s<seed>/f00..f24.jpg (resumable)
       A37g motion history: h-logged / h-collide condition on the REAL frames 0-2 (t0 + the next two 12 Hz CAM_FRONT
       sweeps, Vista's own 3-frame dynamic prior, initial_cond_indices [0, 1, 2]); the plan is the log through frame 2,
       then h-collide adds a* (t - t2)^2 / 2; s-collide = the same shifted plan with the stock single-frame condition
       A37h video g-computation: g-logged / g-collide (base plan h-logged / h-collide) are rolled out in 4 rounds of
       0.5 s; round r starts at T = r / 2 s, conditions on the 3 frames at T, T + 1/12, T + 2/12 (round 0: the real
       ones, later: its own kept frames) and sees the candidate only up to the end of the frames it keeps (T + 8/12),
       continued past-only at the candidate's speed there (constant velocity); it keeps its frames 3..8.
  read <pack> <out>   detector (torchvision Faster R-CNN v2, COCO vehicles) on every generated frame; the lead and the
       static vehicles are found in frame 0 by IoU with their projected annotated boxes and tracked by IoU; distance
       by box height ratio d_f = d_0 h_0 / h_f (d_0 = annotated camera depth at t0). Also the same estimator on the
       REAL keyframes (+0.5 .. +3 s, association by projected annotation) for validation. -> <out>/read.parquet
Usage (pod): python src/vista_leak.py gen /workspace/pack /workspace/out /workspace/Vista logged,collide 0 25"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FR_T = np.arange(25) / 12.0
WP_T = np.array([0.5, 1.0, 1.5, 2.0])
W_IMG, H_IMG, SCALE = 1024, 576, 0.64        # 1600 x 900 -> 1024 x 576 (same aspect, no crop)
VEH = {3, 6, 8}                              # COCO car, bus, truck


def interp(wps, times):
    t = np.r_[0.0, 0.5 * np.arange(1, len(wps) + 1)]
    P = np.vstack([[0.0, 0.0], np.asarray(wps, float)[:, :2]])
    return np.stack([np.interp(times, t, P[:, 0]), np.interp(times, t, P[:, 1])], -1)


def plan_wps(w, name, t2=None):
    if name.startswith("g-"):                                 # A37h rollouts use the h- plans
        name = "h-" + name[2:]
    if name == "h-logged":
        name = "logged"
    if name in ("h-collide", "s-collide"):                    # A37g: logged through frame 2, then collide's a*
        P = np.asarray(w["plans"]["logged"], float)[:, :2].copy()
        t = 0.5 * np.arange(1, 7)
        P[:, 0] += 0.5 * w["a_star"] * np.maximum(t - t2, 0) ** 2
        return P
    if name.startswith("pm"):
        a = float(name[2:])
        t = 0.5 * np.arange(1, 7)
        x = w["v0_past"] * t + 0.5 * a * t * t
        x = np.maximum.accumulate(np.maximum(x, 0)) if a < 0 else x
        return np.stack([x, np.zeros(6)], -1)
    if name.startswith("d"):                                  # A37b dose: logged + constant extra accel A
        P = np.asarray(w["plans"]["logged"], float)[:, :2].copy()
        t = 0.5 * np.arange(1, 7)
        P[:, 0] = np.maximum.accumulate(np.maximum(P[:, 0] + 0.5 * float(name[1:]) * t * t, 0))
        return P
    return np.asarray(w["plans"][name], float)[:, :2]


def next_sweeps(data_root, f0, n=2):
    """the n CAM_FRONT sweeps right after the keyframe image f0 (same log), as (paths, seconds after t0)."""
    log, ts0 = f0.split("__CAM_FRONT__")[0], int(f0.rsplit("__", 1)[1].split(".")[0])
    d = Path(data_root) / "sweeps" / "CAM_FRONT"
    c = sorted((int(q.name.rsplit("__", 1)[1].split(".")[0]), q) for q in d.glob(f"{log}__CAM_FRONT__*.jpg"))
    c = [(t, q) for t, q in c if t > ts0][:n]
    return [str(q) for _, q in c], [(t - ts0) / 1e6 for t, _ in c]


def round_traj(path, T0, Le):
    """Vista trajectory input of a rollout round starting at T0 (its frame 0): positions path(t) (t0 frame) up to Le,
    constant velocity after Le, expressed in the frame of the ego at T0 (heading from the path)."""
    def pos(t):
        t = np.atleast_1d(np.asarray(t, float))
        v = (path(np.array([Le])) - path(np.array([Le - 1 / 12]))) * 12
        return path(np.minimum(t, Le)) + np.maximum(t - Le, 0)[:, None] * v
    o = pos(T0)[0]
    d = pos(T0 + 1 / 12)[0] - o
    psi = float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 1e-3 else 0.0
    q = pos(T0 + WP_T) - o
    c, s_ = np.cos(psi), np.sin(psi)
    x, y = c * q[:, 0] + s_ * q[:, 1], -s_ * q[:, 0] + c * q[:, 1]
    return [v for xx, yy in zip(x, y) for v in (-yy, xx)]


def vista_traj(wps):
    q = interp(wps, WP_T)
    return [v for x, y in q for v in (-y, x)]


def gen(pack, out, vista_dir, plans, seeds, n_steps, wfile=None):
    import torch
    from PIL import Image
    pack, out = Path(pack), Path(out)
    W = json.load(open(pack / "windows.json"))
    if wfile:
        keep = set(open(wfile).read().split())
        W = [w for w in W if w["token"] in keep]
    sys.path.insert(0, vista_dir)
    os.chdir(vista_dir)
    from pytorch_lightning import seed_everything
    from sample import load_img
    from sample_utils import do_sample, init_embedder_options, init_model, init_sampling, set_lowvram_mode
    set_lowvram_mode(False)
    model = init_model({"config": "configs/inference/vista.yaml", "ckpt": "ckpts/vista.safetensors"})
    keys = set(x.input_key for x in model.conditioner.embedders)
    uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]
    sampler = init_sampling(guider="VanillaCFG", steps=n_steps, cfg_scale=float(os.environ.get("VISTA_CFG", 2.5)),
                            num_frames=25)
    n_done, t_start = 0, time.time()
    for w in W:
        img = load_img(str(pack / "imgs" / w["frames"][0]["file"]), H_IMG, W_IMG)
        images = img[None].repeat(25, 1, 1, 1)
        hp, ht = next_sweeps(os.environ.get("NUSC_ROOT", "/workspace/data/nuscenes"), w["frames"][0]["file"])
        t2 = ht[-1] if len(ht) == 2 else None
        for pname in plans:
            if pname[:2] in ("h-", "s-", "g-") and (t2 is None or t2 > 0.25):
                print("SKIP history (no 12 Hz sweeps)", w["token"][:8], pname, flush=True)
                continue
            if pname.startswith("h-") or pname.startswith("g-"):
                im = torch.cat([img[None]] + [load_img(q, H_IMG, W_IMG)[None] for q in hp])
                ims, cidx = torch.cat([im, img[None].repeat(22, 1, 1, 1)]), [0, 1, 2]
            else:
                ims, cidx = images, [0]
            wps = plan_wps(w, pname, t2)
            for sd in seeds:
                d = out / w["token"] / f"{pname}_s{sd}"
                if (d / "f24.jpg").exists():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                seed_everything(sd)
                vd = init_embedder_options(keys)
                vd["cond_frames_without_noise"] = img[None]
                vd["cond_aug"] = 0.0
                vd["cond_frames"] = img[None]
                vd["trajectory"] = torch.tensor(vista_traj(wps), dtype=torch.float32)
                if pname.startswith("g-"):                   # A37h: 4 causal rounds of 0.5 s
                    path = lambda tt, wps=wps: interp(wps, tt)  # noqa: E731
                    kept, trajs = None, []
                    for r in range(4):
                        T0 = 6 * r / 12
                        c3 = im if r == 0 else torch.stack(kept[6 * r: 6 * r + 3]) * 2 - 1
                        vd = init_embedder_options(keys)
                        vd["cond_frames_without_noise"], vd["cond_frames"], vd["cond_aug"] = c3[[0]], c3[[0]], 0.0
                        trajs.append(round_traj(path, T0, T0 + 8 / 12))
                        vd["trajectory"] = torch.tensor(trajs[-1], dtype=torch.float32)
                        seed_everything(100 * sd + r)
                        smp, _, _ = do_sample(torch.cat([c3, c3[[-1]].repeat(22, 1, 1, 1)]), model, sampler, vd,
                                              num_rounds=1, num_frames=25, force_uc_zero_embeddings=uc_keys,
                                              initial_cond_indices=[0, 1, 2])
                        kept = list(smp[:9]) if r == 0 else kept + list(smp[3:9])
                    samples = torch.stack(kept[:25])
                else:
                    samples, _, _ = do_sample(ims, model, sampler, vd, num_rounds=1, num_frames=25,
                                              force_uc_zero_embeddings=uc_keys, initial_cond_indices=cidx)
                arr = (samples.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                for k, a in enumerate(arr):
                    Image.fromarray(a).save(d / f"f{k:02d}.jpg", quality=92)
                try:
                    import imageio
                    imageio.mimsave(d / "video.mp4", list(arr), fps=12, quality=8)
                except Exception as e:                   # media only; frames above are the data
                    print("mp4 failed (non-fatal):", repr(e), flush=True)
                json.dump({"plan": pname, "seed": sd, "traj": vista_traj(wps), "n_steps": n_steps, "cond_idx": cidx,
                           "t2": t2, "hist": hp if pname[:2] in ("h-", "g-") else [],
                           "round_trajs": trajs if pname.startswith("g-") else []},
                          open(d / "meta.json", "w"))
                n_done += 1
                print(f"{n_done} clips, {(time.time() - t_start) / n_done:.1f} s/clip | {w['token'][:8]} {pname} s{sd}",
                      flush=True)


# ------------------------------------------------------------------------------------------------------- readout
def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(box, fr):
    """global annotated box {'t','q','size'(w,l,h)} -> (2D bbox in the 1024 x 576 image, camera depth) or None."""
    w, l, h = box["size"]
    c = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    P = c @ quat_R(box["q"]).T + np.asarray(box["t"])
    P = (P - np.asarray(fr["ego_t"])) @ quat_R(fr["ego_q"])           # global -> ego
    P = (P - np.asarray(fr["cam_t"])) @ quat_R(fr["cam_q"])           # ego -> camera
    if (P[:, 2] <= 0.5).any():
        return None
    uv = P @ np.asarray(fr["K"]).T
    uv = uv[:, :2] / uv[:, 2:3] * SCALE
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    if x1 < 0 or y1 < 0 or x0 > W_IMG or y0 > H_IMG:
        return None
    ctr = (np.asarray(box["t"]) - np.asarray(fr["ego_t"])) @ quat_R(fr["ego_q"])
    depth = float(((ctr - np.asarray(fr["cam_t"])) @ quat_R(fr["cam_q"]))[2])
    return np.array([max(x0, 0), max(y0, 0), min(x1, W_IMG), min(y1, H_IMG)]), depth


def iou(a, B):
    x0, y0 = np.maximum(a[0], B[:, 0]), np.maximum(a[1], B[:, 1])
    x1, y1 = np.minimum(a[2], B[:, 2]), np.minimum(a[3], B[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    return inter / (ar(a) + ar(B) - inter + 1e-6)


def track(dets, box0, thr0=0.3, thr=0.2):
    """IoU tracking from the projected box: per frame (bbox or None); lost after the first miss."""
    out, prev = [], None
    for f, D in enumerate(dets):
        ref, th = (box0, thr0) if prev is None else (prev, thr)
        if len(D) == 0:
            break
        v = iou(ref, D)
        k = int(v.argmax())
        if v[k] < th:
            break
        prev = D[k]
        out.append(prev)
    return out + [None] * (len(dets) - len(out))


def objects(w):
    """lead + static vehicles at t0 with projected boxes (frame 0 camera) and t0 camera depth."""
    fr0 = w["frames"][0]
    lx, ly = w["lead_track_t0frame"][0][:2]
    objs = []
    for v in w["vehicles"]:
        tr = v["track"]
        pr = project(tr[0], fr0)
        if pr is None:
            continue
        p0 = (np.asarray(tr[0]["t"]) - np.asarray(fr0["ego_t"])) @ quat_R(fr0["ego_q"])
        moved = np.linalg.norm(np.asarray(tr[-1]["t"][:2]) - np.asarray(tr[0]["t"][:2])) if len(tr) > 1 else 0.0
        kind = "lead" if np.hypot(p0[0] - lx, p0[1] - ly) < 1.5 else ("static" if moved < 1.0 and len(tr) >= 5 else
                                                                      "other")
        objs.append({"instance": v["instance"], "kind": kind, "box0": pr[0], "d0": pr[1], "track": tr})
    return objs


def read(pack, out):
    import pandas as pd
    import torch
    import torchvision
    from PIL import Image
    pack, out = Path(pack), Path(out)
    W = {w["token"]: w for w in json.load(open(pack / "windows.json"))}
    dev = os.environ.get("VISTA_DEV", "cuda")                       # VISTA_DEV=mps / cpu off the pod
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").to(dev).eval()
    tf = torchvision.transforms.functional.to_tensor

    def detect(paths):
        res = []
        for b in range(0, len(paths), 13):
            ims = [tf(Image.open(p).convert("RGB").resize((W_IMG, H_IMG))).to(dev) for p in paths[b: b + 13]]
            with torch.no_grad():
                for o in net(ims):
                    k = [i for i, (l_, s) in enumerate(zip(o["labels"].tolist(), o["scores"].tolist()))
                         if l_ in VEH and s >= 0.3]
                    res.append(o["boxes"][k].cpu().numpy())
        return res
    rows = []
    for tok in sorted(os.listdir(out)):
        if tok not in W:
            continue
        w = W[tok]
        objs = objects(w)
        # real keyframes (validation of the distance estimator; association by projected annotation)
        reals = detect([pack / "imgs" / fr["file"] for fr in w["frames"]])
        for ob in objs:
            if ob["kind"] == "other":
                continue
            h0 = None
            for k, fr in enumerate(w["frames"]):
                if k >= len(ob["track"]):
                    break
                pr = project(ob["track"][k], fr)
                if pr is None or len(reals[k]) == 0:
                    continue
                v = iou(pr[0], reals[k])
                if v.max() < 0.3:
                    continue
                bh = float(reals[k][v.argmax()][3] - reals[k][v.argmax()][1])
                if k == 0:
                    h0 = bh
                if h0:
                    rows.append({"token": tok, "scene": w["scene"], "src": "real", "plan": "real", "seed": -1,
                                 "kind": ob["kind"], "inst": ob["instance"], "t": 0.5 * k, "h": bh, "h0": h0,
                                 "d0": ob["d0"], "d_true": pr[1]})
        for clip in sorted(os.listdir(out / tok)):
            cdir = out / tok / clip
            if not (cdir / "f24.jpg").exists():
                continue
            plan, seed = clip.rsplit("_s", 1)
            only = os.environ.get("VISTA_PLANS")                        # optional comma list: read only these plans
            if only and plan not in only.split(","):
                continue
            dets = detect([cdir / f"f{k:02d}.jpg" for k in range(25)])
            for ob in objs:
                if ob["kind"] == "other":
                    continue
                tr = track(dets, ob["box0"])
                if tr[0] is None:
                    continue
                h0 = float(tr[0][3] - tr[0][1])
                for k, b in enumerate(tr):
                    rows.append({"token": tok, "scene": w["scene"], "src": "gen", "plan": plan, "seed": int(seed),
                                 "kind": ob["kind"], "inst": ob["instance"], "t": FR_T[k],
                                 "h": float(b[3] - b[1]) if b is not None else np.nan, "h0": h0, "d0": ob["d0"],
                                 "d_true": np.nan})
        print(f"read {tok[:8]} ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df["d_est"] = df.d0 * df.h0 / df.h
    df.to_parquet(out / "read.parquet")
    print(f"-> {out / 'read.parquet'}")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "gen":
        gen(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(","), [int(s) for s in sys.argv[6].split(",")],
            int(sys.argv[7]), sys.argv[8] if len(sys.argv) > 8 else None)
    elif st == "read":
        read(sys.argv[2], sys.argv[3])
