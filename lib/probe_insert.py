"""Intervention test of object permanence ("do-insert"). For held-out windows, a car is inserted into the ground-truth
occupancy a world model receives -- the 4 past keyframes (1.5 s history, 2 Hz) -- moving with a known motion along the
ego's heading at t0, in an adjacent lane that is drivable and free of obstacles in every frame of the window (past and
future, so the continued path is plausible). The model never sees future frames.
Shapes: `box` (solid 4.5 x 1.9 x 1.6 m) and `real` (the voxels of a real, well-observed Occ3D car transplanted from a bank
built from other scenes), to separate "the model ignores motion" from "the insert looks unlike a car".
Conditions: no insertion (baseline), speed 0 / 4 / 8 / 12 m/s for both shapes, and a turning box (8 m/s, 0.15 rad/s,
turning away from the ego's lane). Every condition runs on the same windows with the same random seed (paired).
Measure (the paper's metric): top-down vehicle probability (softmax summed over Occ3D vehicle classes, max over the
-0.5 to 2.0 m band) at t0 / +1 / +2 / +3 s; KEPT at h if >= 50 % of the footprint at the car's continued position has
p >= 0.2. Where did it go: d = max(p_insert - p_baseline, 0) in a corridor around the segment from the car's position
"if it had stopped at t0" (S) to its continued position (C); mass near S and near C (3 m), the corridor mass, and the
progress of the d-weighted centroid along S -> C (0 = left behind, 1 = moved correctly).
Usage: python src/probe_insert.py <dome|come> <fold> <n_windows> <out.parquet> [--seed S] [--bank-fold F]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402

CTR = -40 + 0.4 * (np.arange(200) + 0.5)
GX, GY = np.meshgrid(CTR, CTR, indexing="ij")
BOX = (4.5, 1.9)
OBST = list(range(0, 11))                      # Occ3D classes 0..10 are objects / obstacles
DRIVABLE = 11
CONDS = [("base", None, 0.0, None),
         ("v0", 0.0, 0.0, "box"), ("v4", 4.0, 0.0, "box"), ("v8", 8.0, 0.0, "box"), ("v12", 12.0, 0.0, "box"),
         ("turn8", 8.0, 0.15, "box"),
         ("v0r", 0.0, 0.0, "real"), ("v4r", 4.0, 0.0, "real"), ("v8r", 8.0, 0.0, "real"), ("v12r", 12.0, 0.0, "real")]
X_AHEAD, LANE = 10.0, 3.6                      # start 10 m ahead of the ego at t0, one lane to the side
FT = [-1.5, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0]    # times of the frames used: 4 past (input) + 3 future (evaluation)


def yaw_of(P):
    return math.atan2(P[1, 0], P[0, 0])


def car_state(P0, side, v, w, t, xa=None):
    """global (xy, yaw) of the inserted car at time t (s) relative to t0; it starts xa m ahead of the ego at t0."""
    xa = X_AHEAD if xa is None else xa
    yaw0 = yaw_of(P0)
    x0 = P0[:2, 3] + xa * np.array([math.cos(yaw0), math.sin(yaw0)]) + side * LANE * np.array([-math.sin(yaw0), math.cos(yaw0)])
    if abs(w) < 1e-9:
        return x0 + v * t * np.array([math.cos(yaw0), math.sin(yaw0)]), yaw0
    w = w * side                               # turn away from the ego's lane (left lane turns left, right lane right)
    yaw = yaw0 + w * t
    return x0 + np.array([v / w * (math.sin(yaw) - math.sin(yaw0)), -v / w * (math.cos(yaw) - math.cos(yaw0))]), yaw


def to_ego(P, xy):
    c = np.linalg.inv(P) @ np.array([xy[0], xy[1], 0.0, 1.0])
    return c[:2]


def footprint(P, xy, yaw, dims=BOX, grow=0.0):
    """(200, 200) bool footprint of a car at global (xy, yaw) in the ego frame of pose P, and its centre in that frame."""
    c = to_ego(P, xy)
    yl = yaw - yaw_of(P)
    cs, sn = math.cos(yl), math.sin(yl)
    u = (GX - c[0]) * cs + (GY - c[1]) * sn
    w = -(GX - c[0]) * sn + (GY - c[1]) * cs
    return (np.abs(u) <= dims[0] / 2 + grow) & (np.abs(w) <= dims[1] / 2 + grow), c


def ground_top(occ, m):
    """median top index of drivable voxels in the footprint columns (fallback: 2)."""
    idx = [np.where(c == DRIVABLE)[0].max() for c in occ[m] if (c == DRIVABLE).any()]
    return int(np.median(idx)) if idx else 2


def valid_insert(occs, poses, side, xa, conds):
    """in range, on drivable ground, free of GT obstacles (with a margin) in all 7 frames, for every given condition."""
    for _, v, w, shape in conds:
        if v is None:
            continue
        for k, t in enumerate(FT):
            xy, yaw = car_state(poses[3], side, v, w, t, xa)
            m, c = footprint(poses[k], xy, yaw, dims=(5.2, 2.1))
            if not m.any() or abs(c[0]) > 36 or abs(c[1]) > 36:
                return False
            col = occs[k][m]
            if np.isin(col, OBST).any() or (col == DRIVABLE).any(1).mean() < 0.8:
                return False
            mg, _ = footprint(poses[k], xy, yaw, dims=(5.2, 2.1), grow=0.6)
            if np.isin(occs[k][mg], OBST).any():
                return False
    return True


def insert(occ, P, xy, yaw, patch=None):
    """paste the car into one frame: a solid box, or a real car's voxels (u along its heading, w left, dz above road)."""
    occ = occ.copy()
    dims = BOX if patch is None else patch["dims"]
    m, c = footprint(P, xy, yaw, dims=dims)
    g = ground_top(occ, m)
    if patch is None:
        occ[m, g + 1: g + 5] = 4
        return occ
    yl = yaw - yaw_of(P)
    cs, sn = math.cos(yl), math.sin(yl)
    x = c[0] + patch["u"] * cs - patch["w"] * sn
    y = c[1] + patch["u"] * sn + patch["w"] * cs
    i, j = np.floor((x + 40) / 0.4).astype(int), np.floor((y + 40) / 0.4).astype(int)
    z = np.clip(g + patch["dz"], 0, 15)
    ok = (i >= 0) & (i < 200) & (j >= 0) & (j < 200)
    occ[i[ok], j[ok], z[ok]] = patch["cls"][ok]
    return occ


def build_bank(fold, n, exclude):
    """real cars from Occ3D: well-observed (visibility 80-100 %, >= 50 lidar points), 3.8-5.2 m long, within 25 m, at least
    70 % of footprint columns holding vehicle voxels; voxels stored in the car's own frame."""
    from common import DATA
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    nu = NuScenes("v1.0-trainval", str(DATA / "nuscenes"), verbose=False)
    infos, wins = di.windows({fold})
    rng = np.random.default_rng(123)
    bank = []
    for j in rng.permutation(len(wins)):
        scene, idx = wins[j]
        if scene in exclude:
            continue
        f = infos[scene][idx + 3]
        P = di.pose_of(f)
        sm = nu.get("sample", f["token"])
        occ = None
        for at in sm["anns"]:
            a = nu.get("sample_annotation", at)
            L, W = a["size"][1], a["size"][0]
            if a["category_name"] != "vehicle.car" or not (3.8 <= L <= 5.2) or a["visibility_token"] != "4" or a["num_lidar_pts"] < 50:
                continue
            xy, yaw = np.array(a["translation"][:2]), Quaternion(a["rotation"]).yaw_pitch_roll[0]
            m, c = footprint(P, xy, yaw, dims=(L, W))
            if not m.any() or np.hypot(*c) > 25:
                continue
            occ = di.load_occ(scene, f["token"]) if occ is None else occ
            veh = np.isin(occ[m], di.VEHICLE)
            if veh.any(1).mean() < 0.7:
                continue
            g = ground_top(occ, m)
            mi = np.argwhere(m)
            u_, w_, dz_, cl_ = [], [], [], []
            yl = yaw - yaw_of(P)
            for (ii, jj) in mi:
                for zz in np.where(np.isin(occ[ii, jj], di.VEHICLE))[0]:
                    dx, dy = CTR[ii] - c[0], CTR[jj] - c[1]
                    u_.append(dx * math.cos(yl) + dy * math.sin(yl))
                    w_.append(-dx * math.sin(yl) + dy * math.cos(yl))
                    dz_.append(zz - g)
                    cl_.append(occ[ii, jj, zz])
            bank.append({"u": np.array(u_), "w": np.array(w_), "dz": np.array(dz_), "cls": np.array(cl_, np.uint8),
                         "dims": (L, W), "src": f["token"]})
            break
        if len(bank) == n:
            break
    return bank


def select(fold, n, seed):
    """windows with a near-straight ego path (< 8 deg over 4.5 s) and a valid insertion: start 10 / 0 / -10 m ahead of the
    ego, left lane then right lane, first that is valid for every straight condition; turn_ok = the turning path is valid
    too (turning rows are analysed on that subset only)."""
    infos, wins = di.windows({fold})
    rng = np.random.default_rng(seed)
    out = []
    for j in rng.permutation(len(wins)):
        scene, idx = wins[j]
        fr = infos[scene][idx: idx + 10]
        poses = [di.pose_of(f) for f in fr]
        if abs((yaw_of(poses[9]) - yaw_of(poses[0]) + math.pi) % (2 * math.pi) - math.pi) > math.radians(8):
            continue
        keep = [0, 1, 2, 3, 5, 7, 9]
        occs = [di.load_occ(scene, fr[k]["token"]) for k in keep]
        ps = [poses[k] for k in keep]
        main = [c for c in CONDS if c[0] != "turn8"]
        turn = [c for c in CONDS if c[0] == "turn8"]
        found = None
        for xa in (10.0, 0.0, -10.0):
            for side in (1, -1):
                if valid_insert(occs, ps, side, xa, main):
                    found = {"scene": scene, "idx": idx, "token": fr[3]["token"], "side": side, "xa": xa,
                             "turn_ok": valid_insert(occs, ps, side, xa, turn)}
                    break
            if found:
                break
        if found:
            out.append(found)
        if len(out) == n:
            break
    return infos, out


# ----------------------------------------------------------------------------------------------------------- models
class Dome:
    def __init__(self):
        import os
        from mmengine import Config
        from mmengine.registry import MODELS
        import model  # noqa: F401
        from diffusion import create_diffusion
        os.environ["eval"] = "true"
        cfg = Config.fromfile(str(di.DOME / "config" / "train_dome.py"))
        cfg.model.world_model.attention_mode = "math"
        self.net = MODELS.build(cfg.model.world_model).cuda().eval()
        ck = torch.load(di.DOME / "ckpts" / "dome_latest.pth", map_location="cpu")
        self.net.load_state_dict(ck["ema"] if "ema" in ck else ck["state_dict"], strict=False)
        self.vae = MODELS.build(cfg.model.vae).cuda().eval()
        self.vae.load_state_dict(torch.load(di.DOME / "ckpts" / "occvae_latest.pth", map_location="cpu")["state_dict"])
        self.diff = create_diffusion(timestep_respacing=str(cfg.sample.num_sampling_steps), beta_start=cfg.schedule.beta_start,
                                     beta_end=cfg.schedule.beta_end, replace_cond_frames=cfg.replace_cond_frames,
                                     cond_frames_choices=cfg.cond_frames_choices,
                                     predict_xstart=cfg.schedule.get("predict_xstart", False))
        self.cfg = cfg
        self.vh = torch.tensor(di.VEHICLE, device="cuda")

    @torch.no_grad()
    def __call__(self, pasts, poses_all, seed):
        """pasts: list of (4, 200, 200, 16) past occupancies; poses_all: lists of 10 poses -> (B, 4, 200, 200) vehicle p."""
        from einops import rearrange
        cfg, end, mid = self.cfg, self.cfg.end_frame, self.cfg.mid_frame
        occs = [np.concatenate([p, np.repeat(p[-1:], end - mid, 0)]) for p in pasts]
        metas = [di.get_meta_data(np.stack(ps)) for ps in poses_all]
        occ = torch.from_numpy(np.stack(occs)).long().cuda()
        bs = occ.shape[0]
        torch.manual_seed(seed)
        lat, _ = self.vae.forward_encoder(occ)
        lat, _, _ = self.vae.sample_z(lat)
        lat = lat * cfg.model.vae.scaling_factor
        lat = rearrange(lat, "(b f) c h w -> b f c h w", b=bs) if lat.dim() == 4 else rearrange(lat, "b c f h w -> b f c h w")
        w = cfg.model.vae.encoder_cfg.resolution // 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
        z = self.diff.p_sample_loop(self.net, (bs, end, cfg.base_channel, w, w), None, clip_denoised=False,
                                    model_kwargs={"metas": metas}, progress=False, device="cuda",
                                    initial_cond_indices=list(range(mid)), initial_cond_frames=lat)
        z = rearrange(z / cfg.model.vae.scaling_factor, "b f c h w -> b c f h w")
        logits = self.vae.forward_decoder(z, shapes=[[200, 200], [100, 100], [50, 50]], input_shape=[bs, end, 200, 200, cfg._dim_])
        keep = [mid - 1, mid + 1, mid + 3, mid + 5]
        p = torch.softmax(logits[:, keep].float(), dim=-1)
        return p[..., self.vh].sum(-1)[..., di.ZLO:di.ZHI + 1].amax(-1).cpu().numpy()


class Come:
    def __init__(self):
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
        self.net = MODELS.build(cfg.model).cuda().eval()
        ck = torch.load(ci.COME / "work_dir" / "unet" / "unet_past2s_future3s.pth", map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck.get("state_dict", ck.get("model", ck)), strict=False)
        self.ci = ci
        self.vh = torch.tensor(di.VEHICLE, device="cuda")

    @torch.no_grad()
    def __call__(self, pasts, poses_all, seed):
        ci, net = self.ci, self.net
        src = [p.transpose(0, 3, 2, 1) for p in pasts]                                         # (T, Z, Y, X)
        inputs = {"source_occs": torch.from_numpy(np.stack(src).astype(np.int64)),
                  "source_metas": [{"ego2global": ps[:4]} for ps in poses_all],
                  "metas": [{"scene_range": ci.SCENE_RANGE}] * len(src)}
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
        pv = logits.float().softmax(dim=2)[:, :, self.vh].sum(2)                                # (B, 6, Z, Y, X)
        out = np.zeros((B, 4, 200, 200), np.float32)
        for b in range(B):
            cur = torch.from_numpy(src[b][3]).cuda()
            out[b, 0] = torch.isin(cur, self.vh)[di.ZLO:di.ZHI + 1].any(0).float().T.cpu().numpy()
            for k, f in enumerate((1, 3, 5)):
                w_ = ci.warp(pv[b, f], poses_all[b][3], poses_all[b][4 + f])
                out[b, k + 1] = w_[di.ZLO:di.ZHI + 1].amax(0).T.cpu().numpy()
        return out


# ----------------------------------------------------------------------------------------------------------- measures
def q50(p, m):
    v = np.sort(p[m])[::-1]
    return float(v[max(1, math.ceil(0.5 * len(v))) - 1])


def where(d, S, C):
    """d: (200, 200) extra vehicle probability; S, C: ego-frame centres if stopped / if continued."""
    seg = C - S
    L2 = float(seg @ seg)
    rx, ry = GX - S[0], GY - S[1]
    t = np.clip((rx * seg[0] + ry * seg[1]) / L2, -0.3, 1.3) if L2 > 1e-6 else np.zeros_like(GX)
    dist = np.hypot(rx - t * seg[0], ry - t * seg[1])
    cor = dist <= 4.0
    mass = float(d[cor].sum())
    prog = float((d[cor] * t[cor]).sum() / mass) if (L2 > 1e-6 and mass > 1e-6) else np.nan
    return {"mS": float(d[np.hypot(rx, ry) <= 3.0].sum()), "mC": float(d[np.hypot(GX - C[0], GY - C[1]) <= 3.0].sum()),
            "mcor": mass, "prog": prog}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("fold", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--bank-fold", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    infos, sel = select(a.fold, a.n, a.seed)
    bank = build_bank(a.bank_fold, 60, exclude={s["scene"] for s in sel})
    print(f"{len(sel)} valid windows, {len(bank)} real cars in the bank", flush=True)
    net = {"dome": Dome, "come": Come}[a.model]()
    rows = []
    for b0 in range(0, len(sel), a.bs):
        batch = sel[b0: b0 + a.bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s in batch]
        poses_all = [[di.pose_of(f) for f in fr] for fr in frs]
        base_past = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for s, fr in zip(batch, frs)]
        patches = [bank[int(s["token"][:8], 16) % len(bank)] for s in batch]
        maps = {}
        for cname, v, w, shape in CONDS:
            pasts = []
            for s, bp, ps, pa in zip(batch, base_past, poses_all, patches):
                if v is None:
                    pasts.append(bp)
                    continue
                pasts.append(np.stack([insert(bp[k], ps[k], *car_state(ps[3], s["side"], v, w, FT[k], s["xa"]),
                                              patch=pa if shape == "real" else None) for k in range(4)]))
            maps[cname] = net(pasts, poses_all, seed=1000 * a.seed + b0)
        for bi, (s, ps, pa) in enumerate(zip(batch, poses_all, patches)):
            fpose = [ps[3], ps[5], ps[7], ps[9]]
            for cname, v, w, shape in CONDS[1:]:
                dims = BOX if shape == "box" else pa["dims"]
                for h in range(4):
                    xy, yaw = car_state(ps[3], s["side"], v, w, float(h), s["xa"])
                    m, C = footprint(fpose[h], xy, yaw, dims=dims)
                    S = to_ego(fpose[h], car_state(ps[3], s["side"], 0.0, 0.0, 0.0, s["xa"])[0])
                    p, pb = maps[cname][bi][h], maps["base"][bi][h]
                    row = {"token": s["token"], "scene": s["scene"], "side": s["side"], "cond": cname, "shape": shape,
                           "speed": v, "turn": w, "xa": s["xa"], "turn_ok": s["turn_ok"], "h": h, "q50": q50(p, m), "cov": float((p[m] >= 0.2).mean()),
                           "base_q50": q50(pb, m), "src": pa["src"] if shape == "real" else ""}
                    row.update(where(np.clip(p - pb, 0, None), S, C))
                    rows.append(row)
        print(f"{b0 + len(batch)}/{len(sel)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    t = df[(df.cond != "turn8") | df.turn_ok].assign(kept=lambda d: d.q50 >= 0.2, base_kept=lambda d: d.base_q50 >= 0.2)
    print(t.groupby(["cond", "h"]).agg(n=("kept", "size"), kept=("kept", "mean"), base_kept=("base_kept", "mean"),
                                       mcor=("mcor", "mean"), mS=("mS", "mean"), mC=("mC", "mean"),
                                       prog=("prog", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
