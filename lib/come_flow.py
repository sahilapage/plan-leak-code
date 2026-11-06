"""COME-T: COME with a TRANSPORT pathway -- the model's own forecast is mixed with the current occupancy CARRIED along a
predicted backward flow, so a vehicle it has seen is moved, not regenerated, and cannot fade by construction.
Diagnosis (A14): the models keep a car only if it does not move; per-cell regeneration spreads a moving car thin. Remedy:
  flow head    on COME's per-frame features (after temporal_shared_branch): backward flow b_t (metres, current frame)
               and a gate g_t in [0, 1] for each of the 6 future frames;
  transport    T_{c,t}(z, y, x) = current one-hot vehicle voxels of class c sampled at (x, y) + b_t(x, y) (bilinear);
  output       p_t = g_t T_{c,t} (vehicle classes) + (1 - g_t sum_c T_{c,t}) p_COME,t   (a proper distribution);
  training     COME's own objective (class-weighted NLL over aligned targets, normalised by non-free voxels) on the
               mixture, plus an L1 loss on b_t at real vehicles' future footprints (nuScenes box tracks, src/tracks_prep.py).
Real training data only (no synthetic inserts): passing the insertion test would be an out-of-distribution result.
Starts exactly at COME (zero flow, gate ~ 0.05). fp32.
Usage: python src/come_flow.py train <out_dir> [--iters 6000] [--lr 5e-5] [--lr-head 1e-3] [--w-flow 0.05]
"""
import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import come_ft as cf  # noqa: E402
import dome_infer as di  # noqa: E402

VEH = [3, 4, 5, 9, 10]
RANGE = 40.0
NMAX = 64


class FlowHead(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 3, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, 0.0, -3.0]))           # zero flow, gate ~ 0.05

    def forward(self, feat):
        o = self.net(feat)
        return o[:, :2] * 10.0, torch.sigmoid(o[:, 2:3])                      # flow in metres (scaled), gate


def come_features(net, inputs):
    """COME forward up to the per-frame features; returns (features (B*T, C, H, W), logits (B, T, K, Z, H, W), targets)."""
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
    feat = net.temporal_shared_branch(x)
    logits = net.occ_sem_conv(feat).reshape(B, tT, -1, tZ, H, W)[..., :tH, :tW]
    return feat[..., :tH, :tW], logits, targets


def transport(cur, flow, gate, T):
    """cur: (B, Z, H, W) current labels; flow: (B*T, 2, H, W) metres (dx, dy); gate: (B*T, 1, H, W).
    Returns gated transported vehicle mass per class: (B, T, len(VEH), Z, H, W)."""
    B, Z, H, W = cur.shape
    oh = torch.stack([(cur == c).float() for c in VEH], 1).reshape(B, len(VEH) * Z, H, W)
    oh = oh[:, None].expand(B, T, -1, H, W).reshape(B * T, -1, H, W)
    xs = (torch.arange(W, device=cur.device) + 0.5) / W * 2 * RANGE - RANGE
    ys = (torch.arange(H, device=cur.device) + 0.5) / H * 2 * RANGE - RANGE
    gx = (xs[None, None, :] + flow[:, 0]) / RANGE                                # normalised to [-1, 1] (align_corners=False)
    gy = (ys[None, :, None] + flow[:, 1]) / RANGE
    grid = torch.stack([gx, gy], -1)
    tr = F.grid_sample(oh, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    tr = tr * gate
    return tr.reshape(B, T, len(VEH), Z, H, W)


def mixture_probs(net, head, inputs):
    """p (B, T, K, Z, H, W) of COME-T, the transported part and the predicted flow."""
    feat, logits, targets = come_features(net, inputs)
    B, T = logits.shape[:2]
    flow, gate = head(feat)
    cur = inputs["source_occs"][:, -1].to(logits.device)
    tr = transport(cur, flow, gate, T)                                         # (B, T, 5, Z, H, W)
    p = logits.float().softmax(2)
    A = tr.sum(2, keepdim=True).clamp(max=1.0)
    p = p * (1 - A)
    p[:, :, VEH] = p[:, :, VEH] + tr
    return p, targets, flow.reshape(B, T, 2, *flow.shape[-2:]), gate.reshape(B, T, *gate.shape[-2:])


def flow_targets(tracks, T=6, H=200, W=200):
    """backward-flow targets (B, T, 2, H, W) and mask (B, T, H, W) from box tracks (B, NMAX, 7, 5) in the current frame."""
    B = tracks.shape[0]
    tgt = np.zeros((B, T, 2, H, W), np.float32)
    msk = np.zeros((B, T, H, W), bool)
    xs = (np.arange(W) + 0.5) / W * 2 * RANGE - RANGE
    ys = (np.arange(H) + 0.5) / H * 2 * RANGE - RANGE
    X, Y = np.meshgrid(xs, ys)                                                # (H, W): X varies along W
    for b in range(B):
        for tr in tracks[b]:
            if np.isnan(tr[0, 0]):
                continue
            x0, y0, a0, L, Wd = tr[0]
            for t in range(T):
                if np.isnan(tr[t + 1, 0]):
                    continue
                xt, yt, at = tr[t + 1, 0], tr[t + 1, 1], tr[t + 1, 2]
                cs, sn = math.cos(at), math.sin(at)
                u = (X - xt) * cs + (Y - yt) * sn
                v = -(X - xt) * sn + (Y - yt) * cs
                m = (np.abs(u) <= L / 2) & (np.abs(v) <= Wd / 2)
                if not m.any():
                    continue
                c0, s0 = math.cos(a0), math.sin(a0)
                xsrc = x0 + u[m] * c0 - v[m] * s0                              # same body point at t0
                ysrc = y0 + u[m] * s0 + v[m] * c0
                tgt[b, t, 0][m] = xsrc - X[m]
                tgt[b, t, 1][m] = ysrc - Y[m]
                msk[b, t][m] = True
    return torch.from_numpy(tgt), torch.from_numpy(msk)


class TrackData(cf.Data):
    def __init__(self, infos, wins, tracks, seed):
        super().__init__(infos, wins, None, 0.0, seed)
        self.tracks = tracks
        self.wins = [w for w in wins if (w[0], w[1]) in tracks]

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        s, j = self.wins[rng.integers(0, len(self.wins))]
        fr = self.infos[s][j: j + 10]
        poses = [di.pose_of(f) for f in fr]
        occ = np.stack([di.load_occ(s, f["token"]) for f in fr]).transpose(0, 3, 2, 1).astype(np.int64)
        tr = np.full((NMAX, 7, 5), np.nan, np.float32)
        a = self.tracks[(s, j)][:NMAX]
        tr[: len(a)] = a
        return {"src": occ[:4], "tgt": occ[4:], "poses": np.stack(poses), "tracks": tr}


def loss_fn(net, head, inputs, tracks, w_flow):
    p, targets, flow, gate = mixture_probs(net, head, inputs)
    tgt = targets.reshape(-1)
    pk = p.permute(0, 1, 3, 4, 5, 2).reshape(-1, p.shape[2])
    valid = tgt != -1
    pt = pk[valid].gather(1, tgt[valid][:, None]).squeeze(1).clamp(min=1e-6)
    cw = torch.tensor(net.occ_losses["ce_loss"].class_weight, device=p.device, dtype=torch.float32)
    nll = (-torch.log(pt) * cw[tgt[valid]]).sum()
    avg = ((tgt != -1) & (tgt != net.num_classes - 1)).sum().clamp(min=1)
    ft, fm = flow_targets(tracks.numpy())
    ft, fm = ft.to(p.device), fm.to(p.device)
    lf = ((flow - ft).abs().sum(2) * fm).sum() / fm.sum().clamp(min=1)
    return nll / avg + w_flow * lf, float(nll / avg), float(lf), float(gate.mean())


def build(ckpt=None):
    net, ci = cf.build_come()
    head = FlowHead().cuda()
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(sd["state_dict"], strict=False)
        head.load_state_dict(sd["head"])
    return net, head, ci


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    infos, wins = cf.train_windows()
    tracks = pickle.load(open(a.tracks, "rb"))
    net, head, ci = build()
    net.train()
    cf.freeze_bn(net)
    opt = torch.optim.AdamW([{"params": [q for q in net.parameters() if q.requires_grad], "lr": a.lr},
                             {"params": head.parameters(), "lr": a.lr_head}], weight_decay=0.01)
    ds = TrackData(infos, wins, tracks, a.seed)
    print(f"{len(ds.wins)} training windows with tracks", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    it, t0, run = 0, time.time(), None
    for batch in dl:
        inputs = {"source_occs": batch["src"], "target_occs": batch["tgt"],
                  "source_metas": [{"ego2global": list(q[:4].numpy())} for q in batch["poses"]],
                  "target_metas": [{"ego2global": list(q[4:].numpy())} for q in batch["poses"]],
                  "metas": [{"scene_range": ci.SCENE_RANGE}] * len(batch["src"])}
        if it == 0:
            with torch.no_grad():
                ref = float(sum(net(inputs).values()))
                l0, n0, f0, g0 = loss_fn(net, head, inputs, batch["tracks"], 0.0)
            print(f"start check: COME loss {ref:.4f} | COME-T NLL at init {n0:.4f} (gate {g0:.3f}) | flow L1 {f0:.2f} m", flush=True)
        loss, nll, lf, gm = loss_fn(net, head, inputs, batch["tracks"], a.w_flow)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 5.0)
        opt.step()
        it += 1
        r = np.array([nll, lf, gm])
        run = r if run is None else 0.98 * run + 0.02 * r
        if it % 50 == 0:
            print(f"it {it} nll {run[0]:.4f} flowL1 {run[1]:.3f} m gate {run[2]:.3f} {(time.time() - t0) / it:.2f} s/it", flush=True)
        if it % a.save_every == 0 or it == a.iters:
            torch.save({"state_dict": net.state_dict(), "head": head.state_dict(), "iter": it, "args": vars(a)},
                       out / f"it{it}.pth")
            print("saved", out / f"it{it}.pth", flush=True)
        if it == a.iters:
            break
    print("TRAIN DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("out")
    ap.add_argument("--tracks", default="<data>/data/tracks_train.pkl")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--w-flow", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.mode == "train":
        train(a)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------------- evaluation interface
class ComeTEval:
    """Drop-in replacement for probe_insert.Come: __call__(pasts, poses_all, seed) -> (B, 4, 200, 200) vehicle p at t0
    (the input) and +1 / +2 / +3 s, each warped into that future frame as for COME. Also exposes probs() for the dump."""

    def __init__(self, ckpt):
        import come_infer as ci
        self.net, self.head, _ = build(ckpt)
        self.net.eval()
        self.head.eval()
        self.ci = ci
        self.vh = torch.tensor(VEH, device="cuda")

    @torch.no_grad()
    def probs(self, pasts, poses_all):
        src = [p.transpose(0, 3, 2, 1) for p in pasts]
        inputs = {"source_occs": torch.from_numpy(np.stack(src).astype(np.int64)),
                  "source_metas": [{"ego2global": ps[:4]} for ps in poses_all],
                  "metas": [{"scene_range": self.ci.SCENE_RANGE}] * len(src)}
        p, _, _, _ = mixture_probs(self.net, self.head, inputs)
        return p, src

    @torch.no_grad()
    def __call__(self, pasts, poses_all, seed):
        p, src = self.probs(pasts, poses_all)
        pv = p[:, :, VEH].sum(2)
        B = p.shape[0]
        out = np.zeros((B, 4, 200, 200), np.float32)
        for b in range(B):
            cur = torch.from_numpy(src[b][3]).cuda()
            out[b, 0] = torch.isin(cur, self.vh)[di.ZLO:di.ZHI + 1].any(0).float().T.cpu().numpy()
            for k, f in enumerate((1, 3, 5)):
                w_ = self.ci.warp(pv[b, f], poses_all[b][3], poses_all[b][4 + f])
                out[b, k + 1] = w_[di.ZLO:di.ZHI + 1].amax(0).T.cpu().numpy()
        return out
