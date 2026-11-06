"""Object-level driving world models that differ ONLY in how they condition on the ego plan (A31).
A logged 3 s ego trajectory is not a decision taken at t0: the driver kept reacting to the world during those 3 s, so
the logged future plan is partly a CONSEQUENCE of the other agents' futures (time-varying confounding). A model that
conditions on the whole future plan (chunked, as SWTC / DOME / most video WMs) learns that reverse coupling (the plan
leak). The g-formula alternative predicts one 0.5 s step at a time from the state so far, seeing the ego plan only up
to `lookahead` steps beyond the current step, and is rolled out on its own predictions; with lookahead within the
driver's reaction delay, the logged action carries no information about the next world state that the history lacks.
Models (same transformer, tokens = ego + up to NA agents, inputs = annotated tracks in the t0 ego frame, 2 Hz):
  step<L>  next-step predictor, ego plan visible up to L steps ahead of the current step, rolled out 6 steps
  chunk    one shot from t0, sees the whole 6-step ego future (the plan-conditioned WM)
  marg     one shot from t0, sees no ego future (plan-free)
Stages:
  build <split> <out.npz>          nuScenes train / val windows (3 past + t0 + 6 future keyframes)
  train <mode> <train.npz> <ckpt>  mode in step0 step1 step2 step3 step6 chunk marg
  leak  <val.npz> <out.parquet> <ckpt> ...   plan-leak beta per relation (counterfactual plans = logged + 0.5 a t^2)
  judge <val.npz> <out.parquet> <het_dir> ... --ckpt ...   A29 decision benchmark judges (src/sw_hetero.py tables)
Usage (server, vla env): python src/obj_wm.py build train $H/objwm/train.npz | train step0 ... | leak ... | judge ..."""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))

NA, T, P = 24, 10, 3             # agents per window, keyframes per window (idx 3 = t0), past steps before t0
SC = 10.0                        # coordinate scale


# ------------------------------------------------------------------------------------------------------------- data
def build(split, out):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from pyquaternion import Quaternion
    import realhazard as rh
    nusc = NuScenes("v1.0-trainval", dataroot=str(rh.DATA / "nuscenes"), verbose=False)
    names = set(splits.train if split == "train" else splits.val)
    E, A, M, S, TOK, SCN = [], [], [], [], [], []
    for sc in nusc.scene:
        if sc["name"] not in names:
            continue
        seq = [nusc.get("sample", sc["first_sample_token"])]
        while seq[-1]["next"]:
            seq.append(nusc.get("sample", seq[-1]["next"]))
        poses, inst = [], []
        for s in seq:
            sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            p = np.eye(4)
            p[:3, 3] = ep["translation"]
            p[:3, :3] = Quaternion(ep["rotation"]).rotation_matrix
            poses.append(p)
            d = {}
            for at in s["anns"]:
                a = nusc.get("sample_annotation", at)
                if a["category_name"].startswith(rh.VEH_CATS):
                    d[a["instance_token"]] = (a["translation"][0], a["translation"][1],
                                              Quaternion(a["rotation"]).yaw_pitch_roll[0], a["size"][1], a["size"][0])
            inst.append(d)
        for k0 in range(P, len(seq) - 6):
            Ti = np.linalg.inv(poses[k0])
            yaw0 = math.atan2(poses[k0][1, 0], poses[k0][0, 0])

            def tf(x, y, yaw):
                g = Ti @ np.array([x, y, 0.0, 1.0])
                return g[0], g[1], (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
            e = np.array([tf(poses[k][0, 3], poses[k][1, 3], math.atan2(poses[k][1, 0], poses[k][0, 0]))
                          for k in range(k0 - P, k0 + 7)], np.float32)
            cand = []
            for it, v in inst[k0].items():
                x, y, _ = tf(*v[:3])
                if abs(x) <= 60 and abs(y) <= 40:
                    cand.append((math.hypot(x, y), it))
            cand.sort()
            a = np.zeros((NA, T, 3), np.float32)
            m = np.zeros((NA, T), bool)
            sz = np.zeros((NA, 2), np.float32)
            for j, (_, it) in enumerate(cand[:NA]):
                sz[j] = inst[k0][it][3:5]
                for t, k in enumerate(range(k0 - P, k0 + 7)):
                    if it in inst[k]:
                        a[j, t] = tf(*inst[k][it][:3])
                        m[j, t] = True
            E.append(e)
            A.append(a)
            M.append(m)
            S.append(sz)
            TOK.append(seq[k0]["token"])
            SCN.append(sc["name"])
    np.savez_compressed(out, E=np.stack(E), A=np.stack(A), M=np.stack(M), S=np.stack(S), tok=np.array(TOK),
                        scene=np.array(SCN))
    print(f"{split}: {len(E)} windows from {len(set(SCN))} scenes -> {out}")


# ------------------------------------------------------------------------------------------------------------ model
def feats(pos, msk, size, s, is_ego):
    """pos (B, N, T, 3), msk (B, N, T): token features at current step s (history s-3..s)."""
    import torch
    cur = pos[:, :, s]
    hist = pos[:, :, s - P: s] - cur[:, :, None]
    hm = msk[:, :, s - P: s].float()
    f = [cur[..., :2] / SC, torch.cos(cur[..., 2:3]), torch.sin(cur[..., 2:3]),
         (hist[..., :2] * hm[..., None]).flatten(2) / 5.0, hm, size / 5.0,
         torch.full_like(cur[..., :1], float(is_ego))]
    return torch.cat(f, -1)


def make_net(mode):
    import torch
    import torch.nn as nn
    lk = 6 if mode == "chunk" else (0 if mode == "marg" else int(mode[4:]))
    out = 2 if mode.startswith("step") else 12

    class WM(nn.Module):
        def __init__(self):
            super().__init__()
            fin = 2 + 2 + 2 * P + P + 2 + 1
            self.inp = nn.Linear(fin + 2 * 6 + 6, 128)
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True), 3)
            self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, out))
            self.lk, self.mode = lk, mode

        def forward(self, E, A, M, S, s):
            """E (B, T, 3) ego (future part = plan), A (B, NA, T, 3), M (B, NA, T), S (B, NA, 2); predict from step s.
            Ego plan steps s+1 .. s+lk (truncated at T-1) are added to the ego token, relative to the ego at s."""
            B = E.shape[0]
            em = torch.ones(B, 1, T, dtype=torch.bool, device=E.device)
            fe = feats(E[:, None], em, torch.tensor([[4.084, 1.73]], device=E.device).expand(B, 1, 2), s, 1)
            fa = feats(A, M, S, s, 0)
            plan = torch.zeros(B, 1, 18, device=E.device)
            for i in range(1, self.lk + 1):
                if s + i < T:
                    plan[:, 0, 2 * (i - 1): 2 * i] = (E[:, s + i, :2] - E[:, s, :2]) / 5.0
                    plan[:, 0, 12 + i - 1] = 1.0
            x = torch.cat([torch.cat([fe, plan], -1), torch.cat([fa, torch.zeros(B, NA, 18, device=E.device)], -1)], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=E.device), ~M[:, :, s]], 1)
            h = self.enc(self.inp(x), src_key_padding_mask=pad)
            return self.head(h[:, 1:])
    return WM()


def cv_base(A, M, s, n):
    """constant-velocity positions of every agent for steps s+1 .. s+n from its last two states (B, NA, n, 2);
    zero velocity when the previous state is missing."""
    import torch
    v = (A[:, :, s, :2] - A[:, :, s - 1, :2]) * M[:, :, s - 1, None]
    return A[:, :, s, None, :2] + v[:, :, None] * torch.arange(1, n + 1, device=A.device)[None, None, :, None]


def rollout(net, E, A, M, S):
    """predicted agent positions for steps 4 .. 9 (B, NA, 6, 2) under the ego trajectory E (its future = the plan)."""
    import torch
    A = A.clone()
    M = M.clone()
    if net.mode.startswith("step"):
        out = []
        for s in range(P, T - 1):
            p = cv_base(A, M, s, 1)[:, :, 0] + net(E, A, M, S, s) / 2.0
            d = p - A[:, :, s, :2]
            yaw = torch.where(d.norm(dim=-1) > 0.2, torch.atan2(d[..., 1], d[..., 0]), A[:, :, s, 2])
            A[:, :, s + 1] = torch.cat([p, yaw[..., None]], -1)
            M[:, :, s + 1] = M[:, :, s]
            out.append(p)
        return torch.stack(out, 2)
    return cv_base(A, M, P, 6) + net(E, A, M, S, P).view(-1, NA, 6, 2) / 2.0 * torch.arange(1, 7, device=A.device)[
        None, None, :, None] ** 0.5


def load(npz, dev):
    import torch
    z = np.load(npz)
    return {k: torch.tensor(z[k], device=dev) for k in ("E", "A", "M", "S")}, z["tok"], z["scene"]


def train(mode, npz, ckpt, steps=20000, seed=0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = __import__("os").environ.get("OBJ_DEV", "cuda")          # OBJ_DEV=cpu for CPU-only hosts
    D, _, scn = load(npz, dev)
    us = np.unique(scn)
    va_sc = set(rng.choice(us, len(us) // 10, replace=False))
    va = np.array([s in va_sc for s in scn])
    tr_i, va_i = np.where(~va)[0], np.where(va)[0]
    net = make_net(mode).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)

    def loss_on(ix, train_mode):
        E, A, M, S = (D[k][ix] for k in ("E", "A", "M", "S"))
        if train_mode and rng.random() < 0.5:                         # mirror augmentation
            fl = torch.tensor([1.0, -1.0, -1.0], device=dev)
            E, A = E * fl, A * fl
        if mode.startswith("step"):
            s = int(rng.integers(P, T - 1))
            tgt = A[:, :, s + 1, :2]
            ok = M[:, :, s] & M[:, :, s + 1] & M[:, :, s - 1]
            pred = cv_base(A, M, s, 1)[:, :, 0] + net(E, A, M, S, s) / 2.0
            err = (pred - tgt).norm(dim=-1)
        else:
            tgt = A[:, :, P + 1:, :2]
            ok = (M[:, :, P] & M[:, :, P - 1])[:, :, None] & M[:, :, P + 1:]
            pred = rollout(net, E, A, M, S)
            err = (pred - tgt).norm(dim=-1)
        return torch.nn.functional.huber_loss(err[ok], torch.zeros_like(err[ok]), delta=1.0)

    best, bad = 1e9, 0
    for it in range(steps):
        net.train()
        ix = torch.tensor(rng.choice(tr_i, 256), device=dev)
        loss = loss_on(ix, True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if it % 1000 == 999:
            net.eval()
            with torch.no_grad():
                ade = val_ade(net, D, va_i)
            print(f"{mode} it {it + 1} loss {loss.item():.4f} val rollout ADE {ade:.3f} m", flush=True)
            if ade < best:
                best = ade
                torch.save({"mode": mode, "sd": net.state_dict(), "val_ade": ade, "it": it + 1}, ckpt)
    print(f"{mode} best val rollout ADE {best:.3f} -> {ckpt}")


def val_ade(net, D, ix):
    import torch
    errs = []
    for b in range(0, len(ix), 512):
        j = torch.tensor(ix[b: b + 512], device=D["E"].device)
        E, A, M, S = (D[k][j] for k in ("E", "A", "M", "S"))
        pred = rollout(net, E, A, M, S)
        ok = (M[:, :, P] & M[:, :, P - 1])[:, :, None] & M[:, :, P + 1:]
        errs.append((pred - A[:, :, P + 1:, :2]).norm(dim=-1)[ok])
    return float(torch.cat(errs).mean())


def load_net(ckpt, dev="cuda"):
    import torch
    c = torch.load(ckpt, map_location=dev)
    net = make_net(c["mode"]).to(dev)
    net.load_state_dict(c["sd"])
    net.eval()
    return net


def plan_E(E, wp):
    """ego trajectory with its future (idx 4..9) replaced by plan waypoints wp (B, 6, 2); heading from the steps."""
    import torch
    E = E.clone()
    prev = torch.cat([E[:, P:P + 1, :2], wp[:, :-1]], 1)
    d = wp - prev
    yaw = torch.atan2(d[..., 1], d[..., 0])
    yaw = torch.where(d.norm(dim=-1) > 0.2, yaw, E[:, P:P + 1, 2].expand_as(yaw))
    E[:, P + 1:] = torch.cat([wp, yaw[..., None]], -1)
    return E


def acc_wp(wp, a):
    """logged waypoints + 0.5 a t^2 along x, never reversing (B, 6, 2)."""
    import torch
    t = 0.5 * torch.arange(1, 7, device=wp.device, dtype=wp.dtype)
    x = wp[..., 0] + 0.5 * a * t * t
    x = torch.cummax(torch.maximum(x, torch.zeros_like(x)), 1).values if a < 0 else x
    return torch.stack([x, wp[..., 1]], -1)


# ------------------------------------------------------------------------------------------------------------- leak
def leak(npz, ckpts, out):
    import pandas as pd
    import torch
    from relation_leak import classify
    D, tok, scn = load(npz, "cuda")
    A_np = D["A"].cpu().numpy()
    M_np = D["M"].cpu().numpy()
    rel = {}
    for i in range(len(tok)):
        for j in range(NA):
            if M_np[i, j, P:].all() and M_np[i, j, P - 1]:
                r = classify(A_np[i, j, P:])
                if r is not None:
                    rel[(i, j)] = r
    rows = []
    for ck in ckpts:
        net = load_net(ck)
        for b in range(0, len(tok), 512):
            ix = torch.arange(b, min(b + 512, len(tok)), device="cuda")
            E, A, M, S = (D[k][ix] for k in ("E", "A", "M", "S"))
            wp = E[:, P + 1:, :2]
            with torch.no_grad():
                base = rollout(net, E, A, M, S)
                for a in (2.0, -1.5):
                    wpc = acc_wp(wp, a)
                    cf = rollout(net, plan_E(E, wpc), A, M, S)
                    dx_e = (wpc - wp)[..., 0]
                    dx_a = (cf - base)[..., 0]
                    for ii in range(len(ix)):
                        gi = b + ii
                        for j in range(NA):
                            if (gi, j) not in rel:
                                continue
                            for h in (1, 2, 3):
                                k = 2 * h - 1
                                if abs(float(dx_e[ii, k])) < 0.5:
                                    continue
                                rows.append({"model": Path(ck).stem, "scene": scn[gi], "rel": rel[(gi, j)], "a": a, "h": h,
                                             "beta": float(dx_a[ii, j, k] / dx_e[ii, k])})
        print(f"{ck}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    summarize_leak(df)


def summarize_leak(df, B=1000):
    rng = np.random.default_rng(0)
    for (rel, h), g in df.groupby(["rel", "h"]):
        line = []
        sc = g.scene.unique()
        for m, gm in g.groupby("model"):
            byS = {s: gm[gm.scene == s].beta.to_numpy() for s in sc}
            bs = [np.median(np.concatenate([byS[s] for s in rng.choice(sc, len(sc)) if s in byS] or [[np.nan]]))
                  for _ in range(B)]
            line.append(f"{m} {np.median(gm.beta):+.2f} [{np.nanquantile(bs, .025):+.2f}, {np.nanquantile(bs, .975):+.2f}]")
        print(f"  {rel:9s} +{h} s (n {len(g) // g.model.nunique()}, {len(sc)} scenes): " + " | ".join(line))


# ------------------------------------------------------------------------------------------------------------ judge
def judge(npz, out, hets, ckpts):
    import pandas as pd
    import torch
    from fixed_acc import ACCS, FRONT
    from sw_hetero import REAR
    from sw_realhazard import EGO_W, plan_pose
    D, tok, scn = load(npz, "cuda")
    ix = {t: i for i, t in enumerate(tok)}
    nets = [load_net(c) for c in ckpts]
    rows, nmiss, devs = [], 0, []
    for het in hets:
        T_ = pickle.load(open(Path(het) / "acc" / "truth.pkl", "rb"))
        df = pd.read_parquet(Path(het) / "table.parquet")
        df = df.rename(columns={"self_rptA": "self_rpt", "cvref_rptA": "cvref_rpt"})
        for tk, g in df.groupby("token"):
            if tk not in ix:
                nmiss += 1
                continue
            i = ix[tk]
            t = T_[tk]
            E, A, M, S = (D[k][i: i + 1] for k in ("E", "A", "M", "S"))
            lg = torch.tensor(t["plans"]["logged"], device="cuda", dtype=torch.float32)[None]
            devs.append(float((E[0, P + 1:, :2] - lg[0]).norm(dim=-1).max()))
            ahead = (M[0, :, P] & (A[0, :, P, 0] > 0)).cpu().numpy()
            LW = S[0].cpu().numpy()
            gtcv = cv_base(A, M, P, 6)[0].cpu().numpy()
            preds = {}
            aa = sorted(g.a.unique())
            n = len(aa)
            Eb = torch.cat([plan_E(E, acc_wp(lg, a)) for a in aa])
            with torch.no_grad():
                for net in nets:
                    Y = rollout(net, Eb, A.expand(n, -1, -1, -1), M.expand(n, -1, -1), S.expand(n, -1, -1))
                    preds[net.mode] = {a: Y[q].cpu().numpy() for q, a in enumerate(aa)}
            for _, r in g.iterrows():
                plan = (lg[0] + torch.tensor([[0.5 * r.a * (0.5 * (k + 1)) ** 2, 0.0] for k in range(6)],
                                             device="cuda")).cpu().numpy().tolist()

                def depth(Y):
                    worst = -99.0
                    for h in (1, 2, 3):
                        x, y, _ = plan_pose(plan, h)
                        for j in np.where(ahead)[0]:
                            lx, ly = Y[j, 2 * h - 1]
                            L, W = LW[j]
                            if abs(ly - y) <= (EGO_W + W) / 2:
                                worst = max(worst, min(x + FRONT, lx + L / 2) - max(x + REAR, lx - L / 2))
                    return worst
                o = {k: r[k] for k in ("token", "scene", "stratum", "a", "y", "physics", "h_gap0", "const_a",
                                       "self_orig", "cvref")}
                o["gtcv"] = depth(gtcv)
                for m, pa in preds.items():
                    o[f"ow_{m}"] = depth(pa[r.a])
                rows.append(o)
    print(f"missing windows {nmiss}; logged-plan vs ego-pose max deviation median {np.median(devs):.3f} m, "
          f"p95 {np.quantile(devs, .95):.3f} m")
    pd.DataFrame(rows).to_parquet(out)
    print(f"{len(rows)} candidates -> {out}")


def score(path):
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
    from marg_deep import boot
    df = pd.read_parquet(path)
    js = [c for c in df.columns if c.startswith("ow_")] + ["gtcv", "physics", "self_orig", "cvref", "h_gap0"]
    pairs = [("ow_step0", "ow_chunk"), ("ow_step0", "ow_marg"), ("ow_step0", "gtcv"), ("ow_marg", "ow_chunk"),
             ("ow_chunk", "gtcv"), ("ow_step0", "physics")]
    boot(df, js, [p for p in pairs if p[0] in df and p[1] in df])


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "build":
        build(sys.argv[2], sys.argv[3])
    elif st == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4], *(int(x) for x in sys.argv[5:]))
    elif st == "leak":
        leak(sys.argv[2], sys.argv[4:], sys.argv[3])
    elif st == "judge":
        a = sys.argv[2:]
        k = a.index("--ckpt")
        judge(a[0], a[1], a[2:k], a[k + 1:])
    elif st == "score":
        score(sys.argv[2])
