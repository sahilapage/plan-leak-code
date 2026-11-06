"""Plan-conditioned vs plan-free world model with IDENTICAL inputs (perfect Occ3D history): DOME (conditioned on the plan
it judges) vs COME (scene-centric UNet forecaster: future occupancy in the t0 frame from the past only; the plan enters
only through geometry). Same DOME real-car windows (src/dome_relation_dump.py dirs; src/realhazard.py selection).
  come  <dir> <fold>   COME forecast per window, t0 frame, no warp: plan/<tok>_come.npz veh (4, 200, 200) [t0, +1, +2, +3]
  dome  <dir> <fold>   DOME forecast under every fixed-acceleration plan (audit3/fixed_acc.py ACCS):
                       plan/<tok>_acc.npz veh (len(ACCS), 4, 200, 200), each in its own plan's frame
  score <dir> <fold> [<dir> <fold> ...]
       judges: dome_self (each plan in its own DOME forecast), dome_cvref (src/dome_roc.py CV forecast), dome_oracle
       (logged forecast), come (plan-free forecast, candidate box placed in the t0 frame), physics (detections at
       constant velocity), heuristics; metrics: naive AUROC (collide vs logged), speed-blind AUROC, planner crash rate.
Usage (server, vla env): python src/gt_judges.py come|dome <dir> <fold> | score <dir> <fold> ..."""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
import dome_infer as di  # noqa: E402
import realhazard as rh  # noqa: E402
from dome_roc import auroc, poses  # noqa: E402
from fixed_acc import ACCS, label, strat  # noqa: E402
from planswap import ego_box_in  # noqa: E402
from sw_nodet import physics  # noqa: E402
from sw_realhazard import wp_from_poses  # noqa: E402


def acc_poses(lp, a):
    return lp[:4] + [rh.shift_pose(lp[k], lp[3], 0.5 * a * (0.5 * (k - 3)) ** 2) for k in range(4, 10)]


def run_come(d, fold, bs=2):
    import come_infer as ci
    import types
    sys.path.insert(0, str(ci.COME / "occforecasting"))
    sys.path.insert(0, str(ci.DROOT / "third_party" / "pylibs"))
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
    net = MODELS.build(cfg.model).cuda().eval()
    ck = torch.load(ci.COME / "work_dir" / "unet" / "unet_past2s_future3s.pth", map_location="cpu", weights_only=False)
    print(net.load_state_dict(ck.get("state_dict", ck.get("model", ck)), strict=False))
    d = Path(d)
    W = pickle.load(open(d / "windows.pkl", "rb"))
    infos, _ = di.windows({fold})
    vh = torch.tensor(di.VEHICLE, device="cuda")
    toks = list(W)
    with torch.no_grad():
        for b0 in range(0, len(toks), bs):
            batch = [W[t] for t in toks[b0: b0 + bs]]
            src, smetas = [], []
            for s in batch:
                fr = infos[s["scene"]][s["idx"]: s["idx"] + 10]
                src.append(np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]).transpose(0, 3, 2, 1))
                smetas.append({"ego2global": [di.pose_of(f) for f in fr[:4]]})       # PAST poses only
            inputs = {"source_occs": torch.from_numpy(np.stack(src).astype(np.int64)), "source_metas": smetas,
                      "metas": [{"scene_range": ci.SCENE_RANGE}] * len(batch)}
            x, _ = net.pre_process(inputs)
            B, T, C, Z, H, Wd = x.shape
            x = x.reshape(B, -1, H, Wd)
            enc = []
            for e in net.encoder:
                x = e(x)
                enc.append(x)
            for i in reversed(range(len(net.decoder))):
                x = net.decoder[i](enc[i], x)
            tT, tZ, tH, tW = net.target_seq_size
            x = net.temporal_stem(x).reshape(B * tT, -1, H, Wd)
            logits = net.occ_sem_conv(net.temporal_shared_branch(x)).reshape(B, tT, -1, tZ, H, Wd)[..., :tH, :tW]
            pv = logits.float().softmax(dim=2)[:, :, vh].sum(2)                                   # (B, 6, Z, Y, X)
            for b, s in enumerate(batch):
                res = np.zeros((4, 200, 200), np.float32)
                cur = torch.from_numpy(src[b][3]).cuda()
                res[0] = torch.isin(cur, vh)[di.ZLO:di.ZHI + 1].any(0).float().T.cpu().numpy()
                for k, f in enumerate((1, 3, 5)):
                    res[k + 1] = pv[b, f][di.ZLO:di.ZHI + 1].amax(0).T.cpu().numpy()            # t0 frame
                np.savez_compressed(d / "plan" / f"{s['token']}_come.npz", veh=res.astype(np.float16))
            print(f"COME {b0 + len(batch)}/{len(toks)}", flush=True)


def run_dome(d, fold, bs=2):
    import probe_insert as pi
    d = Path(d)
    W = pickle.load(open(d / "windows.pkl", "rb"))
    infos, _ = di.windows({fold})
    net = pi.Dome()
    toks = [t for t in W if not (d / "plan" / f"{t}_acc.npz").exists()]
    for t in toks:
        s = W[t]
        fr = infos[s["scene"]][s["idx"]: s["idx"] + 10]
        lp = [di.pose_of(f) for f in fr]
        occ = np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]])
        out = []
        for b0 in range(0, len(ACCS), bs):
            accs = ACCS[b0: b0 + bs]
            F = net([occ] * len(accs), [acc_poses(lp, a) for a in accs], seed=0)
            out += [F[i][:4] for i in range(len(accs))]
        np.savez_compressed(d / "plan" / f"{t}_acc.npz", veh=np.stack(out).astype(np.float16))
        print(f"DOME acc {toks.index(t) + 1}/{len(toks)}", flush=True)


def box_max(M, P_frame, P_cand):
    m = ego_box_in(P_frame, P_cand)
    return float(M[m].max()) if m.any() else 0.0


def table(d, fold):
    d = Path(d)
    W = pickle.load(open(d / "windows.pkl", "rb"))
    infos, _ = di.windows({fold})
    naive, fixed = [], []
    for tok, s in W.items():
        fp = {k: d / "plan" / f"{tok}_{k}.npz" for k in ("maps", "cv", "come", "acc")}
        if not all(p.exists() for p in fp.values()):
            continue
        V = np.load(fp["maps"])["veh"].astype(np.float32)        # (2, 1, 4, ...): logged, collide
        Vcv = np.load(fp["cv"])["veh"].astype(np.float32)
        Vc = np.load(fp["come"])["veh"].astype(np.float32)
        Va = np.load(fp["acc"])["veh"].astype(np.float32)
        lp, cp, cv = poses(infos, s)[:3]
        P0 = lp[3]
        t = {"track": s["lead_track"], "LW": s["lead_LW"], "plans": {"logged": wp_from_poses(P0, lp[4:10])},
             "dets": s["dets"]}

        def judges(Pc, self_map):
            r = {"dome_self": 0.0, "dome_cvref": 0.0, "dome_oracle": 0.0, "come": 0.0}
            for h in (1, 2, 3):
                Ph = Pc[3 + 2 * h]
                r["dome_self"] = max(r["dome_self"], box_max(self_map(h), Ph, Ph))
                r["dome_cvref"] = max(r["dome_cvref"], box_max(Vcv[h], cv[3 + 2 * h], Ph))
                r["dome_oracle"] = max(r["dome_oracle"], box_max(V[0, 0, h], lp[3 + 2 * h], Ph))
                r["come"] = max(r["come"], box_max(Vc[h], P0, Ph))
            return r
        for cand, Pc, k in (("collide", cp, 1), ("logged", lp, 0)):
            naive.append({"token": tok, "scene": s["scene"], "cand": cand,
                          **judges(Pc, lambda h, k=k: V[k, 0, h])})
        for j, a in enumerate(ACCS):
            plan = [[x + 0.5 * a * (0.5 * (q + 1)) ** 2, y] for q, (x, y) in enumerate(t["plans"]["logged"])]
            y = label(t, plan)
            if y < 0:
                continue
            Pc = acc_poses(lp, a)
            fixed.append({"token": tok, "scene": s["scene"], "a": a, "y": y, "physics": physics(t, plan),
                          "h_gap0": -s["lead_track"][0][0], "h_v_ego": t["plans"]["logged"][0][0] / 0.5,
                          **judges(Pc, lambda h, j=j: Va[j, h])})
    return pd.DataFrame(naive), pd.DataFrame(fixed)


def score(args):
    from planner_select import curve, at_progress
    N, Fx = [], []
    for i in range(0, len(args), 2):
        n, f = table(args[i], int(args[i + 1]))
        N.append(n)
        Fx.append(f)
    N, Fx = pd.concat(N), pd.concat(Fx)
    Fx.to_parquet(Path(args[0]).parent / "gt_judges_fixed.parquet")
    js = ["dome_self", "dome_cvref", "dome_oracle", "come"]
    rng = np.random.default_rng(0)
    sc = Fx.scene.unique()
    g = {s: Fx[Fx.scene == s] for s in sc}
    res = [pd.concat([g[s] for s in rng.choice(sc, len(sc))]) for _ in range(1000)]
    print(f"{Fx.token.nunique()} windows, {len(sc)} scenes, {len(Fx)} fixed-acc candidates ({int(Fx.y.sum())} crash)")
    for j in js + ["physics", "h_gap0", "h_v_ego"]:
        nv = ""
        if j in js:
            p, q = N[N.cand == "collide"][j], N[N.cand == "logged"][j]
            nv = f"naive AUROC {auroc(p, q):.3f} | "
        reps = [strat(r, j) for r in res]
        print(f"  {j:12s} {nv}speed-blind {strat(Fx, j):.3f} [{np.nanquantile(reps, 0.025):.3f}, "
              f"{np.nanquantile(reps, 0.975):.3f}] | planner crash @ +1.5 {at_progress(curve(Fx, j), 1.5):.3f}")
    for a, b in (("come", "dome_self"), ("come", "dome_cvref"), ("come", "physics")):
        reps = [strat(r, a) - strat(r, b) for r in res]
        print(f"  speed-blind {a} - {b}: {strat(Fx, a) - strat(Fx, b):+.3f} [{np.nanquantile(reps, 0.025):+.3f}, "
              f"{np.nanquantile(reps, 0.975):+.3f}]")


if __name__ == "__main__":
    if sys.argv[1] == "come":
        run_come(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "dome":
        run_dome(sys.argv[2], int(sys.argv[3]))
    else:
        score(sys.argv[2:])
