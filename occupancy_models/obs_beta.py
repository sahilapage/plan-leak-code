"""The observational leak the logs TEACH, per agent relation: beta_obs(h) = slope of the agent's future x-displacement
residual on the ego's, both residuals taken against each one's own constant-velocity extrapolation from the past
(what history alone predicts), through the origin, h = 1 / 2 / 3 s. A model trained on logs to predict
E[agent future | history, ego future] learns beta_obs; the causal effect of the ego plan on a lead is ~0. Compared with
the measured model leak (audit3/relation_leak.py: SWTC lead 1.01, follower 0.75, adjacent 0.69, oncoming -0.21, parked
0.07). Agents and classes as relation_leak.py (isolated and all). Velocities at t0 from the annotation 0.5 s before.
Two populations: (1) the swrh windows (val, the model-side windows), (2) nuScenes TRAIN windows (what the models were
trained on), to show the coupling is a property of the training logs.
Two bounds: 'resid' = what a model using motion history perfectly learns; 'total' = OLS (with intercept) of the
agent's total x-displacement on the ego's, what a model blind to motion history learns.
Usage (server, vla env): python audit3/obs_beta.py <swrh_dir> ... [--train N]"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import realhazard as rh  # noqa: E402
from relation_leak import classify  # noqa: E402


def prev_xy(nusc, ann, P0):
    if not ann["prev"]:
        return None
    a = nusc.get("sample_annotation", ann["prev"])
    g = np.linalg.inv(P0) @ np.array([*a["translation"][:2], 0.0, 1.0])
    return g[:2]


def window_rows(nusc, tok, scene, lp, k0):
    """lp: poses of the scene's frames; k0: index of t0 (needs k0-1 and k0+6)."""
    P0, Pm = lp[k0], lp[k0 - 1]
    Ti = np.linalg.inv(P0)
    ego = np.array([(Ti @ lp[k0 + j][:, 3])[0] for j in range(7)])
    ego_prev = (Ti @ Pm[:, 3])[0]
    v_ego = (0.0 - ego_prev) / 0.5
    agents, out = [], []
    for at in nusc.get("sample", tok)["anns"]:
        ann = nusc.get("sample_annotation", at)
        if not ann["category_name"].startswith(rh.VEH_CATS):
            continue
        tr = rh.lead_track(nusc, ann, P0)
        pv = prev_xy(nusc, ann, P0)
        if tr is not None and pv is not None:
            agents.append((tr, pv))
    for k, (tr, pv) in enumerate(agents):
        rel = classify(tr)
        if rel is None:
            continue
        iso = all(not ((np.abs(o[:, 1] - tr[:, 1]) <= 2.5) & (np.abs(o[:, 0] - tr[:, 0]) <= 25)).any()
                  for j, (o, _) in enumerate(agents) if j != k)
        v_ag = (tr[0, 0] - pv[0]) / 0.5
        for h in (1, 2, 3):
            out.append({"scene": scene, "rel": rel, "iso": iso, "h": h,
                        "r_ag": tr[2 * h, 0] - (tr[0, 0] + v_ag * h), "r_ego": ego[2 * h] - v_ego * h,
                        "d_ag": tr[2 * h, 0] - tr[0, 0], "d_ego": ego[2 * h]})
    return out


def slope(f):
    return float((f.r_ag * f.r_ego).sum() / (f.r_ego ** 2).sum())


def slope_total(f):
    x = f.d_ego - f.d_ego.mean()
    return float((x * (f.d_ag - f.d_ag.mean())).sum() / (x * x).sum())


def boot(f, fn=slope, B=2000):
    rng = np.random.default_rng(0)
    sc = f.scene.unique()
    g = {s: f[f.scene == s] for s in sc}
    v = [fn(pd.concat([g[s] for s in rng.choice(sc, len(sc))])) for _ in range(B)]
    q = np.quantile(v, [0.025, 0.975])
    return f"{fn(f):+.2f} [{q[0]:+.2f}, {q[1]:+.2f}]"


def main():
    from nuscenes.nuscenes import NuScenes
    args = sys.argv[1:]
    n_train = 0
    if "--train" in args:
        i = args.index("--train")
        n_train = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    nusc = NuScenes("v1.0-trainval", dataroot=str(rh.DATA / "nuscenes"), verbose=False)
    rows = []
    for d in map(Path, args):
        truth = pickle.load(open(d / "truth.pkl", "rb"))
        toks = set(json.load(open(d / "soft_tokens.json")))
        for tok in toks:
            sd = nusc.get("sample_data", nusc.get("sample", tok)["data"]["LIDAR_TOP"])
            samp = nusc.get("sample", tok)
            if not samp["prev"]:
                continue
            seq = [nusc.get("sample", samp["prev"]), samp]
            while len(seq) < 8 and seq[-1]["next"]:
                seq.append(nusc.get("sample", seq[-1]["next"]))
            if len(seq) < 8:
                continue
            lp = [di.pose_of(_ego(nusc, s)) for s in seq]
            rows += [dict(r, pop="val_windows") for r in window_rows(nusc, tok, truth[tok]["scene"], lp, 1)]
            del sd
    if n_train:
        rng = np.random.default_rng(0)
        val = set(_val_scenes())
        scenes = [s for s in nusc.scene if s["name"] not in val]
        for s in rng.permutation(len(scenes))[:n_train]:
            sc = scenes[s]
            seq = [nusc.get("sample", sc["first_sample_token"])]
            while seq[-1]["next"]:
                seq.append(nusc.get("sample", seq[-1]["next"]))
            lp = [di.pose_of(_ego(nusc, x)) for x in seq]
            for k0 in range(1, len(seq) - 7, 4):
                Ti = np.linalg.inv(lp[k0])
                if (Ti @ lp[k0 + 6][:, 3])[0] < 9.0:                   # ego moving >= 3 m/s on average
                    continue
                rows += [dict(r, pop="train_scenes") for r in window_rows(nusc, seq[k0]["token"], sc["name"], lp, k0)]
    df = pd.DataFrame(rows)
    df.to_parquet(Path(args[0]).parent / "obs_beta.parquet")
    for pop, P in df.groupby("pop"):
        print(f"== {pop}: {P.scene.nunique()} scenes")
        for rel in ("lead", "follower", "adjacent", "oncoming", "parked"):
            line = []
            for h in (1, 2, 3):
                f = P[(P.rel == rel) & (P.h == h)]
                fi = f[f.iso]
                if len(f) >= 10:
                    line.append(f"+{h} s resid {boot(f)} total {boot(f, slope_total)} (n {len(f)}) | iso resid "
                                f"{boot(fi) if len(fi) >= 10 else 'na'} total "
                                f"{boot(fi, slope_total) if len(fi) >= 10 else 'na'}")
            print(f"{rel:9s} beta_obs: " + " || ".join(line))


def _ego(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return {"ego2global_translation": ep["translation"], "ego2global_rotation": ep["rotation"]}


def _val_scenes():
    from nuscenes.utils.splits import val
    return val


if __name__ == "__main__":
    main()
