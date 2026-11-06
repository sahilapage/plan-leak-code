"""Does the past-only reference fix transfer to DOME? On the DOME real-car windows (src/dome_relation_dump.py dirs):
  run   : one more DOME forecast per window under the past-only constant-velocity plan (v0 = forward displacement over
          the last 0.5 s, kept along the t0 heading; past frames unchanged), saved as plan/<tok>_cv.npz (veh (4, 200,
          200) float16, own frame).
  score : danger(candidate) = max over +1 / +2 / +3 s of the vehicle probability inside the candidate's ego box placed
          in the judging forecast's frame: self (forecast under the candidate), cvref (the CV forecast), oracle (the
          logged forecast). AUROC collide vs logged, recall at 5 / 10 % FA, paired scene bootstrap of differences.
Usage (server, vla env): python src/dome_roc.py run <dir> <fold>
                         python src/dome_roc.py score <dir> <fold> [<dir> <fold> ...]"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
import dome_infer as di  # noqa: E402
import probe_insert as pi  # noqa: E402
import realhazard as rh  # noqa: E402
from planswap import ego_box_in  # noqa: E402


def poses(infos, s):
    lp = [di.pose_of(f) for f in infos[s["scene"]][s["idx"]: s["idx"] + 10]]
    P0 = lp[3]
    cp = lp[:4] + [rh.shift_pose(lp[k], P0, 0.5 * s["a"] * (0.5 * (k - 3)) ** 2) for k in range(4, 10)]
    v0 = float((np.linalg.inv(lp[2]) @ P0[:, 3])[0]) / 0.5
    cv = lp[:4] + [rh.shift_pose(P0, P0, max(v0, 0.0) * 0.5 * (k - 3)) for k in range(4, 10)]
    return lp, cp, cv


def run(d, fold, bs=4):
    d = Path(d)
    W = pickle.load(open(d / "windows.pkl", "rb"))
    infos, _ = di.windows({fold})
    net = pi.Dome()
    toks = [t for t in W if not (d / "plan" / f"{t}_cv.npz").exists()]
    for b0 in range(0, len(toks), bs):
        batch = [W[t] for t in toks[b0: b0 + bs]]
        occ = [np.stack([di.load_occ(s["scene"], f["token"]) for f in infos[s["scene"]][s["idx"]: s["idx"] + 4]])
               for s in batch]
        F = net(occ, [poses(infos, s)[2] for s in batch], seed=b0)
        for i, s in enumerate(batch):
            np.savez_compressed(d / "plan" / f"{s['token']}_cv.npz", veh=F[i][:4].astype(np.float16))
        print(f"{b0 + len(batch)}/{len(toks)}", flush=True)


def scores(d, fold):
    d = Path(d)
    W = pickle.load(open(d / "windows.pkl", "rb"))
    infos, _ = di.windows({fold})
    rows = []
    for tok, s in W.items():
        fp, fc = d / "plan" / f"{tok}_maps.npz", d / "plan" / f"{tok}_cv.npz"
        if not (fp.exists() and fc.exists()):
            continue
        V = np.load(fp)["veh"].astype(np.float32)          # (2, 1, 4, 200, 200): logged, collide
        Vcv = np.load(fc)["veh"].astype(np.float32)        # (4, 200, 200)
        lp, cp, cv = poses(infos, s)
        r = {"token": tok, "scene": s["scene"]}
        for judge in ("self", "cvref", "oracle"):
            for cand, Pc in (("collide", cp), ("logged", lp)):
                best = 0.0
                for h in (1, 2, 3):
                    Ph = Pc[3 + 2 * h]
                    if judge == "self":
                        M, Pf = V[0 if cand == "logged" else 1, 0, h], Ph
                    elif judge == "cvref":
                        M, Pf = Vcv[h], cv[3 + 2 * h]
                    else:
                        M, Pf = V[0, 0, h], lp[3 + 2 * h]
                    box = ego_box_in(Pf, Ph)
                    if box.any():
                        best = max(best, float(M[box].max()))
                r[f"{judge}_{cand}"] = best
        rows.append(r)
    return pd.DataFrame(rows)


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def score(args):
    df = pd.concat([scores(args[i], int(args[i + 1])) for i in range(0, len(args), 2)])
    sc = df.scene.unique()
    g = {s: df[df.scene == s] for s in sc}
    rng = np.random.default_rng(0)
    res = [pd.concat([g[s] for s in rng.choice(sc, len(sc))]) for _ in range(2000)]
    print(f"DOME: {len(df)} windows, {len(sc)} scenes")
    A = {j: auroc(df[f"{j}_collide"], df[f"{j}_logged"]) for j in ("self", "cvref", "oracle")}
    for j in A:
        reps = [auroc(f[f"{j}_collide"], f[f"{j}_logged"]) for f in res]
        rec = []
        for fa in (0.05, 0.10):
            th = np.quantile(df[f"{j}_logged"], 1 - fa)
            rec.append(f"rec@{int(fa * 100)} {(df[f'{j}_collide'] > th).mean():.3f}")
        print(f"  {j:6s} AUROC {A[j]:.3f} [{np.quantile(reps, 0.025):.3f}, {np.quantile(reps, 0.975):.3f}] | "
              + " | ".join(rec))
    for a, b in (("cvref", "self"), ("cvref", "oracle")):
        reps = [auroc(f[f"{a}_collide"], f[f"{a}_logged"]) - auroc(f[f"{b}_collide"], f[f"{b}_logged"]) for f in res]
        print(f"  AUROC {a} - {b}: {A[a] - A[b]:+.3f} [{np.quantile(reps, 0.025):+.3f}, {np.quantile(reps, 0.975):+.3f}]")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]))
    else:
        score(sys.argv[2:])
