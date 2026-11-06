"""Speed-blind protocol (audit3/fixed_acc.py), the remaining judges on the same windows / plans / labels:
  prep    <swrh_dir> <out_dir>   cands.jsonl = [logged] + the fixed-acceleration plans (ACCS) for src/swtc_verifier.py
                                 (--k 12 --soft-all); then the SELF score of plan a = max obstacle probability in its own
                                 ego box in its own forecast.
  physics <swrh_dir> <fold>      past-only physics on detections (SparseDrive at t0, as audit3/sw_physics.py det_cv):
                                 every detection extrapolated at its t0 velocity; score = max over +1 / +2 / +3 s of
                                 the longitudinal overlap depth (negative = gap) between the candidate's ego box and
                                 the nearest laterally overlapping extrapolated box (continuous; higher = more dangerous).
  score   <swrh_dir> <swcv_dir> <fold> [self=<acc_dir> ...] [physics]
Usage: see above (server; vla env for prep / physics / score)."""
import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_acc import ACCS, FRONT, acc_plan, au, label, rows_for, strat  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, box_in_frame, plan_pose  # noqa: E402


def prep(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    T = pickle.load(open(src / "truth.pkl", "rb"))
    with open(out / "cands.jsonl", "w") as f:
        for tok, t in T.items():
            lg = t["plans"]["logged"]
            f.write(json.dumps({"id": tok, "cands": [lg] + [acc_plan(lg, a) for a in ACCS]}) + "\n")
    for n in ("truth.pkl", "ids.txt"):
        shutil.copy(src / n, out / n)
    print("prepared", len(T), "windows,", 1 + len(ACCS), "plans each")


def self_scores(d):
    d = Path(d)
    T = pickle.load(open(d / "truth.pkl", "rb"))
    soft = np.load(d / "soft.npy", mmap_mode="r")
    out = {}
    for i, tok in enumerate(json.load(open(d / "soft_tokens.json"))):
        lg = T[tok]["plans"]["logged"]
        for j, a in enumerate(ACCS, start=1):
            plan, best = acc_plan(lg, a), 0.0
            for h in (1, 2, 3):
                fx, fy, fyaw = plan_pose(plan, h)
                ob = box_in_frame(fx, fy, fyaw, fx + EGO_C * math.cos(fyaw), fy + EGO_C * math.sin(fyaw), fyaw,
                                  EGO_L, EGO_W)
                best = max(best, float(np.asarray(soft[i, j, h - 1], np.float32)[ob].max()))
            out[(tok, a)] = best
    return out


def physics_scores(src, fold):
    import realhazard as rh
    T = pickle.load(open(Path(src) / "truth.pkl", "rb"))
    rh.X_MAX = 1e9
    _, sel = rh.select(fold, len(T))
    out = {}
    for s in sel:
        if s["token"] not in T:
            continue
        lg = T[s["token"]]["plans"]["logged"]
        for a in ACCS:
            plan, worst = acc_plan(lg, a), -1e9
            for h in (1, 2, 3):
                x, y, yaw = plan_pose(plan, h)
                front, rear = x + FRONT, x + EGO_C - EGO_L / 2
                for dt in s["dets"]:
                    px, py = dt["x"] + dt["v"][0] * h, dt["y"] + dt["v"][1] * h
                    if abs(py - y) <= (EGO_W + dt["W"]) / 2:            # longitudinal interval overlap depth
                        worst = max(worst, min(front, px + dt["L"] / 2) - max(rear, px - dt["L"] / 2))
            out[(s["token"], a)] = worst
    return out


def score(args):
    src, cvd, fold = args[0], args[1], int(args[2])
    df = pd.DataFrame(rows_for(src, cvd))
    for extra in args[3:]:
        if extra.startswith("self="):
            name, d = extra.split("=", 1)
            sc = self_scores(d)
            df[f"self_{Path(d).name}"] = [sc.get((t, a), np.nan) for t, a in zip(df.token, df.a)]
        elif extra == "physics":
            sc = physics_scores(src, fold)
            df["physics"] = [sc.get((t, a), np.nan) for t, a in zip(df.token, df.a)]
    T = pickle.load(open(Path(src) / "truth.pkl", "rb"))
    df["h_gap0"] = [-T[t]["track"][0][0] for t in df.token]                     # privileged-free heuristics
    df["h_v_ego"] = [T[t]["plans"]["logged"][0][0] / 0.5 for t in df.token]    # (speed over the first 0.5 s)
    judges = [c for c in df.columns if c in ("cvref", "oracle", "physics", "h_gap0", "h_v_ego") or c.startswith("self_")]
    sc_ = df.scene.unique()
    g = {s: df[df.scene == s] for s in sc_}
    rng = np.random.default_rng(0)
    res = [pd.concat([g[s] for s in rng.choice(sc_, len(sc_))]) for _ in range(2000)]
    print(f"{df.token.nunique()} windows, {len(sc_)} scenes, {len(df)} candidates ({int(df.y.sum())} crash)")
    for j in judges:
        f = df.dropna(subset=[j])
        reps = [strat(r.dropna(subset=[j]), j) for r in res]
        print(f"  {j:16s} SPEED-BLIND AUROC {strat(f, j):.3f} [{np.nanquantile(reps, 0.025):.3f}, "
              f"{np.nanquantile(reps, 0.975):.3f}] | pooled {au(f, j):.3f} (n {len(f)})")
    selfs = [j for j in judges if j.startswith("self_")]
    for j in selfs[1:] + [x for x in ("physics", "cvref", "oracle") if x in judges]:
        base = selfs[0]
        f = df.dropna(subset=[j, base])
        d0 = strat(f, j) - strat(f, base)
        reps = [strat(r.dropna(subset=[j, base]), j) - strat(r.dropna(subset=[j, base]), base) for r in res]
        print(f"  paired speed-blind AUROC {j} - {base}: {d0:+.3f} [{np.nanquantile(reps, 0.025):+.3f}, "
              f"{np.nanquantile(reps, 0.975):+.3f}]")
    df.to_parquet(Path(src).parent / f"speedblind_{Path(src).name}.parquet")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2:])
