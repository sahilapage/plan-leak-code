"""Plan-marginal judging (identification-correct estimator for non-reactive agents): p(world | history, do(plan))
= p(world | history) for agents the ego cannot affect, so a candidate is judged in the forecast MARGINALISED over
plans drawn from a past-only plan prior, not in the forecast conditioned on itself. Monte Carlo: K past-only plans
(constant acceleration a_k around the past speed v0, a_k in PRIOR; straight along the t0 heading); score(candidate) =
mean over k of the max obstacle probability in the candidate's ego box placed in forecast k (geometry only).
The single CV plan (src/sw_hetero.py cv, cvref) is the K = 1, a = 0 special case.
  prep  <het_dir>        <het_dir>/marg/{cands.jsonl, ids.txt, truth.pkl}  (K plans per window, for --k K)
  score <het_dir> <name>=<marg_soft_dir> ...   adds cvmarg_<name> to <het_dir>/table.parquet's rows, prints gain
Usage (server, vla env): python src/het_marg.py prep <het_dir> | score <het_dir> orig=<dir> rpt=<dir>"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
from fixed_acc import ACCS, acc_plan  # noqa: E402
from sw_hetero import ego_max, plan_pose  # noqa: E402

PRIOR = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)          # m/s^2 around the past speed (past-only, plan-free)


def mplan(v0, a):
    """waypoints at 0.5 .. 3 s of a constant-acceleration plan from speed v0 (stops at 0 m/s, never reverses)."""
    out = []
    for t in 0.5 * np.arange(1, 7):
        s = v0 * t + 0.5 * a * t * t if (a >= 0 or v0 + a * t > 0) else v0 * v0 / (-2.0 * a)
        out.append([float(max(s, 0.0)), 0.0])
    return out


def prep(het):
    het = Path(het)
    T = pickle.load(open(het / "acc" / "truth.pkl", "rb"))
    d = het / "marg"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cands.jsonl", "w") as f:
        for tok, t in T.items():
            f.write(json.dumps({"id": tok, "cands": [mplan(t["v0"], a) for a in PRIOR]}) + "\n")
    shutil.copy(het / "acc" / "ids.txt", d / "ids.txt")
    shutil.copy(het / "acc" / "truth.pkl", d / "truth.pkl")
    print(f"prepared {len(T)} windows x {len(PRIOR)} prior plans")


def score(het, specs):
    from planner_select import at_crash, curve
    from sw_hetero import gain
    het = Path(het)
    T = pickle.load(open(het / "acc" / "truth.pkl", "rb"))
    df = pd.read_parquet(het / "table.parquet")
    for spec in specs:
        k, d = spec.split("=", 1)
        S = np.load(Path(d) / "soft.npy", mmap_mode="r")
        ix = {t: i for i, t in enumerate(json.load(open(Path(d) / "soft_tokens.json")))}
        col = []
        for tok, a in zip(df.token, df.a):
            if tok not in ix:
                col.append(np.nan)
                continue
            t, i = T[tok], ix[tok]
            plan = acc_plan(t["plans"]["logged"], a)
            per = [max(ego_max(np.asarray(S[i, q, h - 1], np.float32), plan_pose(mplan(t["v0"], pa), h), plan, h)
                       for h in (1, 2, 3)) for q, pa in enumerate(PRIOR)]
            col.append(float(np.mean(per)))
        df[f"cvmarg_{k}"] = col
    df.to_parquet(het / "table_marg.parquet")
    js = [c for c in df.columns if c.startswith(("cvref", "cvmarg", "self_"))] + ["physics"]
    print(f"{het}: {df.token.nunique()} windows")
    for j in js:
        print(f"  {j:14s} gain {gain(df, j):+.3f} | progress@<=2% {at_crash(curve(df, j), 0.02):.2f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2])
    else:
        score(sys.argv[2], sys.argv[3:])
