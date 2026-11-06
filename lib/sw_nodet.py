"""Fair-physics test: real-car windows selected WITHOUT requiring the camera detector to see the lead (src/realhazard.py
REQUIRE_DET = False; everything else as src/sw_realhazard.py, X_MAX off). Earlier windows required the lead detected at
t0, which hands a detection-based physics judge the hazard by construction.
  prep  <fold> <n> <dir>   truth.pkl (lead track, size, logged plan, detections, lead_det_err = distance from the
                           lead's true t0 centre to the nearest detection), ids.txt, cands.jsonl = [logged] + the
                           fixed-acceleration plans of audit3/fixed_acc.py (for src/swtc_verifier.py --k 12 --soft-all)
  score <dir> name=<soft_dir> [name=<soft_dir> ...]
           per window x a: kinematic crash label (audit3/fixed_acc.py); judges: self_<name> (each plan in its own
           forecast), oracle_<name> (the plan's ego box in the LOGGED-plan forecast, soft index 0), physics (detections at
           constant velocity, overlap depth), heuristics; speed-blind AUROC and the planner's crash rate at matched
           progress, separately for leads detected (err <= 2 m) and missed / mislocated (> 2 m).
Usage (server, vla env): python src/sw_nodet.py prep <fold> <n> <dir> | score <dir> name=<soft_dir> ..."""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
from fixed_acc import ACCS, FRONT, acc_plan, au, label, strat  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, box_in_frame, plan_pose, wp_from_poses  # noqa: E402


def prep(fold, n, out):
    import dome_infer as di
    import realhazard as rh
    rh.X_MAX, rh.REQUIRE_DET = 1e9, False
    infos, sel = rh.select(fold, n)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    with open(out / "cands.jsonl", "w") as f:
        for s in sel:
            lp = [di.pose_of(x) for x in infos[s["scene"]][s["idx"]: s["idx"] + 10]]
            logged = wp_from_poses(lp[3], lp[4:10])
            f.write(json.dumps({"id": s["token"], "cands": [logged] + [acc_plan(logged, a) for a in ACCS]}) + "\n")
            truth[s["token"]] = {"scene": s["scene"], "track": s["lead_track"], "LW": s["lead_LW"], "a": s["a"],
                                 "plans": {"logged": logged}, "dets": s["dets"], "lead_det_err": s["lead_det_err"]}
    (out / "ids.txt").write_text("\n".join(truth))
    pickle.dump(truth, open(out / "truth.pkl", "wb"))
    e = np.array([t["lead_det_err"] for t in truth.values()])
    print(f"prepared {len(truth)} windows, {len({t['scene'] for t in truth.values()})} scenes; lead detected "
          f"(<= 2 m) {np.mean(e <= 2):.3f}, missed / mislocated {np.sum(e > 2)}", flush=True)


def physics(t, plan, static=False):
    worst = -1e9
    for h in (1, 2, 3):
        x, y, yaw = plan_pose(plan, h)
        front, rear = x + FRONT, x + EGO_C - EGO_L / 2
        for d in t["dets"]:
            k = 0.0 if static else h
            px, py = d["x"] + d["v"][0] * k, d["y"] + d["v"][1] * k
            if abs(py - y) <= (EGO_W + d["W"]) / 2:
                worst = max(worst, min(front, px + d["L"] / 2) - max(rear, px - d["L"] / 2))
    return worst


def box_score(m_list, frame_plan, plan):
    best = 0.0
    for h in (1, 2, 3):
        fx, fy, fyaw = plan_pose(frame_plan, h)
        cx, cy, cyaw = plan_pose(plan, h)
        ob = box_in_frame(fx, fy, fyaw, cx + EGO_C * math.cos(cyaw), cy + EGO_C * math.sin(cyaw), cyaw, EGO_L, EGO_W)
        if ob.any():
            best = max(best, float(np.asarray(m_list[h - 1], np.float32)[ob].max()))
    return best


def table(d, softs):
    d = Path(d)
    T = pickle.load(open(d / "truth.pkl", "rb"))
    rows = {}
    for tok, t in T.items():
        lg = t["plans"]["logged"]
        for a in ACCS:
            plan = acc_plan(lg, a)
            y = label(t, plan)
            if y < 0:
                continue
            rows[(tok, a)] = {"token": tok, "scene": t["scene"], "a": a, "y": y, "det": t["lead_det_err"] <= 2.0,
                              "physics": physics(t, plan), "physics_static": physics(t, plan, True), "h_gap0": -t["track"][0][0], "h_v_ego": lg[0][0] / 0.5}
    for name, sd in softs:
        sd = Path(sd)
        soft = np.load(sd / "soft.npy", mmap_mode="r")
        for i, tok in enumerate(json.load(open(sd / "soft_tokens.json"))):
            lg = T[tok]["plans"]["logged"]
            for j, a in enumerate(ACCS, start=1):
                if (tok, a) not in rows:
                    continue
                plan = acc_plan(lg, a)
                rows[(tok, a)][f"self_{name}"] = box_score(soft[i, j], plan, plan)
                rows[(tok, a)][f"oracle_{name}"] = box_score(soft[i, 0], lg, plan)
    return pd.DataFrame(list(rows.values()))


def planner_crash(df, j, prog=1.5):
    best = np.nan
    s = df[j]
    lab = df.set_index(["token", "a"]).y
    for t in np.unique(np.quantile(s.dropna(), np.linspace(0, 1, 101))):
        pick = df[s < t].groupby("token").a.max().reindex(df.token.unique()).fillna(0.0)
        if pick.mean() >= prog:
            c = np.mean([lab.get((tk, a), 0) if a > 0 else 0 for tk, a in pick.items()])
            best = c if np.isnan(best) else min(best, c)
    return best


def score(d, softs):
    df = table(d, softs)
    judges = [c for c in df.columns if c.startswith(("self_", "oracle_")) or c in ("physics", "physics_static", "h_gap0", "h_v_ego")]
    df.to_parquet(Path(d) / "nodet_table.parquet")
    for j in [c for c in judges if c.startswith("self_")]:
        print(f"Spearman({j}, physics_static) {df[j].corr(df.physics_static, method='spearman'):+.3f} | "
              f"({j}, physics CV) {df[j].corr(df.physics, method='spearman'):+.3f}")
    rng = np.random.default_rng(0)
    for nm, sub in (("all", df), ("lead DETECTED", df[df.det]), ("lead MISSED / mislocated", df[~df.det])):
        if sub.token.nunique() < 5:
            print(f"== {nm}: {sub.token.nunique()} windows (too few)")
            continue
        sc = sub.scene.unique()
        g = {s: sub[sub.scene == s] for s in sc}
        res = [pd.concat([g[s] for s in rng.choice(sc, len(sc))]) for _ in range(1000)]
        print(f"== {nm}: {sub.token.nunique()} windows, {len(sc)} scenes, {len(sub)} candidates ({int(sub.y.sum())} crash)")
        for j in judges:
            f = sub.dropna(subset=[j])
            if not len(f):
                continue
            reps = [strat(r.dropna(subset=[j]), j) for r in res]
            print(f"  {j:18s} speed-blind AUROC {strat(f, j):.3f} [{np.nanquantile(reps, 0.025):.3f}, "
                  f"{np.nanquantile(reps, 0.975):.3f}] | planner crash @ +1.5 {planner_crash(f, j):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        score(sys.argv[2], [tuple(a.split("=", 1)) for a in sys.argv[3:]])
