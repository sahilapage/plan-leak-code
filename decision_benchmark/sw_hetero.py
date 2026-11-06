"""Heterogeneous decision benchmark (audit of audit3/planner_baseline.py: on the moving-lead windows the safe extra
acceleration is nearly the same everywhere, so a plan-blind planner is hard to beat and no judge can show value).
Windows (fold, seed 0, at most CAP per scene): near-straight logged path (heading change < 15 deg, lateral drift < 2 m),
mean ego speed >= 3 m/s, SWTC past speed available; the logged path keeps >= 0.5 m from every annotated vehicle.
Strata by the nearest in-lane vehicle at t0 (|y| <= 1.5 m, 2 <= x <= 50 m, fully annotated for 3 s):
  moving   speed > 1 m/s        stopped   speed <= 1 m/s        free   none (safe unless another vehicle cuts in)
balanced up to n per stratum. Candidates: logged + the fixed extra accelerations of audit3/fixed_acc.py (ACCS).
Truth (non-reactive, every annotated vehicle AHEAD at t0 with a full 3 s track): crash if the ego box's longitudinal
interval overlaps a laterally overlapping vehicle box by >= 1 m at any of +1 / +2 / +3 s; safe if every such
interval keeps a >= 2 m gap; else excluded. Windows whose in-corridor vehicles ahead lack a full track are dropped.
  prep  <fold> <n> <out>   <out>/acc (logged + ACCS, for swtc_verifier --k 12) and <out>/cv (past-speed CV plan, --k 1)
  score <out> <name>=<acc_dir> ... [cv=<cv_dir> cv_<name>=<cv_dir> ...]
        judges self_<name>, oracle (logged forecast of the first acc dir), cvref[_<name>], physics (detections x
        CV), plan-blind; speed-blind AUROC, planner crash vs plan-blind (mean crash reduction over mean chosen a in
        GRID), per stratum.
Usage (server, vla env): python src/sw_hetero.py prep <fold> <n> <out> | score <out> orig=<dir> rpt=<dir> cv=<dir>"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
from fixed_acc import ACCS, FRONT, acc_plan, strat  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, box_in_frame, plan_pose, wp_from_poses  # noqa: E402

CAP = 8
REAR = EGO_C - EGO_L / 2


def depth(vehicles, plan):
    """max over +1 / +2 / +3 s and vehicles of the longitudinal interval overlap (m; negative = gap), lateral overlap
    required; -99 if no vehicle ever overlaps laterally."""
    worst = -99.0
    for h in (1, 2, 3):
        x, y, _ = plan_pose(plan, h)
        for v in vehicles:
            lx, ly, _ = v["track"][2 * h]
            L, W = v["LW"]
            if abs(ly - y) <= (EGO_W + W) / 2:
                worst = max(worst, min(x + FRONT, lx + L / 2) - max(x + REAR, lx - L / 2))
    return worst


def lab(vehicles, plan):
    d = depth(vehicles, plan)
    return 1 if d >= 1.0 else (0 if d <= -2.0 else -1)


def prep(fold, n, out):
    from nuscenes.nuscenes import NuScenes
    import dome_infer as di
    import probe_hazard as ph
    import probe_insert as pi
    import realhazard as rh
    from sw_cvjudge import past_speed
    nusc = NuScenes("v1.0-trainval", dataroot=str(rh.DATA / "nuscenes"), verbose=False)
    sd = pickle.load(open(rh.DROOT / "runs" / "sparsedrive" / "sd_clean.pkl", "rb"))
    infos, wins = di.windows({fold})
    v0s = past_speed([infos[s][i + 3]["token"] for s, i in wins])
    rng = np.random.default_rng(0)
    out = Path(out)
    truth, per_scene, count = {}, {}, {"moving": 0, "stopped": 0, "free": 0}
    for j in rng.permutation(len(wins)):
        scene, idx = wins[j]
        if per_scene.get(scene, 0) >= CAP:
            continue
        fr = infos[scene][idx: idx + 10]
        ps = [di.pose_of(f) for f in fr]
        tok = fr[3]["token"]
        if tok not in v0s:
            continue
        if abs((pi.yaw_of(ps[9]) - pi.yaw_of(ps[0]) + math.pi) % (2 * math.pi) - math.pi) > math.radians(15):
            continue
        if max(abs(ph.along(ps[3], ps[k])[1]) for k in range(4, 10)) > 2.0:
            continue
        logged = wp_from_poses(ps[3], ps[4:10])
        if logged[-1][0] / 3.0 < 3.0:
            continue
        veh, bad, lead = [], False, None
        for at in nusc.get("sample", tok)["anns"]:
            ann = nusc.get("sample_annotation", at)
            if not ann["category_name"].startswith(rh.VEH_CATS):
                continue
            tr = rh.lead_track(nusc, ann, ps[3])
            p0 = np.linalg.inv(ps[3]) @ np.array([*ann["translation"][:2], 0.0, 1.0])
            in_corr = abs(p0[1]) <= 3.0 and 0 < p0[0] <= 60
            if tr is None:
                bad |= in_corr                                   # truth would be incomplete
                continue
            if tr[0, 0] <= 0:
                continue
            v = nusc.box_velocity(at)[:2]
            spd = float(np.hypot(*v)) if np.all(np.isfinite(v)) else float(np.hypot(*(tr[2, :2] - tr[0, :2])))
            rec = {"track": tr, "LW": (ann["size"][1], ann["size"][0]), "speed": spd}
            veh.append(rec)
            x0, y0, yw = tr[0]
            if abs(y0) <= 1.5 and 2 <= x0 <= 50 and (spd <= 1.0 or abs(yw) <= math.radians(30)):
                if lead is None or x0 < lead["track"][0, 0]:
                    lead = rec
        if bad or depth(veh, logged) > -0.5:
            continue
        stratum = "free" if lead is None else ("stopped" if lead["speed"] <= 1.0 else "moving")
        if count[stratum] >= n:
            continue
        count[stratum] += 1
        per_scene[scene] = per_scene.get(scene, 0) + 1
        v0 = v0s[tok]
        truth[tok] = {"scene": scene, "stratum": stratum, "vehicles": veh, "plans": {"logged": logged},
                      "dets": rh.detections(sd, tok), "v0": v0, "cv": [[v0 * 0.5 * (k + 1), 0.0] for k in range(6)],
                      "gap0": float(lead["track"][0, 0]) if lead else 60.0,
                      "track": lead["track"] if lead else None, "LW": lead["LW"] if lead else None}
        if all(c >= n for c in count.values()):
            break
    for sub, cands in (("acc", lambda t: [t["plans"]["logged"]] + [acc_plan(t["plans"]["logged"], a) for a in ACCS]),
                       ("cv", lambda t: [t["cv"]])):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "cands.jsonl", "w") as f:
            for tok, t in truth.items():
                f.write(json.dumps({"id": tok, "cands": cands(t)}) + "\n")
        (d / "ids.txt").write_text("\n".join(truth) + "\n")
        pickle.dump(truth, open(d / "truth.pkl", "wb"))
    T = pd.DataFrame([{"scene": t["scene"], "stratum": t["stratum"]} for t in truth.values()])
    print(f"prepared {len(truth)} windows, {T.scene.nunique()} scenes; strata {T.stratum.value_counts().to_dict()}")
    ys = pd.DataFrame([{"stratum": t["stratum"], "a": a, "y": lab(t["vehicles"], acc_plan(t["plans"]["logged"], a))}
                       for t in truth.values() for a in ACCS])
    print("crash share by a (labelled candidates):")
    print(ys[ys.y >= 0].groupby(["stratum", "a"]).y.mean().unstack().round(2).to_string())


def ego_max(m, frame, plan, h):
    fx, fy, fyaw = frame
    cx, cy, cyaw = plan_pose(plan, h)
    ob = box_in_frame(fx, fy, fyaw, cx + EGO_C * math.cos(cyaw), cy + EGO_C * math.sin(cyaw), cyaw, EGO_L, EGO_W)
    return float(m[ob].max()) if ob.any() else 0.0


def physics(t, plan):
    worst = -99.0
    for h in (1, 2, 3):
        x, y, _ = plan_pose(plan, h)
        for d in t["dets"]:
            px, py = d["x"] + d["v"][0] * h, d["y"] + d["v"][1] * h
            if px > 0 and abs(py - y) <= (EGO_W + d["W"]) / 2:
                worst = max(worst, min(x + FRONT, px + d["L"] / 2) - max(x + REAR, px - d["L"] / 2))
    return worst


def table(out, accs, cvs):
    T = pickle.load(open(Path(out) / "acc" / "truth.pkl", "rb"))

    def load(d):
        return (np.load(Path(d) / "soft.npy", mmap_mode="r"),
                {t: i for i, t in enumerate(json.load(open(Path(d) / "soft_tokens.json")))})
    soft = {k: load(d) for k, d in accs.items()}
    cvs = {k: load(d) for k, d in cvs.items()}
    first = next(iter(accs))
    rows = []
    for tok, t in T.items():
        lg = t["plans"]["logged"]
        for j, a in enumerate(ACCS, start=1):
            plan = acc_plan(lg, a)
            y = lab(t["vehicles"], plan)
            if y < 0:
                continue
            r = {"token": tok, "scene": t["scene"], "stratum": t["stratum"], "a": a, "y": y,
                 "physics": physics(t, plan), "h_gap0": -t["gap0"], "h_v_ego": lg[0][0] / 0.5}
            for k, (S, ix) in soft.items():
                if tok not in ix:
                    continue
                i = ix[tok]
                r[f"self_{k}"] = max(ego_max(np.asarray(S[i, j, h - 1], np.float32), plan_pose(plan, h), plan, h)
                                     for h in (1, 2, 3))
                if k == first:
                    r["oracle"] = max(ego_max(np.asarray(S[i, 0, h - 1], np.float32), plan_pose(lg, h), plan, h)
                                      for h in (1, 2, 3))
            for k, (S, ix) in cvs.items():
                if tok in ix:
                    i = ix[tok]
                    r[k] = max(ego_max(np.asarray(S[i, 0, h - 1], np.float32), plan_pose(t["cv"], h), plan, h)
                               for h in (1, 2, 3))
            rows.append(r)
    return pd.DataFrame(rows)


GRID = (2.0, 2.5, 2.75, 3.0, 3.25)


def gain(df, j):
    from planner_select import at_progress, curve
    cj, c0 = curve(df, j), curve(df, "const_a")
    return float(np.nanmean([at_progress(c0, p) - at_progress(cj, p) for p in GRID]))


def score(out, specs):
    from planner_select import at_progress, curve
    accs, cvs = {}, {}
    for s in specs:
        k, d = s.split("=", 1)
        if k.startswith("cv"):
            cvs["cvref" + k[2:]] = d                 # cv=<dir> -> cvref, cv_ctrl=<dir> -> cvref_ctrl
        else:
            accs[k] = d
    df = table(out, accs, cvs)
    judges = [c for c in df.columns if c.startswith("self_")] + ["oracle"] + list(cvs) + ["physics", "h_gap0", "h_v_ego"]
    df = df.dropna(subset=judges)
    df["const_a"] = df.a
    df.to_parquet(Path(out) / "table.parquet")
    sc = df.scene.unique()
    print(f"{df.token.nunique()} windows, {len(sc)} scenes, {len(df)} candidates ({int(df.y.sum())} crash); "
          f"strata {df.groupby('stratum').token.nunique().to_dict()}")
    truth = df[df.y == 0].groupby("token").a.max().reindex(df.token.unique()).fillna(0.0)
    print(f"truth planner mean a {truth.mean():.2f}; crash share by a: " +
          " ".join(f"{a:.2f}:{m:.2f}" for a, m in df.groupby("a").y.mean().items()))
    g = {s: df[df.scene == s] for s in sc}
    rng = np.random.default_rng(0)
    res = []
    for _ in range(300):
        parts = []
        for k, s in enumerate(rng.choice(sc, len(sc))):
            p = g[s].copy()
            p["token"] = p.token + f"#{k}"
            parts.append(p)
        res.append(pd.concat(parts))
    print("  judge                speed-blind AUROC [95% CI] | crash at mean a >= " + " / ".join(map(str, GRID)) +
          " | crash reduction vs plan-blind [95% CI]")
    G = {}
    for j in ["const_a"] + judges:
        c = curve(df, j)
        sb = [strat(r, j) for r in res]
        line = f"  {j:20s} {strat(df, j):.3f} [{np.nanquantile(sb, 0.025):.3f}, {np.nanquantile(sb, 0.975):.3f}] | " + \
            " ".join(f"{at_progress(c, p):.3f}" for p in GRID)
        if j != "const_a":
            gr = [gain(r, j) for r in res]
            G[j] = np.array(gr)
            line += f" | {gain(df, j):+.3f} [{np.nanquantile(gr, 0.025):+.3f}, {np.nanquantile(gr, 0.975):+.3f}]"
        print(line, flush=True)
    selfs = [j for j in judges if j.startswith("self_")]
    pairs = [(c, s_) for c in judges if c.startswith("cvref") for s_ in selfs[:1]] + \
        [("physics", c) for c in judges if c.startswith("cvref")] + [(s_, selfs[0]) for s_ in selfs[1:]]
    for a_, b_ in pairs:
        d = G[a_] - G[b_]
        print(f"  paired gain {a_} - {b_}: {gain(df, a_) - gain(df, b_):+.3f} [{np.nanquantile(d, 0.025):+.3f}, "
              f"{np.nanquantile(d, 0.975):+.3f}]")
    from scipy.stats import spearmanr
    print("  within-window plan sensitivity, median Spearman(score, a) over windows with >= 3 labelled plans: " +
          " | ".join(f"{j} {np.nanmedian([spearmanr(g.a, g[j])[0] for _, g in df.groupby('token') if len(g) >= 3 and g[j].nunique() > 1]):+.2f}"
                     for j in judges if j.startswith(("self_", "oracle", "cvref", "physics"))))
    for st, f in df.groupby("stratum"):
        print(f"  [{st}] {f.token.nunique()} windows: speed-blind " +
              " | ".join(f"{j} {strat(f, j):.3f}" for j in judges))


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        score(sys.argv[2], sys.argv[3:])
