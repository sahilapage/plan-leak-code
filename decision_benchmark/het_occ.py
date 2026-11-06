"""A29 decision benchmark (src/sw_hetero.py windows, truth, labels, physics) for the GT-input occupancy models:
DOME (plan-conditioned) and COME (plan-free scene-centric forecaster), perfect Occ3D history as input.
  windows <het_dir> <fold>   <het_dir>/occ/windows.pkl (scene, idx, token) for src/gt_judges.py run_come
  dome    <het_dir> <fold>   DOME under logged + every ACCS plan + the past-only CV plan (src/dome_roc.py poses):
                             occ/plan/<tok>_dome.npz veh (2 + len(ACCS), 4, 200, 200) [logged, ACCS..., cv]
  come    <het_dir> <fold>   src/gt_judges.py run_come -> occ/plan/<tok>_come.npz (t0 frame)
  score   <het_dir> <fold> [<het_dir> <fold>]   judges dome_self (each plan in its own forecast), dome_oracle (logged
          forecast), dome_cvref (CV forecast), come (plan-free; candidate box in the t0 frame), physics, t0 gap;
          gain vs plan-blind, progress at <= 2 % crashes, paired scene bootstrap; writes occ/table.parquet per dir.
Usage (server, vla env): python src/het_occ.py windows|dome|come <het_dir> <fold> | score <het_dir> <fold> ..."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
from fixed_acc import ACCS, acc_plan  # noqa: E402
from sw_hetero import lab, physics  # noqa: E402


def make_windows(het, fold):
    import dome_infer as di
    T = pickle.load(open(Path(het) / "acc" / "truth.pkl", "rb"))
    infos, _ = di.windows({fold})
    W = {}
    for tok, t in T.items():
        fr = infos[t["scene"]]
        i = next(k for k, f in enumerate(fr) if f["token"] == tok)
        if i - 3 < 0 or i + 7 > len(fr):
            continue
        W[tok] = {"scene": t["scene"], "idx": i - 3, "token": tok, "a": 0.0}
    d = Path(het) / "occ"
    (d / "plan").mkdir(parents=True, exist_ok=True)
    pickle.dump(W, open(d / "windows.pkl", "wb"))
    print(f"{len(W)} / {len(T)} windows with 3 past + 6 future keyframes")


def run_dome(het, fold, bs=2):
    import dome_infer as di
    import probe_insert as pi
    from dome_roc import poses
    from gt_judges import acc_poses
    d = Path(het) / "occ"
    W = pickle.load(open(d / "windows.pkl", "rb"))
    infos, _ = di.windows({fold})
    net = pi.Dome()
    toks = [t for t in W if not (d / "plan" / f"{t}_dome.npz").exists()]
    for n, t in enumerate(toks):
        s = W[t]
        fr = infos[s["scene"]][s["idx"]: s["idx"] + 10]
        lp, _, cv = poses(infos, s)
        occ = np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]])
        plans = [lp] + [acc_poses(lp, a) for a in ACCS] + [cv]
        out = []
        for b0 in range(0, len(plans), bs):
            F = net([occ] * len(plans[b0: b0 + bs]), plans[b0: b0 + bs], seed=0)
            out += [F[i][:4] for i in range(len(plans[b0: b0 + bs]))]
        np.savez_compressed(d / "plan" / f"{t}_dome.npz", veh=np.stack(out).astype(np.float16))
        print(f"DOME {n + 1}/{len(toks)}", flush=True)


def table(het, fold):
    import dome_infer as di
    from dome_roc import poses
    from gt_judges import acc_poses, box_max
    d = Path(het) / "occ"
    W = pickle.load(open(d / "windows.pkl", "rb"))
    T = pickle.load(open(Path(het) / "acc" / "truth.pkl", "rb"))
    infos, _ = di.windows({fold})
    rows = []
    for tok, s in W.items():
        fd, fc = d / "plan" / f"{tok}_dome.npz", d / "plan" / f"{tok}_come.npz"
        if not (fd.exists() and fc.exists()):
            continue
        V = np.load(fd)["veh"].astype(np.float32)
        Vc = np.load(fc)["veh"].astype(np.float32)
        lp, _, cv = poses(infos, s)
        P0 = lp[3]
        t = T[tok]
        lg = t["plans"]["logged"]
        for j, a in enumerate(ACCS, start=1):
            plan = acc_plan(lg, a)
            y = lab(t["vehicles"], plan)
            if y < 0:
                continue
            Pc = acc_poses(lp, a)
            r = {"token": tok, "scene": s["scene"], "stratum": t["stratum"], "a": a, "y": y,
                 "physics": physics(t, plan), "h_gap0": -t["gap0"]}
            r["dome_self"] = max(box_max(V[j, h], Pc[3 + 2 * h], Pc[3 + 2 * h]) for h in (1, 2, 3))
            r["dome_oracle"] = max(box_max(V[0, h], lp[3 + 2 * h], Pc[3 + 2 * h]) for h in (1, 2, 3))
            r["dome_cvref"] = max(box_max(V[-1, h], cv[3 + 2 * h], Pc[3 + 2 * h]) for h in (1, 2, 3))
            r["come"] = max(box_max(Vc[h], P0, Pc[3 + 2 * h]) for h in (1, 2, 3))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(d / "table.parquet")
    return df


def score(args):
    from planner_select import at_crash, curve
    from sw_hetero import gain
    df = pd.concat([table(args[i], int(args[i + 1])) for i in range(0, len(args), 2)], ignore_index=True)
    df["const_a"] = df.a
    js = ["dome_self", "dome_cvref", "dome_oracle", "come", "physics", "h_gap0"]
    sc = df.scene.unique()
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
    G = {j: np.array([gain(r, j) for r in res]) for j in js}
    P = {j: np.array([at_crash(curve(r, j), 0.02) for r in res]) for j in js + ["const_a"]}
    print(f"{df.token.nunique()} windows, {len(sc)} scenes, {len(df)} candidates ({int(df.y.sum())} crash)")
    print(f"  plan-blind progress@<=2% {at_crash(curve(df, 'const_a'), 0.02):.2f}")
    for j in js:
        print(f"  {j:12s} gain {gain(df, j):+.3f} [{np.quantile(G[j], .025):+.3f}, {np.quantile(G[j], .975):+.3f}] | "
              f"progress@<=2% {at_crash(curve(df, j), 0.02):.2f} [{np.quantile(P[j], .025):.2f}, "
              f"{np.quantile(P[j], .975):.2f}]", flush=True)
    for a, b in (("dome_cvref", "dome_self"), ("dome_oracle", "dome_self"), ("come", "dome_self"),
                 ("come", "dome_cvref"), ("physics", "come"), ("physics", "dome_cvref")):
        dd = G[a] - G[b]
        print(f"  paired gain {a} - {b}: {gain(df, a) - gain(df, b):+.3f} [{np.quantile(dd, .025):+.3f}, "
              f"{np.quantile(dd, .975):+.3f}]")
    dd = P["dome_self"] - P["const_a"]
    print(f"  progress@<=2% dome_self - plan-blind: {np.median(dd):+.2f} [{np.quantile(dd, .025):+.2f}, "
          f"{np.quantile(dd, .975):+.2f}]")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "windows":
        make_windows(sys.argv[2], int(sys.argv[3]))
    elif st == "dome":
        run_dome(sys.argv[2], int(sys.argv[3]))
    elif st == "come":
        import gt_judges
        gt_judges.run_come(str(Path(sys.argv[2]) / "occ"), int(sys.argv[3]))
    else:
        score(sys.argv[2:])
