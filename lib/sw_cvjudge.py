"""Training-free fix for plan-leaking world models: the PAST-ONLY reference judge. Every candidate is judged inside ONE
forecast, imagined under the constant-velocity plan from the past (v0 = ego speed over the last keyframe interval,
straight along the current heading). Unlike the logged plan (encodes the real future) or the brake / still plans (in
car-following logs a stopping ego implies a stopping lead), this plan carries no information about the future, so the
leak cannot move the real lead out of, or into, a candidate's way. Same windows as src/sw_realhazard.py.
Usage: python src/sw_cvjudge.py prep <swrh_dir> <out_dir>    (writes cands.jsonl / ids.txt / truth.pkl with the CV plan)
       (then src/swtc_verifier.py --samples <out>/cands.jsonl ... --k 1 --soft-dump <out>/soft --soft-all)
       python src/sw_cvjudge.py score <swrh_dir> <out_dir> [<swrh_dir> <out_dir> ...]"""
import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_insert as pi  # noqa: E402
from common import DATA  # noqa: E402
from sw_realhazard import EGO_C, EGO_L, EGO_W, PLANS, TAU, box_in_frame, plan_pose  # noqa: E402


def quat_mat(t, q):
    w, x, y, z = q
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    M[:3, 3] = t
    return M


def past_speed(tokens):
    infos = pickle.load(open(DATA / "swtc_infos_clean.pkl", "rb"))["infos"]
    by = {e["token"]: e for e in infos}
    out = {}
    for tok in tokens:
        e = by.get(tok)
        p = by.get((e or {}).get("prev") or "")
        if p is None:
            continue
        dt = (e["timestamp"] - p["timestamp"]) / 1e6
        rel = np.linalg.inv(quat_mat(e["ego2global_translation"], e["ego2global_rotation"])) @ \
            quat_mat(p["ego2global_translation"], p["ego2global_rotation"])
        if 0.2 < dt < 1.0:
            out[tok] = max(0.0, -float(rel[0, 3]) / dt)
    return out


def prep(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    truth = pickle.load(open(src / "truth.pkl", "rb"))
    v0 = past_speed(list(truth))
    keep = {}
    with open(out / "cands.jsonl", "w") as f:
        for tok, t in truth.items():
            if tok not in v0:
                continue
            cv = [[v0[tok] * 0.5 * (k + 1), 0.0] for k in range(6)]
            f.write(json.dumps({"id": tok, "cands": [cv]}) + "\n")
            keep[tok] = dict(t, v0=v0[tok], cv=cv)
    pickle.dump(keep, open(out / "truth.pkl", "wb"))
    ids = [ln for ln in open(src / "ids.txt").read().split("\n") if ln and ln.split()[0] in keep]
    open(out / "ids.txt", "w").write("\n".join(ids) + "\n")
    print(f"prepared {len(keep)} / {len(truth)} windows (v0 available); median v0 {np.median([k['v0'] for k in keep.values()]):.1f} m/s")


def score(pairs):
    rows = []
    for src, out in pairs:
        src, out = Path(src), Path(out)
        T = pickle.load(open(out / "truth.pkl", "rb"))
        cvs = np.load(out / "soft.npy", mmap_mode="r")
        own = np.load(src / "soft.npy", mmap_mode="r")
        oi = {t: i for i, t in enumerate(json.load(open(src / "soft_tokens.json")))}
        for i, tok in enumerate(json.load(open(out / "soft_tokens.json"))):
            t = T[tok]
            L, W = t["LW"]
            r = {"scene": t["scene"]}
            for judge in ("cv", "logged", "still", "self"):
                for cand in ("collide", "logged"):
                    v = 0.0
                    for h in (1, 2, 3):
                        if judge == "cv":
                            m, (fx, fy, fyaw) = np.asarray(cvs[i, 0, h - 1], np.float32), plan_pose(t["cv"], h)
                        else:
                            pj = cand if judge == "self" else judge
                            m = np.asarray(own[oi[tok], PLANS.index(pj), h - 1], np.float32)
                            fx, fy, fyaw = plan_pose(t["plans"][pj], h)
                        cx, cy, cyaw = plan_pose(t["plans"][cand], h)
                        ob = box_in_frame(fx, fy, fyaw, cx + EGO_C * math.cos(cyaw), cy + EGO_C * math.sin(cyaw), cyaw,
                                          EGO_L, EGO_W)
                        v = max(v, float(m[ob].max() >= TAU) if ob.any() else 0.0)
                    r[f"{judge}_{cand}"] = v
            lx, ly, lyaw = t["track"][4]
            fx, fy, fyaw = plan_pose(t["cv"], 2)
            r["kept_cv_2"] = float(pi.q50(np.asarray(cvs[i, 0, 1], np.float32),
                                          box_in_frame(fx, fy, fyaw, lx, ly, lyaw, L, W)) >= TAU)
            rows.append(r)
    df = pd.DataFrame(rows)

    def boot(col, B=10000):
        g = df.groupby("scene")[col].agg(["sum", "size"]).to_numpy(float)
        idx = np.random.default_rng(0).integers(0, len(g), (B, len(g)))
        q = np.quantile(g[idx, 0].sum(1) / g[idx, 1].sum(1), [0.025, 0.975])
        return f"{df[col].mean():.3f} [{q[0]:.3f}, {q[1]:.3f}]"

    print(f"{len(df)} windows, {df.scene.nunique()} scenes")
    for j in ("self", "still", "logged", "cv"):
        print(f"  judge {j:6s}: recall (collide) {boot(f'{j}_collide')} | false alarm (logged path) {boot(f'{j}_logged')}")
    print(f"  real lead kept +2 s in the CV forecast {boot('kept_cv_2')}")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3])
    else:
        a = sys.argv[2:]
        score([(a[i], a[i + 1]) for i in range(0, len(a), 2)])
