"""A39: closed-loop planning on REAL nuScenes logs (non-reactive log replay) with the A31 object world models.
A38 showed in the A35 simulator: the whole-plan model brakes hard for no reason (phantom braking), test-time fixes of
it remove that but under-react to the lead, and only causal conditioning is both smooth and safe. A39 asks the same on
real traffic, where per-step lookahead (step1, the common "frame t+1 sees the ego pose at t+1" design) also leaks (A31
beta 0.74). World: every nuScenes val scene; the other vehicles replay their annotated tracks (2 Hz, linear in
between; they never react to the ego, as nuPlan's non-reactive closed loop); the ego follows its logged path
laterally and the planner controls only its progress along it.
Planner (A38's, unchanged): every 0.5 s (one keyframe), constant-acceleration plans CAND over 3 s from the current
speed, never reversing; a plan is kept if for every vehicle ahead (centre x > 0 in the ego frame) that laterally
overlaps the planned path at a keyframe, the predicted penetration of the ego front beyond its rear + HEADWAY x plan
speed stays <= -MARGIN (src/obj_rl.py depth geometry); kept plan closest to clip(0.5 (v_des - v), -2, 2), v_des = the
scene's maximum logged ego speed (an assertive driver that presses against slower leads); none kept -> the plan with
the smallest front value. Rear-end checks are off (logged followers cannot react). Its acceleration is applied for
0.5 s at 10 Hz.
Model input at each decision: ego = its closed-loop history (3 keyframes) + the candidate plan along the path, agents =
the 24 nearest logged vehicles (|x| <= 60 m, |y| <= 40 m) with their logged history, all in the current ego frame
(exactly the A31 window format). Judges: A31 checkpoints (chunk / step0 / step1 / marg), gc_ / pm_ wrappers
(src/gcomp.py), cv (constant velocity), oracle (the logged future = the true do() future for non-reactive agents).
Metrics per episode (10 Hz, until a front crash or the end): front crash (ego front past the rear of a vehicle ahead
whose box laterally overlaps the ego's by >= 0.3 m), TTC < 1.5 s steps (to the nearest such vehicle ahead), hard
brakes (chosen a <= -3), phantom hard brakes (the oracle, same state, keeps a plan with a >= -2), no-safe-plan
decisions, mean speed, minimum gap; and the same restricted to decisions with a vehicle ahead in path within 30 m.
Stages:  build <out.pkl>                         per-scene global ego poses + vehicle tracks (nuScenes v1.0 val)
         run <eps.pkl> <out.csv> <judge> ...      closed loop (A39_HEADWAY, default 1.0)
Usage (4090, vla env): python src/a39_real_closed.py build $O/a39_val.pkl ; ... run $O/a39_val.pkl out.csv $O/chunk.pt ..."""
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))

CAND = (-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
MARGIN, HEADWAY = 1.0, float(os.environ.get("A39_HEADWAY", 1.0))
NA, P, SUB = 24, 3, 5                              # agents per window, history keyframes, physics sub-steps per 0.5 s
EGO_L, EGO_W, EGO_C = 4.084, 1.73, 1.3             # src/sw_realhazard.py footprint (reference point = rear axle)
FRONT = EGO_C + EGO_L / 2
LAT_SLACK = 0.3                                    # required lateral box overlap for a crash / TTC target (m)


# ------------------------------------------------------------------------------------------------------------- data
def build(out):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from pyquaternion import Quaternion
    import realhazard as rh
    nusc = NuScenes("v1.0-trainval", dataroot=str(rh.DATA / "nuscenes"), verbose=False)
    names = set(splits.val)
    eps = []
    for sc in nusc.scene:
        if sc["name"] not in names:
            continue
        seq = [nusc.get("sample", sc["first_sample_token"])]
        while seq[-1]["next"]:
            seq.append(nusc.get("sample", seq[-1]["next"]))
        ego, ags = [], {}
        for k, s in enumerate(seq):
            sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            ego.append((ep["translation"][0], ep["translation"][1], Quaternion(ep["rotation"]).yaw_pitch_roll[0]))
            for at in s["anns"]:
                a = nusc.get("sample_annotation", at)
                if a["category_name"].startswith(rh.VEH_CATS):
                    d = ags.setdefault(a["instance_token"], {"t": {}, "size": (a["size"][1], a["size"][0])})
                    d["t"][k] = (a["translation"][0], a["translation"][1], Quaternion(a["rotation"]).yaw_pitch_roll[0])
        K, n = len(seq), len(ags)
        X, M, S = np.zeros((n, K, 3)), np.zeros((n, K), bool), np.zeros((n, 2))
        for j, d in enumerate(ags.values()):
            S[j] = d["size"]
            for k, v in d["t"].items():
                X[j, k], M[j, k] = v, True
        eps.append({"scene": sc["name"], "ego": np.array(ego), "X": X, "M": M, "S": S})
    pickle.dump(eps, open(out, "wb"))
    print(f"{len(eps)} scenes -> {out}")


class Route:
    """the logged ego path as arclength -> (x, y, yaw), extended 300 m straight beyond its end."""

    def __init__(self, ego):
        p = ego[:, :2]
        yaw = np.unwrap(ego[:, 2])
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        keep = np.r_[True, np.diff(s) > 1e-3]                       # drop standstill duplicates (interp needs increasing s)
        p, yaw, s = p[keep], yaw[keep], s[keep]
        ext = np.arange(1.0, 301.0)
        self.s = np.r_[s, s[-1] + ext]
        self.x = np.r_[p[:, 0], p[-1, 0] + ext * math.cos(yaw[-1])]
        self.y = np.r_[p[:, 1], p[-1, 1] + ext * math.sin(yaw[-1])]
        self.yaw = np.r_[yaw, np.full(len(ext), yaw[-1])]
        self.s_log = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ego[:, :2], axis=0).T))]   # logged arclength per keyframe

    def at(self, s):
        return np.interp(s, self.s, self.x), np.interp(s, self.s, self.y), np.interp(s, self.s, self.yaw)


def to_frame(x, y, yaw, fx, fy, fyaw):
    c, s = math.cos(fyaw), math.sin(fyaw)
    dx, dy = x - fx, y - fy
    return c * dx + s * dy, -s * dx + c * dy, (yaw - fyaw + np.pi) % (2 * np.pi) - np.pi


def agents_at(ep, t):
    """logged vehicle poses at (fractional) keyframe time t, linear between keyframes (N, 3), valid (N,)."""
    X, M = ep["X"], ep["M"]
    k0 = int(math.floor(t))
    k1 = min(k0 + 1, X.shape[1] - 1)
    w = t - k0
    if w < 1e-6:
        return X[:, k0], M[:, k0]
    ok = M[:, k0] & M[:, k1]
    d = (X[:, k1, 2] - X[:, k0, 2] + np.pi) % (2 * np.pi) - np.pi
    p = np.stack([X[:, k0, 0] * (1 - w) + X[:, k1, 0] * w, X[:, k0, 1] * (1 - w) + X[:, k1, 1] * w,
                  X[:, k0, 2] + w * d], 1)
    return p, ok


def in_path(ep, t, ex, ey, eyaw, excl):
    """bumper gap (m) and longitudinal speed (m/s, from +-0.1 keyframe) of every logged vehicle ahead whose box
    laterally overlaps the ego's by >= LAT_SLACK, in the ego frame at time t; others -> inf."""
    p, ok = agents_at(ep, t)
    lx, ly, _ = to_frame(p[:, 0], p[:, 1], p[:, 2], ex, ey, eyaw)
    L, W = ep["S"][:, 0], ep["S"][:, 1]
    tgt = ok & ~excl & (lx > EGO_C) & (np.abs(ly) <= (EGO_W + W) / 2 - LAT_SLACK)
    gap = np.where(tgt, lx - L / 2 - FRONT, np.inf)
    t2 = min(t + 0.2, ep["X"].shape[1] - 1.0)
    p2, ok2 = agents_at(ep, t2)
    vx = np.where(ok2, ((p2[:, 0] - p[:, 0]) * math.cos(eyaw) + (p2[:, 1] - p[:, 1]) * math.sin(eyaw)) / ((t2 - t) * 0.5 + 1e-9), 0.0)
    return gap, vx


# ------------------------------------------------------------------------------------------------------- closed loop
def run(path, judge, dev="cuda"):
    import torch
    import gcomp  # noqa: F401  (gc_ / pm_ names)
    import obj_wm as o
    eps = pickle.load(open(path, "rb"))[:int(os.environ.get("A39_MAXEP", 10 ** 6))]
    from a38_closed import judge_name, load_judge, predict          # mix<k>:<A>:<B> leak-dose judges (A41)
    net = None if judge in ("cv", "oracle") else load_judge(judge, dev)
    cand = np.array(CAND)
    nc = len(cand)
    rows = []
    for ei, ep in enumerate(eps):
        R = Route(ep["ego"])
        K = ep["X"].shape[1]
        if K < P + 8:
            continue
        v_des = float(np.max(np.diff(R.s_log) / 0.5))
        k = P
        s = R.s_log[k]
        v = (R.s_log[k] - R.s_log[k - 1]) / 0.5
        hist = [tuple(ep["ego"][i]) for i in range(k - P, k)]       # closed-loop ego poses at k-3 .. k-1 (global)
        ex, ey, eyaw = R.at(s)
        g0, _ = in_path(ep, float(k), ex, ey, eyaw, np.zeros(len(ep["S"]), bool))
        excl = g0 < 0.5                                             # already overlapping at the start: annotation artifacts
        st = dict(front=0, t_crash=-1.0, vsum=0.0, n=0, hard=0, phantom=0, nosafe=0, ttc=0, gmin=1e3,
                  dec=0, dec_lead=0, hard_lead=0, phantom_lead=0)
        while k <= K - 7:
            ex, ey, eyaw = R.at(s)
            # --- window in the current ego frame (A31 format)
            E = np.zeros((10, 3), np.float32)
            for i, (hx, hy, hyaw) in enumerate(hist):
                E[i] = to_frame(hx, hy, hyaw, ex, ey, eyaw)
            X, Mk = ep["X"], ep["M"]
            cx, cy, _ = to_frame(X[:, k, 0], X[:, k, 1], X[:, k, 2], ex, ey, eyaw)
            near = np.where(Mk[:, k] & (np.abs(cx) <= 60) & (np.abs(cy) <= 40))[0]
            near = near[np.argsort(np.hypot(cx[near], cy[near]))][:NA]
            A = np.zeros((NA, 10, 3), np.float32)
            Mw = np.zeros((NA, 10), bool)
            Sw = np.zeros((NA, 2), np.float32)
            for q, j in enumerate(near):
                Sw[q] = ep["S"][j]
                for i, kk in enumerate(range(k - P, k + 1)):
                    if Mk[j, kk]:
                        A[q, i] = to_frame(X[j, kk, 0], X[j, kk, 1], X[j, kk, 2], ex, ey, eyaw)
                        Mw[q, i] = True
                Mw[q, P + 1:] = Mw[q, P]
            # --- candidate plans along the route
            tt = 0.5 * np.arange(1, 7)
            WP, VP = np.zeros((nc, 6, 2), np.float32), np.zeros((nc, 6))
            for c, a in enumerate(cand):
                d = v * tt + 0.5 * a * tt * tt
                d = np.maximum.accumulate(np.maximum(d, 0)) if a < 0 else d
                px, py, _ = R.at(s + d)
                fx, fy, _ = to_frame(px, py, 0.0, ex, ey, eyaw)
                WP[c] = np.stack([fx, fy], 1)
                VP[c] = np.maximum(v + a * tt, 0)
            # --- predicted agent positions (nc, NA, 6, 2)
            if judge == "oracle":
                Y = np.zeros((NA, 6, 2), np.float32)
                for q, j in enumerate(near):
                    for i in range(6):
                        kk = k + 1 + i
                        if Mk[j, kk]:
                            Y[q, i] = to_frame(X[j, kk, 0], X[j, kk, 1], X[j, kk, 2], ex, ey, eyaw)[:2]
                        else:
                            Y[q, i] = Y[q, i - 1] if i else A[q, P, :2]
                Y = np.repeat(Y[None], nc, 0)
            else:
                Et = torch.tensor(np.repeat(E[None], nc, 0), device=dev)
                At, Mt, St = (torch.tensor(np.repeat(z[None], nc, 0), device=dev) for z in (A, Mw, Sw))
                if judge == "cv":
                    Y = o.cv_base(At, Mt, P, 6).cpu().numpy()
                else:
                    Et = o.plan_E(Et, torch.tensor(WP, device=dev))
                    with torch.no_grad():
                        Y = predict(net, Et, At, Mt, St).cpu().numpy()
            # oracle front values on the same state (for phantom brakes), and the judge's
            Yo = np.zeros((NA, 6, 2))
            for q, j in enumerate(near):
                for i in range(6):
                    kk = k + 1 + i
                    Yo[q, i] = to_frame(X[j, kk, 0], X[j, kk, 1], X[j, kk, 2], ex, ey, eyaw)[:2] if Mk[j, kk] else (
                        Yo[q, i - 1] if i else A[q, P, :2])
            ahead = Mw[:, P] & (A[:, P, 0] > 0) & (np.arange(NA) < len(near))

            def front_val(Yc):
                lat = np.abs(Yc[..., 1] - WP[:, None, :, 1]) <= (EGO_W + Sw[None, :, 1:2]) / 2
                ov = (WP[:, None, :, 0] + FRONT) - (Yc[..., 0] - Sw[None, :, 0:1] / 2) + HEADWAY * VP[:, None, :]
                ov = np.where(lat & ahead[None, :, None], ov, -99.0)
                return ov.reshape(nc, -1).max(1)
            fj = front_val(Y)
            fo = front_val(np.repeat(Yo[None], nc, 0))
            ok = fj <= -MARGIN
            a_des = np.clip(0.5 * (v_des - v), -2, 2)
            cost = np.abs(cand - a_des) + 1e-3 * cand
            pick = int(np.where(ok, cost, np.inf).argmin()) if ok.any() else int(fj.argmin())
            a = cand[pick]
            g_now, _ = in_path(ep, float(k), ex, ey, eyaw, excl)
            lead = bool(np.isfinite(g_now).any() and g_now.min() <= 30)
            gentle = ((fo <= -MARGIN) & (cand >= -2)).any()
            st["dec"] += 1
            st["dec_lead"] += lead
            st["nosafe"] += not ok.any()
            if a <= -3:
                st["hard"] += 1
                st["hard_lead"] += lead
                st["phantom"] += gentle
                st["phantom_lead"] += gentle and lead
            # --- apply for 0.5 s at 10 Hz
            for i in range(SUB):
                v = max(v + a * 0.1, 0.0)
                s += v * 0.1
                t = k + (i + 1) / SUB
                fx_, fy_, fyaw_ = R.at(s)
                gap, vx = in_path(ep, t, fx_, fy_, fyaw_, excl)
                st["vsum"] += v
                st["n"] += 1
                if np.isfinite(gap).any():
                    j = int(np.argmin(gap))
                    st["gmin"] = min(st["gmin"], float(gap[j]))
                    clo = v - vx[j]
                    st["ttc"] += int(clo > 0 and gap[j] < 1.5 * clo)
                    if gap[j] < 0:
                        st["front"], st["t_crash"] = 1, t * 0.5
                        break
            hist = (hist + [(ex, ey, eyaw)])[-P:]                     # the pose at this decision joins the history
            if st["front"]:
                break
            k += 1
        rows.append({"judge": judge_name(judge), "ep": ei, "scene": ep["scene"], "front": st["front"],
                     "t_crash": st["t_crash"], "speed": st["vsum"] / max(st["n"], 1), "hard_brakes": st["hard"],
                     "phantom": st["phantom"], "no_safe_plan": st["nosafe"], "ttc_viol": st["ttc"],
                     "min_gap": st["gmin"], "decisions": st["dec"], "dec_lead": st["dec_lead"],
                     "hard_lead": st["hard_lead"], "phantom_lead": st["phantom_lead"]})
    return pd.DataFrame(rows)


def logged(path):
    """the logging driver over the same span (reference): TTC steps and hard decelerations of the log itself."""
    eps = pickle.load(open(path, "rb"))[:int(os.environ.get("A39_MAXEP", 10 ** 6))]
    rows = []
    for ei, ep in enumerate(eps):
        R = Route(ep["ego"])
        K = ep["X"].shape[1]
        if K < P + 8:
            continue
        v_log = np.diff(R.s_log) / 0.5
        ex, ey, eyaw = ep["ego"][P]
        g0, _ = in_path(ep, float(P), ex, ey, eyaw, np.zeros(len(ep["S"]), bool))
        excl = g0 < 0.5
        ttc, gmin, front = 0, 1e3, 0
        for k in range(P, K - 7):
            for i in range(SUB):
                t = k + (i + 1) / SUB
                w = (i + 1) / SUB
                x = ep["ego"][k, 0] * (1 - w) + ep["ego"][k + 1, 0] * w
                y = ep["ego"][k, 1] * (1 - w) + ep["ego"][k + 1, 1] * w
                gap, vx = in_path(ep, t, x, y, ep["ego"][k, 2], excl)
                if np.isfinite(gap).any():
                    j = int(np.argmin(gap))
                    gmin = min(gmin, float(gap[j]))
                    clo = v_log[k] - vx[j]
                    ttc += int(clo > 0 and gap[j] < 1.5 * clo)
                    front |= int(gap[j] < 0)
        hard = int((np.diff(v_log[P - 1:K - 7]) / 0.5 <= -3).sum())
        rows.append({"judge": "logged_driver", "ep": ei, "scene": ep["scene"], "front": front, "t_crash": -1.0,
                     "speed": float(v_log[P:K - 7].mean()), "hard_brakes": hard, "phantom": 0, "no_safe_plan": 0,
                     "ttc_viol": ttc, "min_gap": gmin, "decisions": K - 7 - P, "dec_lead": 0, "hard_lead": 0,
                     "phantom_lead": 0})
    return pd.DataFrame(rows)


def summarize(df, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.sort(df.ep.unique())
    bs = rng.integers(0, len(eps), (B, len(eps)))
    print(f"{'judge':14s} {'front crash':>22s} {'speed':>6s} {'hard/ep':>7s} {'phantom':>7s} {'ph.share':>8s} "
          f"{'nosafe':>6s} {'TTC/ep':>7s} {'min gap p5':>10s}")
    for j, g in df.groupby("judge", sort=False):
        g = g.set_index("ep").reindex(eps)
        f = g.front.fillna(0).values.astype(float)
        lo, hi = np.percentile(f[bs].mean(1), [2.5, 97.5])
        print(f"{j:14s} {'%.3f [%.3f, %.3f]' % (f.mean(), lo, hi):>22s} {g.speed.mean():6.2f} {g.hard_brakes.mean():7.2f} "
              f"{g.phantom.mean():7.2f} {g.phantom.sum() / max(g.hard_brakes.sum(), 1):8.2f} {g.no_safe_plan.mean():6.2f} "
              f"{g.ttc_viol.mean():7.2f} {np.nanpercentile(g.min_gap, 5):10.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
        sys.exit()
    if sys.argv[1] == "summary":
        summarize(pd.concat([pd.read_csv(f) for f in sys.argv[2:]]))
        sys.exit()
    path, out, judges = sys.argv[2], sys.argv[3], sys.argv[4:]
    dev = os.environ.get("OBJ_DEV", "cuda")
    dfs = [logged(path)]
    for j in judges:
        dfs.append(run(path, j, dev))
        print(f"{j.split(':')[0] if j.startswith('mix') else Path(j).stem} done", flush=True)
        pd.concat(dfs).to_csv(out, index=False)
    summarize(pd.concat(dfs))
