"""A38: closed-loop planning in the A35 world, where the true causal effect of every ego plan is known.
The ego is driven by a model-predictive planner instead of the logging driver: every 0.5 s it scores the candidate
constant-acceleration plans CAND_A (3 s, from the current speed, never reversing) with a judge, keeps the plans whose
predicted front gap (ego into lead) stays >= MARGIN + HEADWAY x the plan's speed and rear gap (follower into ego)
>= MARGIN at every 0.5 s keyframe, takes the
kept plan closest to its desired acceleration clip(0.5 (v0_e - v), -2, 2) (the logging driver's desired speed; ties ->
lower); if none is kept, front safety comes first (RSS-style: a rear-end is the follower's duty): the front-safe plan with
the smallest rear penetration, else the plan with the smallest front penetration; its acceleration is applied for 0.5 s. The world then evolves
exactly as in src/causal_sim.py: lead / adjacent / parked replay the log (exogenous: they never look at the ego), the
follower reacts to the planner's ego (follower_acc on the closed-loop state). Judges:
  <ckpt>      an A35 object WM (chunk / step0 / step1 / marg), or gc_<ckpt> / pm_<ckpt> (src/gcomp.py wrappers)
  cv          constant velocity of every agent from its last two keyframes
  oracle      the true future under do(plan): logged lead / adjacent, follower re-simulated against the plan
Crash = centre gap < LEN to the lead (front) or to the follower (rear), as in causal_sim.make; an episode stops counting
at its first crash. Per episode (until crash or end): front / rear crash, time of crash, mean speed, hard brakes
(chosen accel <= -3), steps with no kept plan, tailgating (share of 10 Hz steps with time gap to the lead < 0.5 s),
TTC violations (steps with time-to-collision to the lead < 1.5 s), minimum bumper gap to the lead.
Usage (any host; OBJ_DEV=cpu for CPU): python src/a38_closed.py <test_prefix> <out.csv> <judge> ..."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_sim as C  # noqa: E402

CAND_A = (-8.0, -6.0) + C.CAND_A                # A35 candidates + emergency braking (the logging driver brakes to -8)
MARGIN, HEADWAY, T_START, NA = 1.0, float(os.environ.get("A38_HEADWAY", 1.0)), 20, 24
KFT = np.arange(1, 7) * C.KF                       # keyframe offsets (10 Hz steps) of +0.5 .. +3 s


def ego_plan(v, a, n_steps):
    """distance travelled at 10 Hz steps 1 .. n_steps under constant accel a from speed v (B,), never reversing."""
    t = C.DT * np.arange(1, n_steps + 1)
    d = v[:, None] * t + 0.5 * a * t * t
    return np.maximum.accumulate(np.maximum(d, 0), 1) if a < 0 else d


def window(X, t):
    """obj_wm window at step t (history keyframes t-15 .. t; future slots empty): E (B, 10, 3), A (B, 24, 10, 3),
    M, S, plus the agent index of each slot (B, 24; -1 empty) and relation."""
    B = X.shape[0]
    ts = t + C.KF * np.arange(-3, 1)
    x0 = X[:, 0, t]
    E = np.zeros((B, 10, 3), np.float32)
    E[:, :4, 0] = X[:, 0, ts] - x0[:, None]
    A = np.zeros((B, NA, 10, 3), np.float32)
    M = np.zeros((B, NA, 10), bool)
    S = np.zeros((B, NA, 2), np.float32)
    J = -np.ones((B, NA), int)
    for e in range(B):
        near = [j for j in range(1, C.NAG) if abs(X[e, j, t] - x0[e]) <= 60]
        for q, j in enumerate(near):
            A[e, q, :4, 0] = X[e, j, ts] - x0[e]
            A[e, q, :, 1] = C.LANE_Y[j]
            M[e, q] = True
            S[e, q] = (C.LEN, C.WID)
            J[e, q] = j
    return E, A, M, S, J


def oracle(X, V, Acc, p, nz, t, d10):
    """true agent x (relative to the ego at t) at the 6 keyframes under do(ego = x0 + d10) (B, 13, 6)."""
    X, V, Acc = X.copy(), V.copy(), Acc.copy()
    x0 = X[:, 0, t]
    for i in range(30):
        u = t + i
        X[:, 0, u + 1] = x0 + d10[:, i]
        V[:, 0, u + 1] = (X[:, 0, u + 1] - X[:, 0, u]) / C.DT
        Acc[:, 0, u] = (V[:, 0, u + 1] - V[:, 0, u]) / C.DT
        a_f = C.follower_acc(p, nz, X, V, Acc, u)
        V[:, 2, u + 1] = np.maximum(V[:, 2, u] + a_f * C.DT, 0)
        X[:, 2, u + 1] = X[:, 2, u] + V[:, 2, u + 1] * C.DT
    return X[:, :, t + KFT] - x0[:, None, None]


class Mix:
    """A38 regime test: predictions of model B + kappa x (A - B); its lead leak is kappa x beta_A + (1 - kappa) x beta_B
    (beta is linear in the predictions). Judge name: mix<kappa>:<ckpt A>:<ckpt B>."""

    def __init__(self, judge, dev):
        import obj_wm as o
        k, a, b = judge[3:].split(":")
        self.k, self.a, self.b = float(k), o.load_net(a, dev), o.load_net(b, dev)


def load_judge(judge, dev):
    import gcomp  # noqa: F401  (patches obj_wm for gc_ / pm_ names)
    import obj_wm as o
    return Mix(judge, dev) if judge.startswith("mix") else o.load_net(judge, dev)


def predict(net, *args):
    import obj_wm as o
    if isinstance(net, Mix):
        yb = o.rollout(net.b, *args)
        return yb + net.k * (o.rollout(net.a, *args) - yb)
    return o.rollout(net, *args)


def judge_name(judge):
    return judge.split(":")[0] if judge.startswith("mix") else Path(judge).stem


def run(prefix, judge, dev="cpu"):
    import torch
    torch.set_num_threads(int(os.environ.get("A38_THREADS", 16)))
    H = dict(np.load(f"{prefix}_hist.npz"))
    p = {k[2:]: H[k] for k in H if k.startswith("p_")}
    nz = {k[2:]: H[k] for k in H if k.startswith("n_")}
    X, V, Acc = H["X"].copy(), H["V"].copy(), H["Acc"].copy()
    B, n = X.shape[0], X.shape[2]
    net = None
    if judge not in ("cv", "oracle"):
        net = load_judge(judge, dev)
    a_cmd = np.zeros(B)
    alive = np.ones(B, bool)
    crash, t_crash = np.zeros(B, int), np.full(B, -1)
    vsum, vcnt, hard, n_none = np.zeros(B), np.zeros(B), np.zeros(B, int), np.zeros(B, int)
    tail, ttcv, gmin = np.zeros(B), np.zeros(B, int), np.full(B, 1e3)
    nc = len(CAND_A)
    a_hist = np.zeros((B, n))
    for t in range(T_START, n - 31):                         # the oracle needs 3 s of log ahead
        if (t - T_START) % C.KF == 0:
            v = V[:, 0, t]
            d10 = {a: ego_plan(v, a, 30) for a in CAND_A}
            xe = np.stack([d10[a][:, KFT - 1] for a in CAND_A], 1)                      # (B, nc, 6)
            if judge == "oracle":
                Y = np.stack([oracle(X, V, Acc, p, nz, t, d10[a]) for a in CAND_A], 1)   # (B, nc, 13, 6)
                lead, fol = Y[:, :, 1], Y[:, :, 2]
            else:
                E, A, M, S, J = window(X, t)
                if judge == "cv":
                    v_ag = (A[:, :, 3, 0] - A[:, :, 2, 0]) * M[:, :, 2]
                    Yq = np.repeat((A[:, :, 3, 0, None] + v_ag[..., None] * np.arange(1, 7))[:, None], nc, 1)
                else:
                    Ec = np.repeat(E[:, None], nc, 1)
                    Ec[:, :, 4:, 0] = xe
                    tt = lambda z: torch.tensor(z.reshape((-1,) + z.shape[2:]), device=dev)  # noqa: E731
                    rep = lambda z: np.repeat(z[:, None], nc, 1)                               # noqa: E731
                    with torch.no_grad():
                        Yq = predict(net, tt(Ec), tt(rep(A)), tt(rep(M)), tt(rep(S)))[..., 0].cpu().numpy()
                    Yq = Yq.reshape(B, nc, NA, 6)
                big = np.full((B, nc, 6), 1e3)
                lead, fol = big.copy(), -big.copy()
                for e in range(B):
                    for q in np.where(J[e] == 1)[0]:
                        lead[e] = Yq[e, :, q]
                    for q in np.where(J[e] == 2)[0]:
                        fol[e] = Yq[e, :, q]
            ve = np.stack([np.maximum(v[:, None] + a * 0.5 * np.arange(1, 7), 0) for a in CAND_A], 1)
            front = ((xe + C.EGO_LEN / 2) - (lead - C.LEN / 2) + HEADWAY * ve).max(-1)    # (B, nc)
            rear = ((fol + C.LEN / 2) - (xe - C.EGO_LEN / 2)).max(-1)
            f_ok = front <= -MARGIN
            ok = f_ok & (rear <= -MARGIN)
            a_des = np.clip(0.5 * (p["v0_e"] - v), -2, 2)
            cand = np.array(CAND_A)
            cost = np.abs(cand[None] - a_des[:, None]) + 1e-3 * cand[None]               # ties -> lower accel
            pick = np.where(ok.any(1), np.where(ok, cost, np.inf).argmin(1),
                            np.where(f_ok.any(1), np.where(f_ok, rear, np.inf).argmin(1), front.argmin(1)))
            n_none += alive & ~ok.any(1)
            a_cmd = cand[pick]
            hard += alive & (a_cmd <= -3)
        a_hist[:, t] = a_cmd
        V[:, 0, t + 1] = np.maximum(V[:, 0, t] + a_cmd * C.DT, 0)
        X[:, 0, t + 1] = X[:, 0, t] + V[:, 0, t + 1] * C.DT
        Acc[:, 0, t] = (V[:, 0, t + 1] - V[:, 0, t]) / C.DT
        a_f = C.follower_acc(p, nz, X, V, Acc, t)
        Acc[:, 2, t] = a_f
        V[:, 2, t + 1] = np.maximum(V[:, 2, t] + a_f * C.DT, 0)
        X[:, 2, t + 1] = X[:, 2, t] + V[:, 2, t + 1] * C.DT
        vsum += alive * V[:, 0, t + 1]
        vcnt += alive
        gap = X[:, 1, t + 1] - X[:, 0, t + 1] - C.LEN
        clo = V[:, 0, t + 1] - V[:, 1, t + 1]
        tail += alive & (gap < 0.5 * V[:, 0, t + 1])
        ttcv += alive & (clo > 0) & (gap < 1.5 * np.maximum(clo, 1e-6))
        gmin = np.where(alive, np.minimum(gmin, gap), gmin)
        fr = alive & (X[:, 1, t + 1] - X[:, 0, t + 1] < C.LEN)
        rr = alive & (X[:, 0, t + 1] - X[:, 2, t + 1] < C.LEN)
        crash[fr] = 1
        crash[rr & ~fr] = 2
        t_crash[fr | rr] = t + 1
        alive &= ~(fr | rr)
    if os.environ.get("A38_TRACE"):                                      # debugging / figures
        np.savez_compressed(f"{os.environ['A38_TRACE']}_{Path(judge).stem}.npz", X=X, V=V, a=a_hist)
    return pd.DataFrame({"judge": judge_name(judge), "ep": np.arange(B), "front": crash == 1, "rear": crash == 2,
                         "t_crash": t_crash * C.DT, "speed": vsum / np.maximum(vcnt, 1), "hard_brakes": hard,
                         "no_safe_plan": n_none, "tailgate": tail / np.maximum(vcnt, 1), "ttc_viol": ttcv,
                         "min_gap": gmin})


def logged(prefix):
    """the logging driver's own record over the same span (reference)."""
    H = np.load(f"{prefix}_hist.npz")
    X, V = H["X"], H["V"]
    gap = X[:, 1, T_START:-30] - X[:, 0, T_START:-30] - C.LEN
    clo = V[:, 0, T_START:-30] - V[:, 1, T_START:-30]
    fr = (gap < 0).any(1)
    rr = (X[:, 0, T_START:-30] - X[:, 2, T_START:-30] < C.LEN).any(1)
    return pd.DataFrame({"judge": "logged_driver", "ep": np.arange(len(X)), "front": fr, "rear": rr & ~fr,
                         "t_crash": -1.0, "speed": V[:, 0, T_START:-30].mean(1), "hard_brakes": 0, "no_safe_plan": 0,
                         "tailgate": (gap < 0.5 * V[:, 0, T_START:-30]).mean(1),
                         "ttc_viol": ((clo > 0) & (gap < 1.5 * np.maximum(clo, 1e-6))).sum(1), "min_gap": gap.min(1)})


def summarize(df, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    eps = df.ep.unique()
    print(f"{'judge':16s} {'front crash':>22s} {'rear crash':>22s} {'speed':>6s} {'hard/ep':>7s} {'nosafe':>6s} "
          f"{'tailgate':>8s} {'TTC<1.5/ep':>10s} {'min gap p5':>10s}")
    for j, g in df.groupby("judge", sort=False):
        g = g.set_index("ep").loc[eps]
        bs = rng.integers(0, len(eps), (B, len(eps)))
        f, r = g.front.values.astype(float), g.rear.values.astype(float)
        ci = lambda x: (x.mean(), *np.percentile(x[bs].mean(1), [2.5, 97.5]))  # noqa: E731
        print(f"{j:16s} {'%.3f [%.3f, %.3f]' % ci(f):>22s} {'%.3f [%.3f, %.3f]' % ci(r):>22s} "
              f"{g.speed.mean():6.2f} {g.hard_brakes.mean():7.2f} {g.no_safe_plan.mean():6.2f} {g.tailgate.mean():8.3f} "
              f"{g.ttc_viol.mean():10.2f} {np.percentile(g.min_gap, 5):10.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summarize(pd.concat([pd.read_csv(f) for f in sys.argv[2:]]))
        sys.exit()
    prefix, out, judges = sys.argv[1], sys.argv[2], sys.argv[3:]
    dev = os.environ.get("OBJ_DEV", "cpu")
    dfs = [logged(prefix)]
    for j in judges:
        dfs.append(run(prefix, j, dev))
        print(f"{judge_name(j)} done", flush=True)
        pd.concat(dfs).to_csv(out, index=False)
    summarize(pd.concat(dfs))
