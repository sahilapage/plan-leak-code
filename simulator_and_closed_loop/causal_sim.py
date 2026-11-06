"""A35: a driving world whose causal effects are KNOWN, to test the leak law and the fixes against ground truth.
One straight road at 10 Hz, logged at 2 Hz in the obj_wm window format (3 past + t0 + 6 future keyframes):
  lead      exogenous: speed tracks a traffic factor (OU) with random hard-braking events; cannot react to the ego
  ego       the LOGGING driver: IDM on the lead seen with a reaction delay (0.5-1.0 s) plus brake-light anticipation
            (feed-forward of the lead's delayed acceleration) and OU noise -> the logged plan carries the lead's future
  follower  reacts to the ego (IDM with delay 0.4-0.8 s and brake-light anticipation): a TRUE causal effect of the plan
  adjacent  2 cars in the next lane driven by the SAME traffic factor as the lead (rho 0.7) and shared braking
            events: correlated with the ego plan through a common cause, no causal effect
  parked    8 static cars on the shoulder
Interventions: do(ego follows plan) from t0 re-simulates only the follower (everything else is exogenous), so the
true effect of the plan is 0 for lead / adjacent / parked and is computed exactly for the follower.
Stages:
  make <out_prefix> <n_episodes> <seed> [slow]   episodes -> <prefix>.npz (obj_wm format + rel ids) and
                                                  <prefix>_hist.npz (10 Hz histories + parameters for do())
  coupling <prefix>                               log coupling per relation (history-aware / -blind slopes, A31 law)
  eval <test_prefix> <out_prefix> <ckpt> ...      leak beta vs TRUE beta, judge errors under reactive truth, belief
Usage (server, vla env): python src/causal_sim.py make $W/sim_train 1200 0 | coupling ... | eval ..."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DT, KF, NSTEP = 0.1, 5, 450                  # 10 Hz, 5 steps per 0.5 s keyframe, 45 s episodes
NAG = 13                                     # 0 ego, 1 lead, 2 follower, 3-4 adjacent, 5-12 parked
REL = np.array([0, 1, 2, 3, 3] + [4] * 8)    # 1 lead, 2 follower, 3 adjacent, 4 parked
LEN, WID = 4.5, 1.9
LANE_Y = np.array([0.0, 0.0, 0.0, 3.5, 3.5] + [-3.6] * 8)
EGO_LEN = 4.084


def ou(rng, shape, tau, n):
    z = np.zeros(shape + (n,))
    a = np.exp(-DT / tau)
    e = rng.standard_normal(shape + (n,)) * np.sqrt(1 - a * a)
    z[..., 0] = rng.standard_normal(shape)
    for t in range(1, n):
        z[..., t] = a * z[..., t - 1] + e[..., t]
    return z


def events(rng, B, n, rate, dur):
    """boolean braking timeline (B, n): Poisson onsets at `rate` per s, each lasting U(dur) s, plus its depth."""
    on = np.zeros((B, n), bool)
    depth = np.ones((B, n))
    for b in range(B):
        t = rng.exponential(1 / rate)
        while t < n * DT:
            i0, i1 = int(t / DT), int((t + rng.uniform(*dur)) / DT)
            on[b, i0:i1] = True
            depth[b, i0:i1] = rng.uniform(0.0, 0.5)
            t += rng.exponential(1 / rate) + (i1 - i0) * DT
    return on, depth


def params(rng, B, slow=False):
    p = {"vb": rng.uniform(4, 15, B), "gap0": rng.uniform(12, 35, B),
         "tau_e": rng.uniform(0.8, 1.4, B) if slow else rng.uniform(0.5, 1.0, B),
         "T_e": rng.uniform(1.0, 2.0, B), "v0_e": None, "c_e": rng.uniform(0.3, 0.7, B),
         "tau_f": rng.uniform(0.4, 0.8, B), "T_f": rng.uniform(0.8, 1.6, B), "gap0_f": rng.uniform(8, 25, B),
         "c_f": rng.uniform(0.3, 0.7, B), "vb_adj": rng.uniform(4, 15, B), "x_adj": rng.uniform(-25, 35, (B, 2)),
         "x_park": rng.uniform(0, 650, (B, 8))}
    p["v0_e"] = p["vb"] * rng.uniform(1.3, 1.6, B) + 3.0
    p["v0_f"] = p["v0_e"] + 3.0
    return p


def noises(rng, B, n):
    z = ou(rng, (B,), 8.0, n)
    zi = ou(rng, (B, 2), 8.0, n)
    on, depth = events(rng, B, n, 1 / 12, (2.0, 5.0))
    on_a = on[:, None].repeat(2, 1) & (rng.random((B, 2, 1)) < 0.5)
    return {"z": z, "zi": zi, "on": on, "depth": depth, "on_a": on_a, "eta": rng.standard_normal((B, 3, n)),
            "ne": ou(rng, (B,), 1.0, n), "nf": ou(rng, (B,), 1.0, n)}


def idm(v, gap, dv, v0, T, a_max=2.0, b=3.0, s0=2.5):
    ss = s0 + np.maximum(0, v * T + v * dv / (2 * np.sqrt(a_max * b)))
    return a_max * (1 - (v / np.maximum(v0, 0.1)) ** 4 - (ss / np.maximum(gap, 0.1)) ** 2)


def exo_acc(p, nz, X, V, t):
    """accelerations of lead (B,) and adjacent (B, 2) at step t (exogenous: they never look at ego / follower)."""
    rho = 0.7
    vt_l = p["vb"] * (1 + 0.25 * nz["z"][:, t])
    vt_l = np.where(nz["on"][:, t], vt_l * nz["depth"][:, t], vt_l).clip(0)
    a_l = np.clip(0.8 * (vt_l - V[:, 1, t]), -6, 2) + 0.2 * nz["eta"][:, 0, t]
    za = rho * nz["z"][:, None, t] + np.sqrt(1 - rho ** 2) * nz["zi"][:, :, t]
    vt_a = p["vb_adj"][:, None] * (1 + 0.25 * za)
    vt_a = np.where(nz["on_a"][:, :, t], vt_a * nz["depth"][:, None, t], vt_a).clip(0)
    a_a = np.clip(0.8 * (vt_a - V[:, 3:5, t]), -6, 2) + 0.2 * nz["eta"][:, 1:3, t]
    return a_l, a_a


def follower_acc(p, nz, X, V, Acc, t):
    d = np.maximum(t - np.round(p["tau_f"] / DT).astype(int), 0)
    b = np.arange(len(d))
    gap = X[b, 0, d] - X[b, 2, d] - LEN
    dv = V[:, 2, t] - V[b, 0, d]
    a = idm(V[:, 2, t], gap, dv, p["v0_f"], p["T_f"]) + p["c_f"] * np.minimum(Acc[b, 0, d], 0) + 0.3 * nz["nf"][:, t]
    return np.clip(a, -8, 3)


def simulate(p, nz, B, n=NSTEP):
    X, V, Acc = np.zeros((B, NAG, n)), np.zeros((B, NAG, n)), np.zeros((B, NAG, n))
    X[:, 1, 0] = p["gap0"] + LEN
    X[:, 2, 0] = -p["gap0_f"] - LEN
    X[:, 3:5, 0] = p["x_adj"]
    X[:, 5:, :] = p["x_park"][:, :, None]
    V[:, :5, 0] = p["vb"][:, None]
    V[:, 3:5, 0] = p["vb_adj"][:, None]
    b = np.arange(B)
    de = np.round(p["tau_e"] / DT).astype(int)
    for t in range(n - 1):
        a_l, a_a = exo_acc(p, nz, X, V, t)
        d = np.maximum(t - de, 0)
        gap = X[b, 1, d] - X[b, 0, d] - LEN
        dv = V[:, 0, t] - V[b, 1, d]
        a_e = idm(V[:, 0, t], gap, dv, p["v0_e"], p["T_e"]) + p["c_e"] * np.minimum(Acc[b, 1, d], 0) + \
            0.3 * nz["ne"][:, t]
        a_e = np.clip(a_e, -8, 3)
        a_f = follower_acc(p, nz, X, V, Acc, t)
        Acc[:, 0, t], Acc[:, 1, t], Acc[:, 2, t], Acc[:, 3:5, t] = a_e, a_l, a_f, a_a
        for j in range(5):
            V[:, j, t + 1] = np.maximum(V[:, j, t] + Acc[:, j, t] * DT, 0)
            X[:, j, t + 1] = X[:, j, t] + V[:, j, t + 1] * DT
    return X, V, Acc


def make(prefix, n_ep, seed, slow=False):
    rng = np.random.default_rng(seed)
    p = params(rng, n_ep, slow)
    nz = noises(rng, n_ep, NSTEP)
    X, V, Acc = simulate(p, nz, n_ep)
    crash = (X[:, 1] - X[:, 0] < LEN) | (X[:, 0] - X[:, 2] < LEN)          # bumper overlap, lead / follower
    T_, P_ = 10, 3
    E, A, M, S, REL_, TOK, SCN, T0 = [], [], [], [], [], [], [], []
    for e in range(n_ep):
        for k in range(P_ + 4, NSTEP // KF - 7, 2):                            # one window per s, 2 s burn-in
            ts = np.arange(k - P_, k + 7) * KF
            if crash[e, ts[0]: ts[-1] + 1].any():
                continue
            x0 = X[e, 0, k * KF]
            ego = np.stack([X[e, 0, ts] - x0, np.zeros(T_), np.zeros(T_)], -1)
            near = [j for j in range(1, NAG) if abs(X[e, j, k * KF] - x0) <= 60]
            a = np.zeros((24, T_, 3), np.float32)
            m = np.zeros((24, T_), bool)
            sz = np.zeros((24, 2), np.float32)
            rl = np.zeros(24, np.int8)
            for q, j in enumerate(near):
                a[q, :, 0] = X[e, j, ts] - x0
                a[q, :, 1] = LANE_Y[j]
                m[q] = True
                sz[q] = (LEN, WID)
                rl[q] = REL[j] * 16 + j                                         # relation * 16 + agent index
            E.append(ego.astype(np.float32))
            A.append(a)
            M.append(m)
            S.append(sz)
            REL_.append(rl)
            TOK.append(f"{seed}_{e}_{k}")
            SCN.append(f"ep{seed}_{e}")
            T0.append((e, k * KF))
    np.savez_compressed(f"{prefix}.npz", E=np.stack(E), A=np.stack(A), M=np.stack(M), S=np.stack(S),
                        rel=np.stack(REL_), tok=np.array(TOK), scene=np.array(SCN), t0=np.array(T0))
    np.savez_compressed(f"{prefix}_hist.npz", X=X, V=V, Acc=Acc, **{f"p_{k}": v for k, v in p.items()},
                        **{f"n_{k}": v for k, v in nz.items()})
    print(f"{prefix}: {len(E)} windows from {n_ep} episodes (crash-free windows; episodes with a crash: "
          f"{crash.any(1).mean():.3f})")


def coupling(prefix):
    """the A31 leak law in this world: per relation, slope of the agent's 2 s displacement on the ego's, against
    their constant-velocity extrapolations (history-aware) and raw (history-blind); plus corr of accelerations."""
    z = np.load(f"{prefix}.npz")
    A, E, rel = z["A"], z["E"], z["rel"] // 16
    P_, h = 3, 4                                                                # +2 s = 4 keyframes
    de = E[:, P_ + h, 0] - E[:, P_, 0]
    dcv_e = h * (E[:, P_, 0] - E[:, P_ - 1, 0])
    for r, name in ((1, "lead"), (2, "follower"), (3, "adjacent"), (4, "parked")):
        i, j = np.where(rel == r)
        da = A[i, j, P_ + h, 0] - A[i, j, P_, 0]
        dcv = h * (A[i, j, P_, 0] - A[i, j, P_ - 1, 0])
        x, y = de[i] - dcv_e[i], da - dcv
        aware = np.cov(y, x)[0, 1] / np.var(x)
        blind = np.cov(da, de[i])[0, 1] / np.var(de[i])
        print(f"  {name:9s} n {len(i):6d} | history-aware slope {aware:+.2f} | history-blind slope {blind:+.2f}")


def do_follower(H, e, t0, ego_x, n=31):
    """re-simulate the follower of episode e from step t0 with the ego forced to ego_x (n,) at steps t0 .. t0+n-1."""
    X, V, Acc = (H[k][e: e + 1].copy() for k in ("X", "V", "Acc"))
    p = {k[2:]: H[k][e: e + 1] for k in H if k.startswith("p_")}
    nz = {k[2:]: H[k][e: e + 1] for k in H if k.startswith("n_")}
    for t in range(t0, t0 + n - 1):
        X[0, 0, t + 1] = ego_x[t + 1 - t0]
        V[0, 0, t + 1] = (ego_x[t + 1 - t0] - ego_x[t - t0]) / DT
        Acc[0, 0, t] = (V[0, 0, t + 1] - V[0, 0, t]) / DT
        a_f = follower_acc(p, nz, X, V, Acc, t)
        V[0, 2, t + 1] = max(V[0, 2, t] + a_f[0] * DT, 0)
        X[0, 2, t + 1] = X[0, 2, t] + V[0, 2, t + 1] * DT
    return X[0, 2, t0: t0 + n]


def plan_x(Xe, t0, a):
    """ego positions t0 .. t0+30 (10 Hz): logged + 0.5 a t^2, never reversing (as obj_wm.acc_wp)."""
    t = DT * np.arange(31)
    x = Xe[t0: t0 + 31] - Xe[t0] + 0.5 * a * t * t
    if a < 0:
        x = np.maximum.accumulate(np.maximum(x, 0))
    return x + Xe[t0]


LEAK_A = (2.0, -1.5)
CAND_A = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
BELIEF_A = np.round(np.arange(0.0, 8.01, 0.25), 2)


def eval_(prefix, out, ckpts):
    """registered A35 outputs: (1) leak beta per relation for every model vs the true beta; (2) judge errors on
    candidate plans under REACTIVE truth (front: ego into lead; rear: follower into ego, penetration >= 1 m at any 0.5 s
    keyframe); (3) believed vs true largest safe extra accel (lead)."""
    import pandas as pd
    import torch
    import gcomp  # noqa: F401  (patches obj_wm for gc_ / pm_ names)
    import obj_wm as o
    z = np.load(f"{prefix}.npz")
    H = dict(np.load(f"{prefix}_hist.npz"))
    Xh = H["X"]
    D = {k: torch.tensor(z[k], device="cuda") for k in ("E", "A", "M", "S")}
    rel, scn, t0s = z["rel"], z["scene"], z["t0"]
    N = len(scn)
    nets = {Path(c).stem: o.load_net(c) for c in ckpts}
    kf = np.arange(1, 7) * KF

    def plans(i, alist):
        e, t0 = t0s[i]
        return {a: plan_x(Xh[e, 0], t0, a) for a in alist}

    # ---- truth under do(): ego keyframes and agent keyframes for every plan in every window
    def truth(i, alist):
        e, t0 = t0s[i]
        x0 = Xh[e, 0, t0]
        out_ = {}
        for a, xe in plans(i, alist).items():
            xf = do_follower(H, e, t0, xe)
            ag = {}
            for q in range(24):
                if rel[i, q] == 0:
                    continue
                j = rel[i, q] % 16
                ag[q] = (xf[kf] if j == 2 else Xh[e, j, t0 + kf]) - x0
            out_[a] = (xe[kf] - x0, ag)
        return out_

    def predict(net, idx, alist):
        """predicted agent x at keyframes (len(idx) * len(alist), 24, 6) for plans in alist."""
        res = []
        for b in range(0, len(idx), 256):
            ii = idx[b: b + 256]
            j = torch.tensor(ii, device="cuda")
            E, A, M, S = (D[k][j] for k in ("E", "A", "M", "S"))
            wp = E[:, o.P + 1:, :2]
            Ys = []
            for a in alist:
                xe = torch.tensor(np.stack([plans(i, [a])[a][kf] - Xh[t0s[i][0], 0, t0s[i][1]] for i in ii]),
                                  device="cuda", dtype=torch.float32)
                with torch.no_grad():
                    Ys.append(o.rollout(net, o.plan_E(E, torch.stack([xe, wp[..., 1]], -1)), A, M, S)[..., 0])
            res.append(torch.stack(Ys, 1).cpu().numpy())
        return np.concatenate(res)                                              # (n, len(alist), 24, 6)

    rows = []
    idx = np.arange(N)
    T_leak = {i: truth(i, (0.0,) + LEAK_A) for i in idx}
    for name, net in nets.items():
        Y = predict(net, idx, (0.0,) + LEAK_A)
        for i in idx:
            tr = T_leak[i]
            for ai, a in enumerate(LEAK_A, 1):
                dxe = tr[a][0] - tr[0.0][0]
                for q in tr[0.0][1]:
                    for h in (1, 2, 3):
                        k = 2 * h - 1
                        if abs(dxe[k]) < 0.5:
                            continue
                        rows.append({"model": name, "scene": scn[i], "rel": int(rel[i, q] // 16), "a": a, "h": h,
                                     "beta": float((Y[i, ai, q, k] - Y[i, 0, q, k]) / dxe[k]),
                                     "beta_true": float((tr[a][1][q][k] - tr[0.0][1][q][k]) / dxe[k])})
        print(f"leak {name} done", flush=True)
    pd.DataFrame(rows).to_parquet(f"{out}_leak.parquet")

    # ---- judge: front (lead) and rear (follower) penetration >= 1 m at any keyframe, predicted vs reactive truth
    def pen(xe, ag, i):
        front, rear = -99.0, -99.0
        for q, xa in ag.items():
            r = rel[i, q] // 16
            if r == 1:
                front = max(front, float(np.max((xe + EGO_LEN / 2) - (xa - LEN / 2))))
            elif r == 2:
                rear = max(rear, float(np.max((xa + LEN / 2) - (xe - EGO_LEN / 2))))
        return front, rear
    jrows = []
    T_c = {i: truth(i, CAND_A) for i in idx}
    preds = {name: predict(net, idx, CAND_A) for name, net in nets.items()}
    for i in idx:
        for ci, a in enumerate(CAND_A):
            xe, ag = T_c[i][a]
            f, r = pen(xe, ag, i)
            row = {"scene": scn[i], "win": i, "a": a, "front_true": f, "rear_true": r}
            for name in nets:
                pf, pr = pen(xe, {q: preds[name][i, ci, q] for q in ag}, i)
                row[f"front_{name}"], row[f"rear_{name}"] = pf, pr
            jrows.append(row)
    pd.DataFrame(jrows).to_parquet(f"{out}_judge.parquet")
    print("judge done", flush=True)

    # ---- belief: largest safe extra accel (front only, lead windows), believed vs true
    lead_w = np.array([(rel[i] // 16 == 1).any() for i in idx])
    li = idx[lead_w]
    bp = {name: predict(net, li, tuple(BELIEF_A)) for name, net in nets.items()}
    brows = []
    for n_, i in enumerate(li):
        e, t0 = t0s[i]
        x0 = Xh[e, 0, t0]
        q = int(np.where(rel[i] // 16 == 1)[0][0])
        xl = Xh[e, rel[i, q] % 16, t0 + kf] - x0

        def first(fr):
            c = np.array(fr) >= 1.0
            k_ = int(c.argmax()) if c.any() else len(BELIEF_A)
            return 0.0 if k_ == 0 else float(BELIEF_A[max(k_, 1) - 1])
        xes = [plan_x(Xh[e, 0], t0, a)[kf] - x0 for a in BELIEF_A]
        row = {"scene": scn[i], "win": i,
               "a_true": first([np.max((xe + EGO_LEN / 2) - (xl - LEN / 2)) for xe in xes])}
        for name in nets:
            row[f"a_{name}"] = first([np.max((xe + EGO_LEN / 2) - (bp[name][n_, k, q] - LEN / 2))
                                      for k, xe in enumerate(xes)])
        brows.append(row)
    pd.DataFrame(brows).to_parquet(f"{out}_belief.parquet")
    print(f"eval -> {out}_*.parquet", flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "make":
        make(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), len(sys.argv) > 5 and sys.argv[5] == "slow")
    elif st == "coupling":
        coupling(sys.argv[2])
    elif st == "eval":
        eval_(sys.argv[2], sys.argv[3], sys.argv[4:])
