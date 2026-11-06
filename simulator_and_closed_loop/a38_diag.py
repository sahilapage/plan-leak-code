"""A38 exploratory diagnostic (written after the A38 hard-brake count was seen, before this script ran):
WHY does the whole-plan (chunk) planner brake hard ~6x per episode when no other judge does?
Hypothesis (A35: chunk lead beta +1.59): in the chunk model the lead follows the ego's plan, so a moderate brake is
predicted to pull the lead back into the ego (unsafe) while accelerating is predicted to push it away (safe); only the
extreme -6 / -8 plans, outside the logged plan range where chunk's leak collapses (A32), look safe -> phantom hard brakes.
Prediction: at the judge's hard-brake decisions (a <= -3), the TRUE do() future (oracle, same state) keeps a gentler
plan (a >= -2) in >= 50 % of them (phantom); for step0 the share is lower.
Replays the closed loop exactly as src/a38_closed.py run() (same planner, same world), and at every decision also
evaluates the oracle on the same state (the oracle never drives). Per decision: judge pick, judge / oracle kept masks,
judge / oracle front values per candidate.
Usage: python audit3/a38_diag.py <test_prefix> <judge ckpt> <out.npz>   (A38_HEADWAY as in a38_closed.py)"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import a38_closed as A  # noqa: E402
import causal_sim as C  # noqa: E402


def fronts(xe, v, lead, fol):
    ve = np.stack([np.maximum(v[:, None] + a * 0.5 * np.arange(1, 7), 0) for a in A.CAND_A], 1)
    front = ((xe + C.EGO_LEN / 2) - (lead - C.LEN / 2) + A.HEADWAY * ve).max(-1)
    rear = ((fol + C.LEN / 2) - (xe - C.EGO_LEN / 2)).max(-1)
    return front, rear


def main(prefix, judge, out, dev="cpu"):
    import torch
    torch.set_num_threads(int(os.environ.get("A38_THREADS", 12)))
    import gcomp  # noqa: F401
    import obj_wm as o
    H = dict(np.load(f"{prefix}_hist.npz"))
    p = {k[2:]: H[k] for k in H if k.startswith("p_")}
    nz = {k[2:]: H[k] for k in H if k.startswith("n_")}
    X, V, Acc = H["X"].copy(), H["V"].copy(), H["Acc"].copy()
    B, n = X.shape[0], X.shape[2]
    net = A.load_judge(judge, dev)
    cand = np.array(A.CAND_A)
    nc = len(cand)
    alive = np.ones(B, bool)
    a_cmd = np.zeros(B)
    rec = {k: [] for k in ("t", "alive", "pick", "jf", "jr", "of", "orr", "gap", "v", "vl")}
    for t in range(A.T_START, n - 31):
        if (t - A.T_START) % C.KF == 0:
            v = V[:, 0, t]
            d10 = {a: A.ego_plan(v, a, 30) for a in A.CAND_A}
            xe = np.stack([d10[a][:, A.KFT - 1] for a in A.CAND_A], 1)
            Y = np.stack([A.oracle(X, V, Acc, p, nz, t, d10[a]) for a in A.CAND_A], 1)
            of, orr = fronts(xe, v, Y[:, :, 1], Y[:, :, 2])
            E, Aa, M, S, J = A.window(X, t)
            Ec = np.repeat(E[:, None], nc, 1)
            Ec[:, :, 4:, 0] = xe
            tt = lambda z: torch.tensor(z.reshape((-1,) + z.shape[2:]), device=dev)  # noqa: E731
            rep = lambda z: np.repeat(z[:, None], nc, 1)                               # noqa: E731
            with torch.no_grad():
                Yq = A.predict(net, tt(Ec), tt(rep(Aa)), tt(rep(M)), tt(rep(S)))[..., 0].cpu().numpy()
            Yq = Yq.reshape(B, nc, A.NA, 6)
            big = np.full((B, nc, 6), 1e3)
            lead, fol = big.copy(), -big.copy()
            for e in range(B):
                for q in np.where(J[e] == 1)[0]:
                    lead[e] = Yq[e, :, q]
                for q in np.where(J[e] == 2)[0]:
                    fol[e] = Yq[e, :, q]
            jf, jr = fronts(xe, v, lead, fol)
            f_ok = jf <= -A.MARGIN
            ok = f_ok & (jr <= -A.MARGIN)
            a_des = np.clip(0.5 * (p["v0_e"] - v), -2, 2)
            cost = np.abs(cand[None] - a_des[:, None]) + 1e-3 * cand[None]
            pick = np.where(ok.any(1), np.where(ok, cost, np.inf).argmin(1),
                            np.where(f_ok.any(1), np.where(f_ok, jr, np.inf).argmin(1), jf.argmin(1)))
            a_cmd = cand[pick]
            for k, val in (("t", np.full(B, t)), ("alive", alive.copy()), ("pick", pick), ("jf", jf), ("jr", jr),
                           ("of", of), ("orr", orr), ("gap", X[:, 1, t] - X[:, 0, t] - C.LEN), ("v", v.copy()),
                           ("vl", V[:, 1, t].copy())):
                rec[k].append(val)
        V[:, 0, t + 1] = np.maximum(V[:, 0, t] + a_cmd * C.DT, 0)
        X[:, 0, t + 1] = X[:, 0, t] + V[:, 0, t + 1] * C.DT
        Acc[:, 0, t] = (V[:, 0, t + 1] - V[:, 0, t]) / C.DT
        a_f = C.follower_acc(p, nz, X, V, Acc, t)
        Acc[:, 2, t] = a_f
        V[:, 2, t + 1] = np.maximum(V[:, 2, t] + a_f * C.DT, 0)
        X[:, 2, t + 1] = X[:, 2, t] + V[:, 2, t + 1] * C.DT
        fr = alive & (X[:, 1, t + 1] - X[:, 0, t + 1] < C.LEN)
        rr = alive & (X[:, 0, t + 1] - X[:, 2, t + 1] < C.LEN)
        alive &= ~(fr | rr)
    np.savez_compressed(out, cand=cand, X=X, V=V, **{k: np.stack(v_, 1) for k, v_ in rec.items()})
    R = {k: np.stack(v_, 1) for k, v_ in rec.items()}
    hard = R["alive"] & (cand[R["pick"]] <= -3)
    okk = (R["of"] <= -A.MARGIN) & (R["orr"] <= -A.MARGIN)
    gentle = (okk & (cand >= -2)[None, None]).any(-1)
    print(f"{A.judge_name(judge)}: decisions {R['alive'].sum()}, hard {hard.sum()}, phantom (oracle keeps a >= -2) "
          f"{(hard & gentle).sum()} = {(hard & gentle).sum() / max(hard.sum(), 1):.3f}")
    jok = (R["jf"] <= -A.MARGIN)
    print("  share judged front-safe per candidate", dict(zip(cand, (jok & R["alive"][..., None]).sum((0, 1))
                                                               / R["alive"].sum())))
    print("  share truly front-safe per candidate ", dict(zip(cand, ((R["of"] <= -A.MARGIN) & R["alive"][..., None])
                                                               .sum((0, 1)) / R["alive"].sum())))


if __name__ == "__main__":
    main(*sys.argv[1:4], dev=os.environ.get("OBJ_DEV", "cpu"))
