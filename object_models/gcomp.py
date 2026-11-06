"""A36: test-time g-computation of FROZEN plan-conditioned world models.
A model trained on logs to predict the world given the whole future ego plan learns E[world | h, plan], which leaks the
other agents' futures through the logged plan (A31). The g-formula needs p(world_{t+1} | h_t), with the ego entering
only through its realised past. Any frozen plan-conditioned model gives it at test time: at each 0.5 s step, show the
model the ego only up to now, average its next-step prediction over past-only continuations of the plan (law of total
probability; uniform prior over the A30 prior plans), keep that step, append the candidate's next ego step to the
history and repeat. Unlike the plan-marginal query (A30, one-shot average over whole prior plans), the candidate plan
still enters through the ego's realised past, so agents that react to the ego (followers) keep reacting.
Wrappers (by checkpoint name; the frozen weights are the A31 checkpoints, nothing is trained):
  gc_<ckpt>  test-time g-computation of <ckpt>     pm_<ckpt>  plan-marginal positions (mean over prior plans)
Importing this module patches obj_wm.rollout / obj_wm.load_net so obj_wm / obj_rl stages accept these names.
Stages:
  leak <val.npz> <out.parquet> <ckpt> ...        obj_wm.leak (gc_ / pm_ names allowed)
  ade  <val.npz> <ckpt> ...                       rollout ADE under the logged plan, all val windows
  grid <npz> <out.parquet> <sel> --ckpt ...       obj_rl.grid
  frontier <train_grid> <eval_grid> <out>         matched-progress frontier with the A36 contrasts
  belief <eval_grid> <het_dir> ...                believed / true safe extra accel (A32 P1 statistic)
Usage (server, vla env): python src/gcomp.py leak $W/val.npz $W/a36_leak.parquet $W/gc_chunk.pt ..."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
import obj_wm as o  # noqa: E402

PRIOR = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)         # A30 / A32 past-only prior plans (constant accel)
_roll, _load = o.rollout, o.load_net


def frame(X, pose):
    """positions X (..., 3) (x, y, yaw) into the frame of pose (B, 3); X has batch first."""
    import torch
    sh = (X.shape[0],) + (1,) * (X.dim() - 2)
    px, py, pw = (pose[:, k].view(sh) for k in range(3))
    c, s = torch.cos(pw), torch.sin(pw)
    dx, dy = X[..., 0] - px, X[..., 1] - py
    yaw = (X[..., 2] - pw + np.pi) % (2 * np.pi) - np.pi
    return torch.stack([c * dx + s * dy, -s * dx + c * dy, yaw], -1)


def unframe(p, pose):
    """2D positions p (B, NA, 2) from the frame of pose (B, 3) back to the t0 frame."""
    import torch
    c, s = torch.cos(pose[:, 2])[:, None], torch.sin(pose[:, 2])[:, None]
    return torch.stack([pose[:, 0, None] + c * p[..., 0] - s * p[..., 1],
                        pose[:, 1, None] + s * p[..., 0] + c * p[..., 1]], -1)


def straight(sp, ak, n, dev):
    """distance travelled at t = 0.5 .. 0.5 n under constant accel ak from speed sp (B,), never reversing (B, n)."""
    import torch
    t = 0.5 * torch.arange(1, n + 1, device=dev)
    d = sp[:, None] * t + 0.5 * ak * t * t
    return torch.cummax(d.clamp(min=0), 1).values if ak < 0 else d


def cont(E, s, ak):
    """E with the ego steps after s replaced by the past-only continuation ak (along the heading at s)."""
    import torch
    E = E.clone()
    n = o.T - 1 - s
    if n <= 0:
        return E
    sp = (E[:, s, :2] - E[:, s - 1, :2]).norm(dim=-1) / 0.5
    yaw = E[:, s, 2]
    d = straight(sp, ak, n, E.device)
    E[:, s + 1:, 0] = E[:, s, None, 0] + d * torch.cos(yaw)[:, None]
    E[:, s + 1:, 1] = E[:, s, None, 1] + d * torch.sin(yaw)[:, None]
    E[:, s + 1:, 2] = yaw[:, None]
    return E


def chunk_next(net, E, A, M, S, s, ak):
    """next-step agent positions (t0 frame) from a one-shot model queried at step s: history s-3..s re-expressed in the
    ego frame at s (the frame the model was trained in), ego future = past-only continuation ak."""
    import torch
    P = o.P
    pose = E[:, s]
    E2 = torch.zeros_like(E)
    E2[:, :P + 1] = frame(E[:, s - P: s + 1], pose)
    sp = (E[:, s, :2] - E[:, s - 1, :2]).norm(dim=-1) / 0.5
    E2[:, P + 1:, 0] = straight(sp, ak, o.T - 1 - P, E.device)
    A2 = torch.zeros_like(A)
    A2[:, :, :P + 1] = frame(A[:, :, s - P: s + 1], pose)
    M2 = M.clone()
    M2[:, :, :P + 1] = M[:, :, s - P: s + 1]
    M2[:, :, P + 1:] = M[:, :, s, None]
    return unframe(_roll(net, E2, A2, M2, S)[:, :, 0], pose)


def gc_rollout(net, E, A, M, S, prior=PRIOR):
    """test-time g-computation: agent positions for steps 4 .. 9 (B, NA, 6, 2) under the ego trajectory E."""
    import torch
    A, M = A.clone(), M.clone()
    out = []
    for s in range(o.P, o.T - 1):
        ps = []
        for ak in prior:
            if net.mode.startswith("step"):
                ps.append(o.cv_base(A, M, s, 1)[:, :, 0] + net(cont(E, s, ak), A, M, S, s) / 2.0)
            else:
                ps.append(chunk_next(net, E, A, M, S, s, ak))
        p = torch.stack(ps).mean(0)
        d = p - A[:, :, s, :2]
        yaw = torch.where(d.norm(dim=-1) > 0.2, torch.atan2(d[..., 1], d[..., 0]), A[:, :, s, 2])
        A[:, :, s + 1] = torch.cat([p, yaw[..., None]], -1)
        M[:, :, s + 1] = M[:, :, s]
        out.append(p)
    return torch.stack(out, 2)


def pm_rollout(net, E, A, M, S, prior=PRIOR):
    """plan-marginal positions: mean over whole prior plans from t0 (the candidate plan is ignored)."""
    import torch
    return torch.stack([_roll(net, cont(E, o.P, ak), A, M, S) for ak in prior]).mean(0)


class Wrap:
    def __init__(self, kind, net):
        self.kind, self.net, self.mode = kind, net, f"{kind}_{net.mode}"


def roll(net, E, A, M, S):
    if isinstance(net, Wrap):
        return (gc_rollout if net.kind == "gc" else pm_rollout)(net.net, E, A, M, S)
    return _roll(net, E, A, M, S)


def load_net(ckpt, dev="cuda"):
    p = Path(ckpt)
    for kind in ("gc", "pm"):
        if p.stem.startswith(kind + "_"):
            return Wrap(kind, _load(str(p.with_name(p.stem[len(kind) + 1:] + p.suffix)), dev))
    return _load(ckpt, dev)


o.rollout, o.load_net = roll, load_net


def ade(npz, ckpts):
    import torch
    D, _, _ = o.load(npz, "cuda")
    n = len(D["E"])
    for ck in ckpts:
        net = load_net(ck)
        errs = []
        with torch.no_grad():
            for b in range(0, n, 512):
                j = torch.arange(b, min(b + 512, n), device="cuda")
                E, A, M, S = (D[k][j] for k in ("E", "A", "M", "S"))
                pred = roll(net, E, A, M, S)
                ok = (M[:, :, o.P] & M[:, :, o.P - 1])[:, :, None] & M[:, :, o.P + 1:]
                errs.append((pred - A[:, :, o.P + 1:, :2]).norm(dim=-1)[ok])
        e = torch.cat(errs)
        print(f"{Path(ck).stem:14s} logged-plan rollout ADE {float(e.mean()):.3f} m (n {len(e)})", flush=True)


def frontier(train_path, eval_path, out, seeds=(0, 1, 2)):
    """obj_rl.frontier (A32, exploratory there) with the A36 registered contrasts; same policy class / lambda sweep."""
    import obj_rl as R
    tr = pd.read_parquet(train_path).sort_values(["token", "a"]).reset_index(drop=True)
    ev = pd.read_parquet(eval_path).sort_values(["token", "a"]).reset_index(drop=True)
    G, n = R.GRID, len(R.GRID)
    fc = ["f_v0", "f_gap", "f_lv", "f_has"]
    Ftr, Fev = tr[fc].to_numpy(np.float32)[::n], ev[fc].to_numpy(np.float32)[::n]
    sc_ev = ev.scene.to_numpy()[::n]
    truth_ev = (ev.truth >= 1.0).to_numpy().reshape(-1, n)
    models = [c for c in tr.columns if c not in ("token", "scene", "a") and not c.startswith("f_") and c in ev]
    rows, picks = [], {}
    for m in models:
        crash_tr = (tr[m] >= 1.0).to_numpy().reshape(-1, n).astype(np.float32)
        for sd in seeds:
            for lam in 2.0 ** np.arange(3, 10.01, 0.5):
                pk = R.train_policy(Ftr, G[None].astype(np.float32) - lam * crash_tr, seed=sd)(Fev)
                picks[(m, sd, lam)] = pk
                rows.append({"model": m, "seed": sd, "lam": lam, "a": G[pk].mean(),
                             "crash": truth_ev[np.arange(len(pk)), pk].mean()})
        print(f"  {m} done", flush=True)
    F = pd.DataFrame(rows)
    F.to_parquet(out)

    def at(Fm, p):
        f = Fm.sort_values("a").groupby("a", as_index=False).crash.min()
        return float(np.interp(p, f.a, f.crash)) if f.a.min() <= p <= f.a.max() else np.nan
    print("real crash at matched mean extra accel (mean over policy seeds):")
    for p in (2.5, 3.0, 3.5, 4.0):
        print(f"  progress {p}: " + " | ".join(
            f"{m} {np.nanmean([at(F[(F.model == m) & (F.seed == sd)], p) for sd in seeds]):.3f}" for m in models))
    rng = np.random.default_rng(0)
    sc = np.unique(sc_ev)
    idx = {s: np.where(sc_ev == s)[0] for s in sc}
    pairs = [("gc_step1", "step0"), ("gc_chunk", "step0"), ("step1", "gc_step1"), ("chunk", "gc_chunk"),
             ("gc_step1", "pm_step1"), ("step1", "step0")]
    for p in (3.0, 4.0):
        for a_, b_ in pairs:
            groups = [(f"{a_}{s}", f"{b_}{s}") for s in ("", "_s1", "_s2") if f"{a_}{s}" in models and f"{b_}{s}" in models]
            if not groups:
                continue
            ds = []
            for _ in range(300):
                ii = np.concatenate([idx[s] for s in rng.choice(sc, len(sc))])
                vals = []
                for ga, gb in groups:
                    for sd in seeds:
                        fa = pd.DataFrame([{"a": G[pk[ii]].mean(), "crash": truth_ev[ii, pk[ii]].mean()}
                                           for (m, s_, lam), pk in picks.items() if s_ == sd and m == ga])
                        fb = pd.DataFrame([{"a": G[pk[ii]].mean(), "crash": truth_ev[ii, pk[ii]].mean()}
                                           for (m, s_, lam), pk in picks.items() if s_ == sd and m == gb])
                        vals.append(at(fa, p) - at(fb, p))
                ds.append(np.nanmean(vals))
            pt = np.nanmean([at(F[(F.model == ga) & (F.seed == sd)], p) - at(F[(F.model == gb) & (F.seed == sd)], p)
                             for ga, gb in groups for sd in seeds])
            print(f"  matched progress {p}: {a_} - {b_} {pt:+.3f} [{np.nanquantile(ds, .025):+.3f}, "
                  f"{np.nanquantile(ds, .975):+.3f}] ({len(groups)} WM seeds x {len(seeds)} policy seeds)", flush=True)


def belief(path, hets):
    """A32 P1 statistic for any grid column: median believed / true largest safe extra accel on lead windows."""
    import pickle
    import obj_rl as R
    df = pd.read_parquet(path).sort_values(["token", "a"])
    n = len(R.GRID)
    strat = {}
    for het in hets:
        T = pickle.load(open(Path(het) / "acc" / "truth.pkl", "rb"))
        strat |= {k: v["stratum"] for k, v in T.items()}
    W = pd.DataFrame({"token": df.token.to_numpy()[::n], "scene": df.scene.to_numpy()[::n]})
    W["stratum"] = W.token.map(strat)
    W["a_true"] = R.first_safe(df, "truth")
    cols = [c for c in df.columns if c not in ("token", "scene", "a", "truth") and not c.startswith("f_")]
    for c in cols:
        W[f"a_{c}"] = R.first_safe(df, c)
    V = W[(W.stratum != "free") & (W.a_true > 0) & (W.a_true < R.GRID[-1])]
    rng = np.random.default_rng(0)
    sc = V.scene.unique()
    byS = {s: V[V.scene == s] for s in sc}
    boots = [pd.concat([byS[s] for s in rng.choice(sc, len(sc))]) for _ in range(1000)]
    print(f"belief: {len(V)} lead windows ({len(sc)} scenes)")
    for c in cols:
        r = V[f"a_{c}"] / V.a_true
        bs = [float((b[f"a_{c}"] / b.a_true).median()) for b in boots]
        print(f"  {c:12s} median ratio {r.median():.2f} [{np.quantile(bs, .025):.2f}, {np.quantile(bs, .975):.2f}] | "
              f"share >= 1.5x {(r >= 1.5).mean():.2f} | censored {(V[f'a_{c}'] >= R.GRID[-1]).mean():.2f}")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "leak":
        o.leak(sys.argv[2], sys.argv[4:], sys.argv[3])
    elif st == "ade":
        ade(sys.argv[2], sys.argv[3:])
    elif st == "grid":
        import obj_rl as R
        a = sys.argv[2:]
        k = a.index("--ckpt")
        R.grid(a[0], a[1], a[2], a[k + 1:])
    elif st == "frontier":
        frontier(sys.argv[2], sys.argv[3], sys.argv[4])
    elif st == "belief":
        belief(sys.argv[2], sys.argv[3:])
