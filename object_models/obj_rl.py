"""A32: does the leak law predict reward hacking? (registered 12cec89)
A plan-conditioned WM with lead leak beta makes the closing under an extra acceleration a look (1 - beta) of the real
one, so the acceleration it believes safe is the real limit / (1 - beta), and a policy trained inside it learns to
tailgate. Instruments: the A31 object WMs (src/obj_wm.py) + GT-CV physics + truth (annotated tracks, non-reactive).
  grid   <npz> <out.parquet> <sel> --ckpt ...   predicted / GT-CV / truth overlap depth for every extra-accel grid value
         (0 .. 8 m/s^2, step 0.25) along each window's logged path; sel = a29 (A29 tokens) | train (filtered train)
  belief <grid.parquet> <het_dir> ...           P1 / P2: believed vs true safe acceleration ratio per window
  policy <train_grid.parquet> <a29_grid.parquet> <het_dir> ...   P3: softmax policies trained inside each WM
Usage (server, vla env): python src/obj_rl.py grid ... | belief ... | policy ..."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit3"))
import obj_wm as o  # noqa: E402

GRID = np.round(np.arange(0.0, 8.01, 0.25), 2)
PRIOR = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)        # src/het_marg.py prior plans (past-only)
PM = ("chunk", "step1")
PEN = True                                            # A32 deviation 1 geometry
BETA = {"step0": 0.00, "marg": 0.0, "step6": 0.38, "step3": 0.41, "step2": 0.44, "chunk": 0.68, "step1": 0.74}


def depth_t(Y, LW, ahead, wp):
    """vectorised src/sw_hetero.py depth: max over +1/+2/+3 s and agents ahead of the longitudinal overlap (m).
    Y (B, NA, 6, 2) agent positions at +0.5 .. +3 s, LW (B, NA, 2), ahead (B, NA) bool, wp (B, 6, 2) ego plan."""
    import torch
    from fixed_acc import FRONT
    from sw_hetero import REAR
    from sw_realhazard import EGO_W
    if PEN:
        # A32 deviation 1: penetration of the ego front beyond the rear of a laterally overlapping vehicle that was
        # ahead at t0, at every 0.5 s step (no pass-through between checks)
        xe, ye = wp[:, :, 0][:, None], wp[:, :, 1][:, None]
        lx, ly = Y[..., 0], Y[..., 1]
        L, W = LW[..., 0:1], LW[..., 1:2]
        lat = (ly - ye).abs() <= (EGO_W + W) / 2
        ov = (xe + FRONT) - (lx - L / 2)
        ov = torch.where(lat & ahead[..., None], ov, torch.full_like(ov, -99.0))
        return ov.flatten(1).max(1).values
    k = [1, 3, 5]
    xe, ye = wp[:, k, 0][:, None], wp[:, k, 1][:, None]
    lx, ly = Y[:, :, k, 0], Y[:, :, k, 1]
    L, W = LW[..., 0:1], LW[..., 1:2]
    lat = (ly - ye).abs() <= (EGO_W + W) / 2
    ov = torch.minimum(xe + FRONT, lx + L / 2) - torch.maximum(xe + REAR, lx - L / 2)
    ov = torch.where(lat & ahead[..., None], ov, torch.full_like(ov, -99.0))
    return ov.flatten(1).max(1).values


def select(D, tok, sel):
    import torch
    if sel == "train":
        E, A, M = D["E"], D["A"], D["M"]
        dyaw = (E[:, -1, 2] - E[:, o.P, 2] + np.pi) % (2 * np.pi) - np.pi
        ok = (dyaw.abs() <= np.radians(15)) & (E[:, o.P + 1:, 1].abs().max(1).values <= 2.0) & (E[:, -1, 0] >= 9.0)
        full = M[:, :, o.P:].all(-1) & (A[:, :, o.P, 0] > 0)
        d = depth_t(A[:, :, o.P + 1:, :2], D["S"], full, E[:, o.P + 1:, :2])
        ok &= d <= -0.5
        return torch.where(ok)[0].cpu().numpy()
    toks = set()
    for het in sel.split(","):
        toks |= set(pd.read_parquet(Path(het) / "table.parquet").token)
    return np.array([i for i, t in enumerate(tok) if t in toks])


def grid(npz, out, sel, ckpts, bs=64):
    import torch
    D, tok, scn = o.load(npz, "cuda")
    ix = select(D, tok, sel)
    print(f"{len(ix)} windows selected ({sel})", flush=True)
    nets = [o.load_net(c) for c in ckpts]
    names = [Path(c).stem for c in ckpts]
    G = torch.tensor(GRID, device="cuda", dtype=torch.float32)
    n = len(GRID)
    rows = {"token": [], "scene": [], "a": []}
    cols = {k: [] for k in ["truth", "gtcv"] + names}
    feat = []
    for b in range(0, len(ix), bs):
        j = torch.tensor(ix[b: b + bs], device="cuda")
        E, A, M, S = (D[k][j] for k in ("E", "A", "M", "S"))
        B = len(j)
        wp = E[:, o.P + 1:, :2]
        t = 0.5 * torch.arange(1, 7, device="cuda")
        wpa = (wp[:, None] + torch.stack([0.5 * G[:, None] * t * t, torch.zeros(n, 6, device="cuda")], -1)[None])
        wpa = wpa.reshape(B * n, 6, 2)
        rep = lambda x: x.repeat_interleave(n, 0)  # noqa: E731
        Er, Ar, Mr, Sr = rep(E), rep(A), rep(M), rep(S)
        Er = o.plan_E(Er, wpa)
        ahead = Mr[:, :, o.P] & (Ar[:, :, o.P, 0] > 0)
        full = Mr[:, :, o.P:].all(-1) & (Ar[:, :, o.P, 0] > 0)
        cols["truth"].append(depth_t(Ar[:, :, o.P + 1:, :2], Sr, full, wpa))
        cols["gtcv"].append(depth_t(o.cv_base(Ar, Mr, o.P, 6), Sr, ahead, wpa))
        with torch.no_grad():
            for net, nm in zip(nets, names):
                cols[nm].append(depth_t(o.rollout(net, Er, Ar, Mr, Sr), Sr, ahead, wpa))
                if net.mode in PM:
                    # A32 amendment: plan-marginal query, K past-only prior plans (straight, const accel from v0)
                    v0 = ((E[:, o.P, 0] - E[:, o.P - 1, 0]) / 0.5).clamp(min=0)
                    dk = []
                    for ak in PRIOR:
                        s_ = v0[:, None] * t + 0.5 * ak * t * t
                        if ak < 0:
                            s_ = torch.cummax(torch.maximum(s_, torch.zeros_like(s_)), 1).values
                        Ek = o.plan_E(E, torch.stack([s_, torch.zeros_like(s_)], -1))
                        Yk = rep(o.rollout(net, Ek, A, M, S))
                        dk.append(depth_t(Yk, Sr, ahead, wpa))
                    cols.setdefault(f"pm_{nm}", []).append(torch.stack(dk, 1).median(1).values)
        for q in range(B):
            gi = int(ix[b + q])
            rows["token"] += [tok[gi]] * n
            rows["scene"] += [scn[gi]] * n
            rows["a"] += GRID.tolist()
        # t0 features (past-only): ego speed, in-lane lead gap / speed / presence
        v0 = (E[:, o.P, 0] - E[:, o.P - 1, 0]) / 0.5
        inl = M[:, :, o.P] & (A[:, :, o.P, 1].abs() <= 1.5) & (A[:, :, o.P, 0] >= 2) & (A[:, :, o.P, 0] <= 50)
        gx = torch.where(inl, A[:, :, o.P, 0], torch.full_like(A[:, :, o.P, 0], 99.0))
        g0, jl = gx.min(1)
        lv = (A[torch.arange(B), jl, o.P, 0] - A[torch.arange(B), jl, o.P - 1, 0]) / 0.5 * M[torch.arange(B), jl, o.P - 1]
        has = g0 < 99
        f = torch.stack([v0 / 10, torch.where(has, g0, torch.full_like(g0, 60.0)) / 30,
                         torch.where(has, lv, v0) / 10, has.float()], -1)
        feat.append(f.repeat_interleave(n, 0))
        if b % (bs * 20) == 0:
            print(f"  {b + B} / {len(ix)}", flush=True)
    df = pd.DataFrame(rows)
    for k, v in cols.items():
        df[k] = torch.cat(v).cpu().numpy()
    F = torch.cat(feat).cpu().numpy()
    for q, name in enumerate(["f_v0", "f_gap", "f_lv", "f_has"]):
        df[name] = F[:, q]
    df.to_parquet(out)
    print(f"{df.token.nunique()} windows x {n} -> {out}")


def first_safe(df, col):
    """largest grid a before the first grid value with overlap >= 1 m (0 if crash at a = 0; 8 if never)."""
    c = (df[col] >= 1.0).to_numpy().reshape(-1, len(GRID))
    first = np.where(c.any(1), c.argmax(1), len(GRID))
    return np.where(first == 0, 0.0, GRID[np.maximum(first, 1) - 1])


def belief(path, hets):
    from scipy.stats import spearmanr
    df = pd.read_parquet(path).sort_values(["token", "a"])
    strat = {}
    for het in hets:
        T = pickle.load(open(Path(het) / "acc" / "truth.pkl", "rb"))
        strat |= {k: v["stratum"] for k, v in T.items()}
    toks = df.token.unique()
    W = pd.DataFrame({"token": df.token.to_numpy()[:: len(GRID)], "scene": df.scene.to_numpy()[:: len(GRID)]})
    assert (W.token.to_numpy() == toks).all()
    W["stratum"] = W.token.map(strat)
    models = [c for c in ["step0", "marg", "step6", "step3", "step2", "chunk", "step1", "gtcv", "pm_chunk", "pm_step1"]
              if c in df]
    W["a_true"] = first_safe(df, "truth")
    for m in models:
        W[f"a_{m}"] = first_safe(df, m)
    V = W[(W.stratum != "free") & (W.a_true > 0) & (W.a_true < GRID[-1])].copy()
    print(f"belief test: {len(V)} lead windows with a finite true limit ({V.scene.nunique()} scenes); "
          f"median true safe extra accel {V.a_true.median():.2f}")
    med = {}
    rng = np.random.default_rng(0)
    sc = V.scene.unique()
    byS = {s: V[V.scene == s] for s in sc}
    boots = [pd.concat([byS[s] for s in rng.choice(sc, len(sc))]) for _ in range(1000)]
    for m in models:
        r = V[f"a_{m}"] / V.a_true
        med[m] = float(r.median())
        bs = [float((b[f"a_{m}"] / b.a_true).median()) for b in boots]
        cens = float((V[f"a_{m}"] >= GRID[-1]).mean())
        pred = 1 / (1 - BETA[m]) if m in BETA else float("nan")
        print(f"  {m:6s} median ratio {med[m]:.2f} [{np.quantile(bs, .025):.2f}, {np.quantile(bs, .975):.2f}] | law "
              f"1/(1-beta) {pred:.2f} | mean believed {V[f'a_{m}'].mean():.2f} vs true {V.a_true.mean():.2f} | "
              f"censored at 8: {cens:.2f} | share believed >= 1.5 x true {(r >= 1.5).mean():.2f}")
    if "chunk" in med and "step0" in med:
        d = [float((b.a_chunk / b.a_true).median() - (b.a_step0 / b.a_true).median()) for b in boots]
        print(f"  P1: chunk {med['chunk']:.2f} (>= 1.5?) | step0 {med['step0']:.2f} (in [0.8, 1.25]?)")
        print(f"  P2: chunk - step0 median ratio {med['chunk'] - med['step0']:+.2f} [{np.quantile(d, .025):+.2f}, "
              f"{np.quantile(d, .975):+.2f}]")
    ms = [m for m in models if m in BETA]
    print(f"  P2: Spearman(median ratio, 1/(1-beta)) over {len(ms)} WMs = "
          f"{spearmanr([med[m] for m in ms], [1 / (1 - BETA[m]) for m in ms])[0]:+.2f}")
    W.to_parquet(Path(path).with_name("belief_windows.parquet"))


def train_policy(F, R, steps=1500, seed=0):
    """softmax policy over GRID from t0 features F (N, 4); R (N, len(GRID)) reward table; returns a predictor."""
    import torch
    torch.manual_seed(seed)
    F, R = torch.tensor(F, device="cuda"), torch.tensor(R, device="cuda")
    net = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(),
                              torch.nn.Linear(64, len(GRID))).cuda()
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(steps):
        p = torch.softmax(net(F), -1)
        loss = -(p * R).sum(-1).mean() - 0.01 * -(p * torch.log(p + 1e-9)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda X: net(torch.tensor(X, device="cuda")).argmax(-1).cpu().numpy()


def policy(train_path, eval_path, hets):
    tr = pd.read_parquet(train_path).sort_values(["token", "a"])
    ev = pd.read_parquet(eval_path).sort_values(["token", "a"])
    n = len(GRID)
    strat = {}
    for het in hets:
        T = pickle.load(open(Path(het) / "acc" / "truth.pkl", "rb"))
        strat |= {k: v["stratum"] for k, v in T.items()}
    fc = ["f_v0", "f_gap", "f_lv", "f_has"]
    Ftr = tr[fc].to_numpy(np.float32)[::n]
    Fev = ev[fc].to_numpy(np.float32)[::n]
    ev_scene = ev.scene.to_numpy()[::n]
    truth_ev = (ev.truth >= 1.0).to_numpy().reshape(-1, n)
    models = [c for c in ["truth", "gtcv", "step0", "marg", "step6", "step3", "step2", "chunk", "step1", "pm_chunk",
                          "pm_step1"] if c in tr]
    print(f"policy: train {len(Ftr)} windows, eval {len(Fev)} A29 windows ({len(set(ev_scene))} scenes)")
    truth_tr = (tr.truth >= 1.0).to_numpy().reshape(-1, n).astype(np.float32)
    lam_star = None
    for lam in 2.0 ** np.arange(1, 13):
        pk = train_policy(Ftr, GRID[None].astype(np.float32) - lam * truth_tr)(Ftr)
        cr = float(truth_tr[np.arange(len(pk)), pk].mean())
        print(f"  calibrate lambda {lam:6.0f}: truth-trained policy TRAIN crash {cr:.3f}, mean a {GRID[pk].mean():.2f}")
        if cr <= 0.02:
            lam_star = float(lam)
            break
    print(f"  lambda* = {lam_star}")
    lams = (lam_star / 2, lam_star, lam_star * 2)
    res = {}
    for lam in lams:
        line = []
        for m in models:
            crash_tr = (tr[m] >= 1.0).to_numpy().reshape(-1, n).astype(np.float32)
            R = GRID[None].astype(np.float32) - lam * crash_tr
            pick = train_policy(Ftr, R)(Fev)
            real = truth_ev[np.arange(len(pick)), pick]
            believed = (ev[m] >= 1.0).to_numpy().reshape(-1, n)[np.arange(len(pick)), pick]
            res[(lam, m)] = pd.DataFrame({"scene": ev_scene, "a": GRID[pick], "crash": real.astype(float),
                                          "believed": believed.astype(float)})
            line.append(f"{m} a {GRID[pick].mean():.2f} real {real.mean():.3f} (believed {believed.mean():.3f})")
        print(f"  lambda {lam:4.0f}: " + " | ".join(line), flush=True)
    rng = np.random.default_rng(0)
    for lam, (ma, mb) in [(l_, pr) for l_ in lams for pr in (("chunk", "step0"), ("step1", "step0"),
                                                             ("pm_chunk", "step0"), ("pm_step1", "step0"),
                                                             ("chunk", "pm_chunk"))]:
        if (lam, ma) not in res or (lam, mb) not in res:
            continue
        a, b = res[(lam, ma)], res[(lam, mb)]
        sc = np.unique(a.scene)
        ga = {s: (a[a.scene == s].crash.sum(), (a.scene == s).sum()) for s in sc}
        gb = {s: b[b.scene == s].crash.sum() for s in sc}
        d = []
        for _ in range(2000):
            ss = rng.choice(sc, len(sc))
            nn = sum(ga[s][1] for s in ss)
            d.append((sum(ga[s][0] for s in ss) - sum(gb[s] for s in ss)) / nn)
        print(f"  P3 lambda {lam:.0f}{' (primary)' if lam == lam_star else ''}: real crash {ma} - {mb} {a.crash.mean() - b.crash.mean():+.3f} "
              f"[{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]")
    pd.concat([v.assign(lam=k[0], model=k[1]) for k, v in res.items()]).to_parquet(
        Path(eval_path).with_name("policy_eval.parquet"))


def policy_seeds(train_path, eval_path, lam, seeds=(0, 1, 2)):
    """A32 robustness: real crash at a fixed lambda for every WM column (incl. WM seeds) x policy seeds."""
    tr = pd.read_parquet(train_path).sort_values(["token", "a"])
    ev = pd.read_parquet(eval_path).sort_values(["token", "a"])
    n = len(GRID)
    fc = ["f_v0", "f_gap", "f_lv", "f_has"]
    Ftr, Fev = tr[fc].to_numpy(np.float32)[::n], ev[fc].to_numpy(np.float32)[::n]
    truth_ev = (ev.truth >= 1.0).to_numpy().reshape(-1, n)
    models = [c for c in tr.columns if c not in ("token", "scene", "a") and not c.startswith("f_")]
    print(f"lambda {lam}: real crash (believed) per WM column x policy seed {seeds}")
    for m in models:
        crash_tr = (tr[m] >= 1.0).to_numpy().reshape(-1, n).astype(np.float32)
        out = []
        for sd in seeds:
            pick = train_policy(Ftr, GRID[None].astype(np.float32) - lam * crash_tr, seed=sd)(Fev)
            bel = (ev[m] >= 1.0).to_numpy().reshape(-1, n)[np.arange(len(pick)), pick].mean()
            out.append((truth_ev[np.arange(len(pick)), pick].mean(), bel, GRID[pick].mean()))
        o_ = np.array(out)
        print(f"  {m:14s} real {' '.join(f'{x:.3f}' for x in o_[:, 0])} (mean {o_[:, 0].mean():.3f}) | believed "
              f"{o_[:, 1].mean():.3f} | mean a {o_[:, 2].mean():.2f}", flush=True)


def frontier(train_path, eval_path, out, seeds=(0, 1, 2)):
    """A32 robustness (exploratory): safety-progress frontier. For each WM column and policy seed, sweep lambda
    (2^3 .. 2^10, half-octave steps); record mean chosen extra accel (progress) and real crash on the A29 windows;
    real crash at MATCHED progress (linear interpolation along each frontier) removes the lambda dependence."""
    tr = pd.read_parquet(train_path).sort_values(["token", "a"])
    ev = pd.read_parquet(eval_path).sort_values(["token", "a"])
    n = len(GRID)
    fc = ["f_v0", "f_gap", "f_lv", "f_has"]
    Ftr, Fev = tr[fc].to_numpy(np.float32)[::n], ev[fc].to_numpy(np.float32)[::n]
    sc_ev = ev.scene.to_numpy()[::n]
    truth_ev = (ev.truth >= 1.0).to_numpy().reshape(-1, n)
    models = [c for c in tr.columns if c not in ("token", "scene", "a") and not c.startswith("f_")]
    rows, picks = [], {}
    for m in models:
        crash_tr = (tr[m] >= 1.0).to_numpy().reshape(-1, n).astype(np.float32)
        for sd in seeds:
            for lam in 2.0 ** np.arange(3, 10.01, 0.5):
                pk = train_policy(Ftr, GRID[None].astype(np.float32) - lam * crash_tr, seed=sd)(Fev)
                picks[(m, sd, lam)] = pk
                rows.append({"model": m, "seed": sd, "lam": lam, "a": GRID[pk].mean(),
                             "crash": truth_ev[np.arange(len(pk)), pk].mean()})
        print(f"  {m} done", flush=True)
    F = pd.DataFrame(rows)
    F.to_parquet(out)

    def at(Fm, p):
        f = Fm.sort_values("a")
        f = f.groupby("a", as_index=False).crash.min()
        return float(np.interp(p, f.a, f.crash)) if f.a.min() <= p <= f.a.max() else np.nan
    print("real crash at matched mean extra accel (mean over policy seeds):")
    for p in (2.5, 3.0, 3.5, 4.0, 4.5):
        line = []
        for m in models:
            v = [at(F[(F.model == m) & (F.seed == sd)], p) for sd in seeds]
            line.append(f"{m} {np.nanmean(v):.3f}")
        print(f"  progress {p}: " + " | ".join(line))
    # scene bootstrap of the matched-progress difference (pooled over WM seeds and policy seeds)
    rng = np.random.default_rng(0)
    sc = np.unique(sc_ev)
    idx = {s: np.where(sc_ev == s)[0] for s in sc}
    pairs = [("step1", "step0"), ("chunk", "step0"), ("pm_step1", "step0"), ("pm_chunk", "step0"),
             ("step1", "pm_step1")]
    for p in (3.0, 4.0):
        for a_, b_ in pairs:
            groups = [(f"{a_}{s}", f"{b_}{s}") for s in ("", "_s1", "_s2")
                      if f"{a_}{s}" in models and f"{b_}{s}" in models]
            if not groups:
                continue
            ds = []
            for _ in range(300):
                ii = np.concatenate([idx[s] for s in rng.choice(sc, len(sc))])
                vals = []
                for ga, gb in groups:
                    for sd in seeds:
                        fa, fb = [], []
                        for (m, s_, lam), pk in picks.items():
                            if s_ != sd or m not in (ga, gb):
                                continue
                            (fa if m == ga else fb).append({"a": GRID[pk[ii]].mean(),
                                                            "crash": truth_ev[ii, pk[ii]].mean()})
                        vals.append(at(pd.DataFrame(fa), p) - at(pd.DataFrame(fb), p))
                ds.append(np.nanmean(vals))
            pt = np.nanmean([at(F[(F.model == ga) & (F.seed == sd)], p) - at(F[(F.model == gb) & (F.seed == sd)], p)
                             for ga, gb in groups for sd in seeds])
            print(f"  matched progress {p}: {a_} - {b_} {pt:+.3f} [{np.nanquantile(ds, .025):+.3f}, "
                  f"{np.nanquantile(ds, .975):+.3f}] ({len(groups)} WM seeds x {len(seeds)} policy seeds)", flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "grid":
        a = sys.argv[2:]
        k = a.index("--ckpt")
        grid(a[0], a[1], a[2], a[k + 1:])
    elif st == "belief":
        belief(sys.argv[2], sys.argv[3:])
    elif st == "policy":
        policy(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif st == "frontier":
        frontier(sys.argv[2], sys.argv[3], sys.argv[4])
    elif st == "policy_seeds":
        policy_seeds(sys.argv[2], sys.argv[3], float(sys.argv[4]))
