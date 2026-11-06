"""Appendix figures, same style as the main figures.
  app_scenes.pdf    all 18 OmniNWM scenes: extra generated-car displacement vs extra ego displacement over time; and a
                    dose fan for one moving scene (braking, logged, mild, crashing plans)
  app_gallery.pdf   four more scenes at the real contact, logged vs crashing plan, real car's box projected in green
  app_rerender.pdf  test-time detection fix: OmniNWM under the crashing plan vs its past-only world re-rendered from the
                    crashing plan's cameras (scene of Fig. 1)
  app_lookahead.pdf where the plan enters decides the leak; policy frontiers; SWTC as reward
  app_phantom.pdf   judged vs truly safe share per candidate plan (simulator)
Usage: python figs_app.py <windows_merged.json> <out_omni>"""
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from style import BLUE, GRAY, GREEN, INK, LBLUE, LGRAY, LRED, RED, clean, plt
from teaser import box_faces, project

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
sys.path.insert(0, str(HERE.parents[1] / "src"))
FRONT = 1.3 + 4.1 / 2
TR = json.load(open(HERE / "omni_tracks_all.json"))


def _gen(tr, pl):
    g = np.array([np.nan if v is None else v for v in tr["plans"][pl]["gen"]])
    out = g.copy()
    for k in range(len(g)):                       # light 3-frame median against single-frame readout noise
        w = g[max(k - 1, 0):k + 2]
        out[k] = np.nanmedian(w) if np.isfinite(w).any() else np.nan
    return out


def scenes():
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.1), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.28))
    for tok, tr in TR.items():
        real = np.array(tr["real_rear"])
        ec = np.array(tr["plans"]["collide_s0_n25"]["ego_front"])
        el = np.array(tr["plans"]["logged_s0_n25"]["ego_front"])
        hit = np.nonzero(real - ec <= 0)[0]
        k1 = hit[0] + 1 if len(hit) else len(ec)
        de = (ec - el)[:k1]
        dg = (_gen(tr, "collide_s0_n25") - _gen(tr, "logged_s0_n25"))[:k1]
        a.plot(de, dg, color=RED, lw=0.8, alpha=0.55)
        if np.isfinite(dg[-1]):
            a.plot(de[-1], dg[-1], "o", ms=2.6, color=RED, mec="white", mew=0.3)
    m = 12.5
    a.plot([0, m], [0, m], color=INK, lw=0.9, ls="--")
    a.plot([0, m], [0, 0], color=GREEN, lw=1.6)
    a.text(0.4, 11.6, r"dashed: $\beta=1$, keeps its gap", fontsize=6.5, ha="left", va="center")
    a.text(12.3, 0.45, "truth: the car does not react", fontsize=6.5, ha="right", va="bottom", color=GREEN)
    a.set_xlim(0, m)
    a.set_ylim(-2.5, m)
    a.set_xlabel("extra ego advance under the crashing plan (m)", labelpad=1)
    a.set_ylabel("extra advance of the generated car (m)", labelpad=1)
    a.set_title("(a) all 18 scenes, up to the real contact", fontsize=7.5, loc="left", pad=3)
    clean(a, "both")
    tok = "8d67c4024af6470faada45dc41ba9a16"
    tr = TR[tok]
    t = np.array(tr["t"])
    plans = (("d-2_s0_n25", "brake $-2$", BLUE), ("logged_s0_n25", "logged", GRAY),
             ("d1_s0_n25", "speed up $+1$", "#e8a09a"), ("collide_s0_n25", "crash $+3.75$", RED))
    for pl, lab, col in plans:
        b.plot(t, tr["plans"][pl]["ego_front"], color=col, lw=1.4)
        b.plot(t, _gen(tr, pl), color=col, lw=1.2, ls=(0, (3, 1.5)))
    b.plot(t, tr["real_rear"], color=GREEN, lw=2.2)
    hs = [Line2D([], [], color=c, lw=2.5) for _, _, c in plans] + [Line2D([], [], color=GREEN, lw=2.2),
                                                                     Line2D([], [], color=INK, lw=1.3),
                                                                     Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 1.5)))]
    b.legend(hs, [p[1] for p in plans] + ["real car", "ego front", "generated car"], fontsize=6.0, loc="upper left",
             ncol=2, handlelength=1.6, columnspacing=0.8, labelspacing=0.25, borderpad=0.35)
    b.set_xlabel("time (s)", labelpad=1)
    b.set_ylabel("distance ahead (m)", labelpad=1)
    b.set_ylim(0, 62)
    b.set_xlim(0, t[-1])
    b.set_title("(b) one scene, four plans (m/s$^2$)", fontsize=7.5, loc="left", pad=3)
    clean(b, "both")
    fig.savefig(HERE / "app_scenes.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / "app_scenes.png", bbox_inches="tight", pad_inches=0.01, dpi=200)


def _frame_with_box(ax, d, pl, k, w, t, col, label=None):
    vid = iio.imread(d / pl / "rgb.mp4")
    z = np.load(d / pl / "front3.npz")
    M = np.load(d / pl / "rel_poses.npy")
    H, W = vid.shape[1] // 2, vid.shape[2] // 3
    y0, y1 = int(0.25 * H), H
    ax.imshow(vid[k][:H, W:2 * W][y0:y1], extent=(0, W, y1, y0), aspect="auto")
    lt = np.array(w["lead_track_t0frame"])
    cx = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 0])
    cy = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 1])
    _, rear = box_faces(cx, cy, *w["lead_LW"])
    uv, dep = project(rear, M[k], z["cam2ego"][1, k].astype(np.float64), z["cam_k"][1, k].astype(np.float64))
    if (dep > 0.3).all():
        ax.add_patch(Polygon(uv, closed=True, fc=GREEN, alpha=0.16, lw=0))
        ax.add_patch(Polygon(uv, closed=True, fill=False, ec=GREEN, lw=1.2))
    ax.set_xlim(60, W - 60)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(col)
        sp.set_linewidth(1.5)
    if label:
        ax.text(0.02, 0.05, label, transform=ax.transAxes, fontsize=6.2, color="white",
                bbox=dict(fc="black", alpha=0.6, lw=0, pad=1.0))


def gallery(windows, out_omni):
    W = {x["token"]: x for x in json.load(open(windows))}
    S = {r["token"]: r for r in pd.read_csv(R / "a37_final" / "scores18.csv").to_dict("records")}
    toks = ["6b3c851413064994bb0babae7e74efad", "8d67c4024af6470faada45dc41ba9a16",
            "918b8f22899b460abb6903efe15d663f", "b4799bade00a402693dfa295adeea027"]
    fig, axs = plt.subplots(len(toks), 2, figsize=(5.5, 1.36 * len(toks)), gridspec_kw=dict(wspace=0.03, hspace=0.2))
    for r, tok in enumerate(toks):
        w, d = W[tok], Path(out_omni) / tok
        tr = TR[tok]
        t = np.array(tr["t"])
        real = np.array(tr["real_rear"])
        k = int(np.nonzero(real - np.array(tr["plans"]["collide_s0_n25"]["ego_front"]) <= 0)[0][0])
        s = S[tok]
        _frame_with_box(axs[r, 0], d, "logged_s0_n25", k, w, t, BLUE)
        _frame_with_box(axs[r, 1], d, "collide_s0_n25", k, w, t, RED)
        state = "stopped" if s["lead_stopped"] else "moving"
        axs[r, 0].set_title(f"{s['scene']}, {state} car ahead, logged plan, $t$ = {t[k]:.1f} s", fontsize=7, loc="left",
                            pad=2)
        axs[r, 1].set_title(f"crashing plan (+{s['a_star']:.2f} m/s$^2$): real contact, $\\beta$ = {s['beta']:.2f}",
                            fontsize=7, loc="left", pad=2)
    fig.savefig(HERE / "app_gallery.pdf", bbox_inches="tight", pad_inches=0.01, dpi=220)
    fig.savefig(HERE / "app_gallery.png", bbox_inches="tight", pad_inches=0.01, dpi=150)


def rerender(windows, out_omni):
    from a37_rerender import render
    tok = "ad26ae1c24e14afd9bc1e0bcc4e378b7"
    w = {x["token"]: x for x in json.load(open(windows))}[tok]
    d = Path(out_omni) / tok
    tr = TR[tok]
    t = np.array(tr["t"])
    real = np.array(tr["real_rear"])
    k_hit = int(np.nonzero(real - np.array(tr["plans"]["collide_s0_n25"]["ego_front"]) <= 0)[0][0])
    G = {}
    for pl in ("collide_s0_n25", "pm-1.5_s0_n25"):
        G[pl] = {"z": np.load(d / pl / "front3.npz"), "M": np.load(d / pl / "rel_poses.npy"),
                 "v": iio.imread(d / pl / "rgb.mp4")}
    c, q = G["collide_s0_n25"], G["pm-1.5_s0_n25"]
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 2.75), gridspec_kw=dict(wspace=0.03, hspace=0.16))
    for col_i, k in enumerate((k_hit - 8, k_hit - 4)):
        img, _ = render(q["v"][k, :448, 800:1600], q["z"]["depth"][1, k], q["z"]["seg"][1, k], q["z"]["cam_k"][1, k],
                        q["z"]["cam2ego"][1, k], q["M"][k], c["z"]["cam_k"][1, k], c["z"]["cam2ego"][1, k], c["M"][k])
        for row, (im, colr) in enumerate(((c["v"][k, :448, 800:1600], RED), (img, BLUE))):
            ax = axs[row, col_i]
            ax.imshow(im[100:448], extent=(0, 800, 448, 100), aspect="auto")
            lt = np.array(w["lead_track_t0frame"])
            cx = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 0])
            cy = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 1])
            _, rear = box_faces(cx, cy, *w["lead_LW"])
            uv, dep = project(rear, c["M"][k], c["z"]["cam2ego"][1, k].astype(np.float64),
                              c["z"]["cam_k"][1, k].astype(np.float64))
            if (dep > 0.3).all():
                ax.add_patch(Polygon(uv, closed=True, fill=False, ec=GREEN, lw=1.2))
            ax.set_xlim(60, 740)
            ax.set_ylim(448, 100)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(colr)
                sp.set_linewidth(1.5)
            if row == 0:
                ax.set_title(f"$t$ = {t[k]:.2f} s, {t[k_hit] - t[k]:.2f} s before the real contact", fontsize=7.5, pad=2)
    axs[0, 0].set_ylabel("OmniNWM given\nthe crashing plan", fontsize=7, color=RED)
    axs[1, 0].set_ylabel("past-only world,\nre-rendered", fontsize=7, color=BLUE)
    fig.savefig(HERE / "app_rerender.pdf", bbox_inches="tight", pad_inches=0.01, dpi=220)
    fig.savefig(HERE / "app_rerender.png", bbox_inches="tight", pad_inches=0.01, dpi=150)


def vista():
    root = Path.home() / "work/a37_pod/out_vista"
    toks = ["8d67c4024af6470faada45dc41ba9a16", "918b8f22899b460abb6903efe15d663f"]
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 2.3), gridspec_kw=dict(wspace=0.03, hspace=0.08))
    for r, tok in enumerate(toks):
        for c, (pl, lab, col) in enumerate((("logged", "Vista, logged plan, +2.0 s", BLUE),
                                            ("collide", "Vista, crashing plan, +2.0 s", RED))):
            im = plt.imread(root / tok / f"{pl}_s0" / "f24.jpg")
            H = im.shape[0]
            ax = axs[r, c]
            ax.imshow(im[int(0.18 * H):int(0.92 * H)], aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(col)
                sp.set_linewidth(1.5)
            if r == 0:
                ax.set_title(lab, fontsize=7.5, pad=2)
    fig.savefig(HERE / "app_vista.pdf", bbox_inches="tight", pad_inches=0.01, dpi=220)
    fig.savefig(HERE / "app_vista.png", bbox_inches="tight", pad_inches=0.01, dpi=120)


def _frontier(F, fam, grid):
    out = []
    for m in [c for c in F.model.unique() if c == fam or c.startswith(fam + "_s")]:
        for sd in F.seed.unique():
            f = F[(F.model == m) & (F.seed == sd)].sort_values("a")
            a, c = f.a.to_numpy(), f.crash.to_numpy()
            c = np.minimum.accumulate(c[::-1])[::-1]
            out.append([np.interp(p, a, c) if a.min() <= p <= a.max() else np.nan for p in grid])
    return np.nanmean(out, 0)


def lookahead():
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.75), gridspec_kw=dict(width_ratios=[1.1, 1.2, 0.95], wspace=0.42))
    a = ax[0]
    names = ["causal", "0.5 s", "1 s", "1.5 s", "3 s", "whole"]
    vals = [[0.00, -0.01, 0.00], [0.74, 0.72, 0.66], [0.44], [0.41], [0.38], [0.68, 0.62, 0.61]]
    for i, v in enumerate(vals):
        lk = i > 0
        a.bar(i, np.mean(v) if abs(np.mean(v)) > 0.012 else 0.012, 0.62, color=LRED if lk else LBLUE,
              ec=RED if lk else BLUE, lw=0.8, zorder=2)
        if len(v) > 1:
            a.plot([i] * len(v), v, "o", ms=2.0, color=INK, zorder=3)
    a.axhline(0.67, color=GRAY, lw=0.8, ls="--")
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, fontsize=6.3, rotation=35, ha="right", rotation_mode="anchor")
    a.set_xlabel("plan seen ahead", labelpad=1)
    a.set_ylabel(r"plan leak $\beta$", labelpad=1)
    a.set_ylim(-0.05, 0.9)
    a.set_title("(a) plan seen ahead", fontsize=7.5, loc="left", pad=3)
    clean(a)
    b = ax[1]
    F = pd.read_parquet(R / "data" / "a32_frontier.parquet")
    grid = np.linspace(1.5, 4.5, 31)
    for fam, lab, col, ls in (("truth", "true world", INK, ":"), ("step0", "causal", BLUE, "-"),
                              ("step1", "next 0.5 s", RED, "-"), ("pm_step1", "next 0.5 s, plan-marginal", "#8c6bb1", "--"),
                              ("chunk", "whole plan", GRAY, "-")):
        b.plot(grid, 100 * _frontier(F, fam, grid), color=col, ls=ls, lw=1.3, label=lab)
    b.set_xlabel("progress (mean extra accel., m/s$^2$)", labelpad=1)
    b.set_ylabel("real crash rate (%)", labelpad=1)
    b.set_ylim(0, 10)
    b.legend(fontsize=5.8, loc="upper left", handlelength=1.6, labelspacing=0.2, borderpad=0.3)
    b.set_title("(b) policies trained in the model", fontsize=7.5, loc="left", pad=3)
    clean(b, "both")
    c = ax[2]
    lab = ["self-judge", "one past plan", "plan-marginal", "retrained"]
    v = [6.5, 2.5, 2.1, 1.4]
    for i, x in enumerate(v):
        c.bar(i, x, 0.62, color=LRED if i == 0 else LBLUE, ec=RED if i == 0 else BLUE, lw=0.8, zorder=2)
        c.text(i, x + 0.12, f"{x:.1f}", ha="center", va="bottom", fontsize=6.0)
    c.set_xticks(range(4))
    c.set_xticklabels(lab, fontsize=6.0, rotation=35, ha="right", rotation_mode="anchor")
    c.set_ylabel("real crash rate (%)", labelpad=1)
    c.set_ylim(0, 7.6)
    c.set_title("(c) SWTC as reward", fontsize=7.5, loc="left", pad=3)
    clean(c)
    fig.savefig(HERE / "app_lookahead.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / "app_lookahead.png", bbox_inches="tight", pad_inches=0.01, dpi=200)


def phantom():
    d = json.load(open(HERE / "phantom_data.json"))
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    names = {"chunk": "(a) whole-plan model", "step0": "(b) causal model", "gc_chunk": "(c) whole plan + g-computation"}
    for ax, m in zip(axs, ("chunk", "step0", "gc_chunk")):
        cnd = np.array(d[m]["cand"])
        sel = cnd >= -4
        ax.plot(cnd[sel], 100 * np.array(d[m]["true"])[sel], "-o", color=GRAY, lw=1.3, ms=2.6, label="truly safe")
        ax.plot(cnd[sel], 100 * np.array(d[m]["judged"])[sel], "-o", color=RED if m == "chunk" else BLUE, lw=1.3, ms=2.6,
                label="judged safe")
        ax.set_title(names[m], fontsize=7.5, loc="left", pad=3)
        ax.set_xlabel("candidate plan (m/s$^2$)", labelpad=1)
        clean(ax, "both")
    axs[0].set_ylabel("states (%)", labelpad=1)
    axs[0].legend(fontsize=6.0, loc="lower left", handlelength=1.6, borderpad=0.3)
    axs[0].annotate("braking judged less\nsafe than cruising", xy=(-2, 60), xytext=(0.9, 82), fontsize=6.0, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color=INK))
    fig.savefig(HERE / "app_phantom.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / "app_phantom.png", bbox_inches="tight", pad_inches=0.01, dpi=200)


def relation():
    """Leak by relation in the occupancy models against the coupling in the training logs (numbers as in
    audit3/fig_leak_law.py: results/relation_leak.txt, ii_relation_leak.txt, dome_relation_leak.txt, obs_beta_v2.txt)."""
    rels = ["lead", "follower", "adjacent", "oncoming", "parked"]
    band = {"lead": (0.67, 0.96), "follower": (0.53, 0.92), "adjacent": (0.52, 0.73), "oncoming": (-0.35, -0.15),
            "parked": (0.00, 0.01)}
    models = {"SWTC": ("o", INK, {"lead": (1.01, 0.99, 1.07), "follower": (0.75, 0.61, 0.93),
                                  "adjacent": (0.69, 0.13, 0.79), "oncoming": (-0.21, -0.51, -0.05),
                                  "parked": (0.07, 0.04, 0.14)}),
              "DOME": ("D", GRAY, {"lead": (0.91, 0.73, 1.03), "follower": (0.64, 0.33, 0.86),
                                   "parked": (0.00, 0.00, 0.11)}),
              "I$^2$-World": ("s", BLUE, {"lead": (0.54, 0.30, 0.71), "follower": (0.46, 0.40, 0.57),
                                          "oncoming": (0.00, -0.11, 0.00), "parked": (0.00, 0.00, 0.00)})}
    fig, ax = plt.subplots(figsize=(5.0, 1.75))
    for i, r in enumerate(rels):
        lo, hi = band[r]
        ax.add_patch(plt.Rectangle((i - 0.36, lo - 0.02), 0.72, hi - lo + 0.04, color=LGRAY, lw=0, zorder=1))
        if r != "follower":                         # a follower may truly react to the ego: true effect unknown
            ax.plot([i - 0.36, i + 0.36], [0, 0], color=GREEN, lw=1.4, ls=(0, (3, 1.5)), zorder=2)
    for j, (m, (mk, col, d)) in enumerate(models.items()):
        for i, r in enumerate(rels):
            if r in d:
                v, lo, hi = d[r]
                ax.errorbar(i + (j - 1) * 0.2, v, yerr=[[v - lo], [hi - v]], fmt=mk, color=col, ms=3.6, lw=0.9,
                            capsize=1.6, zorder=3, label=m if r == "lead" else None)
    ax.axhline(0, color=GRAY, lw=0.5, zorder=0)
    ax.set_xticks(range(len(rels)), ["car ahead", "car behind", "adjacent,\nsame direction", "oncoming", "parked"])
    ax.set_ylabel(r"plan leak $\beta$ (+2 s)", labelpad=1)
    ax.set_ylim(-0.6, 1.15)
    ax.set_xlim(-0.5, len(rels) - 0.5)
    clean(ax, "y")
    h, lab = ax.get_legend_handles_labels()
    h += [plt.Rectangle((0, 0), 1, 1, color=LGRAY), Line2D([], [], color=GREEN, lw=1.4, ls=(0, (3, 1.5)))]
    lab += ["coupling in the training logs", "true effect (0)"]
    ax.legend(h, lab, fontsize=6.2, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              handlelength=1.5, columnspacing=1.0)
    fig.savefig(HERE / "app_relation.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / "app_relation.png", bbox_inches="tight", pad_inches=0.01, dpi=200)


if __name__ == "__main__":
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["scenes", "lookahead", "phantom", "vista", "gallery", "rerender"]
    for f in which:
        globals()[f](*sys.argv[1:3]) if f in ("gallery", "rerender") else globals()[f]()
