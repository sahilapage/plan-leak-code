"""Data figures (numbers from results/ and the audited claims table, audit_final/FULL_AUDIT_AND_PLAN.md):
  leak.pdf   (a) plan leak of released world models, car ahead vs parked cars; (b) same data and network, only how the
             model sees the plan
  cost.pdf   (a) policies trained inside a world model, real crash rate; (b) phantom braking as an inverted plan
             order; (c) closed-loop hard braking vs leak dose (simulator); (d) near-misses vs dose (real traffic)
Usage: python figs.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from style import BLUE, GRAY, INK, LBLUE, LGRAY, LRED, RED, clean, plt

R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parent


def bar_labels(ax, xs, vals, his, fmt="{:.2f}", dy=0.03, fs=6.3):
    for x, v, h in zip(xs, vals, his):
        ax.text(x, max(h, v) + dy, fmt.format(v), ha="center", va="bottom", fontsize=fs, color=INK)


def leak():
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.3), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.2))
    # (a) released world models: (name, lead, lo, hi, parked, plo, phi)
    rel = [("OmniNWM", 1.02, 0.98, 1.08, 0.23, 0.12, 0.60), ("Vista$^*$", 1.04, 0.97, 1.13, None, None, None),
           ("SWTC", 1.01, 0.99, 1.07, 0.07, 0.04, 0.14), ("DOME", 0.91, 0.73, 1.03, 0.00, 0.00, 0.11),
           ("I$^2$-World", 0.54, 0.30, 0.71, 0.00, 0.00, 0.00)]
    x = np.arange(len(rel))
    wd = 0.36
    for i, (n, m, lo, hi, p, plo, phi) in enumerate(rel):
        a.bar(i - wd / 2, m, wd, color=LRED, ec=RED, lw=0.8, hatch="////" if "Vista" in n else None, zorder=2)
        a.errorbar(i - wd / 2, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.7, capsize=1.6, zorder=3)
        a.text(i - wd / 2, hi + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=6.0)
        if p is not None:
            a.bar(i + wd / 2, max(p, 0.012), wd, color=LBLUE, ec=BLUE, lw=0.8, zorder=2)
            a.errorbar(i + wd / 2, p, yerr=[[p - plo], [phi - p]], color=INK, lw=0.7, capsize=1.6, zorder=3)
            a.text(i + wd / 2, phi + 0.03, f"{p:.2f}", ha="center", va="bottom", fontsize=6.0)
    a.axhline(1.0, color=GRAY, lw=0.8, ls="--", zorder=1)
    a.text(5.12, 1.03, "gap never closes", fontsize=6.0, va="bottom", ha="right", color=GRAY)
    a.set_xticks(x)
    a.set_xticklabels([r[0] for r in rel], fontsize=6.6)
    a.set_ylim(0, 1.58)
    a.set_yticks([0, 0.5, 1.0, 1.5])
    a.set_xlim(-0.6, 5.15)
    a.set_ylabel(r"plan leak $\beta$")
    from matplotlib.patches import Patch
    a.legend([Patch(fc=LRED, ec=RED), Patch(fc=LBLUE, ec=BLUE)], ["car ahead", "parked cars (control)"],
             loc="upper left", ncol=2, fontsize=6.3, handlelength=1.2, columnspacing=0.8,
             borderpad=0.3)
    a.set_title("(a) released world models (correct: 0)", fontsize=7.5, loc="left", pad=3)
    clean(a)
    # (b) object world models, same data and network: (label, seeds, leaky?)
    ctl = [("whole\nplan", [0.68, 0.62, 0.61], True), ("next\n0.5 s", [0.74, 0.72, 0.66], True),
           ("causal\n(past only)", [0.00, -0.01, 0.00], False), ("next 0.5 s,\nrepaired", [-0.02, -0.02, -0.02], False)]
    for i, (n, s, lk) in enumerate(ctl):
        m = float(np.mean(s))
        b.bar(i, m if abs(m) > 0.012 else 0.012 * np.sign(m or 1), 0.6, color=LRED if lk else LBLUE,
              ec=RED if lk else BLUE, lw=0.8, zorder=2)
        b.plot([i] * 3, s, "o", ms=2.2, color=INK, zorder=3)
        b.text(i, max(max(s), 0) + 0.04, f"{m:.2f}".replace("-0.00", "0.00").replace("-", "\u2212"), ha="center", va="bottom", fontsize=6.0)
    b.axhline(0.67, color=GRAY, lw=0.8, ls="--", zorder=1)
    b.text(3.45, 0.70, "leak present in\nthe driving logs", fontsize=6.0, ha="right", va="bottom", color=GRAY, linespacing=1.0)
    b.axhline(0, color=INK, lw=0.6)
    b.set_xticks(range(len(ctl)))
    b.set_xticklabels([c[0] for c in ctl], fontsize=6.6, linespacing=1.0)
    b.set_ylim(-0.1, 1.0)
    b.set_yticks([0, 0.5, 1.0])
    b.set_xlim(-0.55, 3.55)
    b.set_title("(b) one model, four ways to give it the plan", fontsize=7.5, loc="left", pad=3)
    clean(b)
    fig.savefig(OUT / "leak.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / "leak.png", bbox_inches="tight", pad_inches=0.01, dpi=220)


def _k(j):
    if j.startswith("step0"):
        return 0.0
    if j.startswith("chunk"):
        return 1.0
    if j.startswith("mix"):
        return float(j[3:])
    return np.nan


def cost():
    fig, axs = plt.subplots(1, 4, figsize=(5.5, 1.38), gridspec_kw=dict(width_ratios=[1.32, 1.0, 0.86, 0.86], wspace=0.62))
    # (a) policies trained inside the world model, real crash rate at equal progress (3 model x 3 policy seeds)
    a = axs[0]
    pol = [("true\nworld", 2.4, LGRAY, GRAY), ("causal", 2.2, LBLUE, BLUE), ("next\n0.5 s", 6.2, LRED, RED),
           ("re-\npaired", 2.2, LBLUE, BLUE)]                  # repaired: g-computation of the next-0.5 s model
    for i, (n, v, fc, ec) in enumerate(pol):
        a.bar(i, v, 0.66, color=fc, ec=ec, lw=0.8, zorder=2)
        a.text(i, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=6.0)
    a.set_xticks(range(4))
    a.set_xticklabels([p[0] for p in pol], fontsize=6.2, linespacing=0.95)
    a.set_ylim(0, 7.6)
    a.set_ylabel("real crash rate (%)", labelpad=1)
    a.set_title("(a) policies trained in each", fontsize=7.5, loc="left", pad=3)
    clean(a)
    # (b) phantom braking: share of states where a plan is judged / truly safe (whole-plan model, simulator)
    b = axs[1]
    d = json.load(open(OUT / "phantom_data.json"))["chunk"]
    idx = [d["cand"].index(-2.0), d["cand"].index(0.0)]
    for j, (key, fc, ec, lab) in enumerate((("true", LGRAY, GRAY, "truly safe"), ("judged", LRED, RED, "judged safe"))):
        v = [d[key][i] for i in idx]
        xs = np.arange(2) + (j - 0.5) * 0.36
        b.bar(xs, v, 0.36, color=fc, ec=ec, lw=0.8, zorder=2, label=lab)
        for xx, vv in zip(xs, v):
            b.text(xx, vv + 0.02, f"{100 * vv:.0f}", ha="center", va="bottom", fontsize=5.8)
    b.set_xticks([0, 1])
    b.set_xticklabels(["brake", "cruise"], fontsize=6.6)
    b.set_ylim(0, 1.5)
    b.set_yticks([0, 0.5, 1.0])
    b.set_yticklabels(["0", "50", "100"])
    b.set_ylabel("plan is safe (% of states)", labelpad=1)
    b.legend(loc="upper right", fontsize=5.6, handlelength=1.0, borderpad=0.25, labelspacing=0.15, ncol=1, bbox_to_anchor=(1.0, 1.01))
    b.set_title("(b) judged vs truly safe", fontsize=7.5, loc="left", pad=3)
    clean(b)
    # (c) simulator: hard brakes per episode vs leak dose k, 3 model seeds, 1 s headway
    c = axs[2]
    curves = []
    for f in (R / "a40" / "a40_hw1.0.csv", R / "a41" / "sim_s1_hw1.0.csv", R / "a41" / "sim_s2_hw1.0.csv"):
        dd = pd.read_csv(f)
        dd = dd[dd.judge != "logged_driver"]
        dd["k"] = dd.judge.map(lambda j: 0.0 if j.startswith("step0") else float(j[3:]))
        g = dd.groupby("k").hard_brakes.mean()
        curves.append(g[g.index.isin([0.0, 0.25, 0.5, 0.75, 1.0])])
    C = pd.concat(curves, axis=1)
    c.fill_between(C.index, C.min(1), C.max(1), color=LRED, alpha=0.6, lw=0)
    c.plot(C.index, C.mean(1), "-o", color=RED, lw=1.4, ms=3)
    c.set_xlabel("leak dose $k$", labelpad=1)
    c.set_ylabel("hard brakes per episode", labelpad=1)
    c.set_xticks([0, 0.5, 1.0])
    c.set_ylim(0, None)
    c.set_yticks([0, 5, 10])
    c.set_title("(c) closed loop: sim.", fontsize=7.5, loc="left", pad=3)
    clean(c, "both")
    # (d) real nuScenes traffic, log replay: near-miss steps per episode vs k, 0.5 s headway, 3 model seeds
    e = axs[3]
    rows = []
    dd = pd.read_csv(R / "a41" / "real_hw0.5.csv")
    dd = dd[dd.judge != "logged_driver"]
    dd["k"] = dd.judge.map(lambda j: 0.0 if j.startswith("step0") else float(j[3:]))
    rows.append(dd.groupby("k").ttc_viol.mean())
    for s_ in (1, 2):
        dd = pd.read_csv(R / "a42" / f"s{s_}_hw0.5.csv")
        dd = dd[dd.judge != "logged_driver"]
        dd["k"] = dd.judge.map(_k)
        rows.append(dd.dropna(subset=["k"]).groupby("k").ttc_viol.mean())
    rows = [r_[r_.index.isin([0.0, 1.0, 3.0])] for r_ in rows]
    for r_ in rows:                                    # thin: model seeds; bold: their mean
        e.plot(r_.index, r_.values, "-o", color=RED, lw=0.8, ms=2, alpha=0.45)
    m_ = pd.concat(rows, axis=1).mean(1)
    e.plot(m_.index, m_.values, "-o", color=RED, lw=1.4, ms=3)
    o = pd.read_csv(R / "a39" / "hw0.5_g3.csv")
    e.axhline(o[o.judge == "oracle"].ttc_viol.mean(), color=GRAY, lw=0.8, ls="--")
    e.text(3.0, o[o.judge == "oracle"].ttc_viol.mean() - 0.07, "true-effect\nplanner", fontsize=5.6, color=GRAY, ha="right", va="top",
           linespacing=0.95)
    e.set_xticks([0, 1, 3])
    e.set_xlabel("leak dose $k$", labelpad=1)
    e.set_ylabel("near-misses per episode", labelpad=1)
    e.set_ylim(0, None)
    e.set_title("(d) real traffic", fontsize=7.5, loc="left", pad=3)
    clean(e, "both")
    fig.savefig(OUT / "cost.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / "cost.png", bbox_inches="tight", pad_inches=0.01, dpi=220)


if __name__ == "__main__":
    leak()
    cost()
