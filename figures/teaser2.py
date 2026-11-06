"""Fig. 1 (large). OmniNWM, scene-0780 (car ahead stopped at a red light), same seed, at t = 0, 1.25 s and 2.0 s (0.9 m
before the real contact). Row 1: logged plan. Row 2: a plan that accelerates into the stopped car. Row 3: the repair,
the same crashing plan's cameras looking into OmniNWM's world generated under a past-only plan (pm-1.5), re-rendered
with its own depth (src/a37_rerender.py; App. C). Green: the real car's annotated 3D box projected into the generated camera; arrow: the car OmniNWM
generates (nearest surface read from its own depth and semantics).
Also writes dist.pdf: distances ahead over time in scene-0780 (used next to the 3D occupancy figure).
Usage: python teaser2.py <windows_merged.json> <out_omni>"""
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from style import BLUE, GREEN, INK, RED, plt
from teaser import box_faces, project

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from a37_rerender import render  # noqa: E402

HERE = Path(__file__).resolve().parent
TR = json.load(open(HERE / "omni_tracks_all.json"))
MAIN = "ad26ae1c24e14afd9bc1e0bcc4e378b7"
MORE = (("6b3c851413064994bb0babae7e74efad", "scene-0097: stopped car"),
        ("8d67c4024af6470faada45dc41ba9a16", "scene-0556: moving car"),
        ("918b8f22899b460abb6903efe15d663f", "scene-1070: moving car, night"))
Y0, Y1, X0, X1 = 0.33, 0.97, 120, 680           # crop of the 448 x 800 front view


def contact(tr):
    real = np.array(tr["real_rear"])
    return int(np.nonzero(real - np.array(tr["plans"]["collide_s0_n25"]["ego_front"]) <= 0)[0][0])


def gen_at(tr, pl, k):
    g = np.array([np.nan if v is None else v for v in tr["plans"][pl]["gen"]])
    return float(np.nanmedian(g[max(k - 1, 0):k + 2]))


def repaired(d, k, cand="collide_s0_n25", prior="pm-1.5_s0_n25"):
    """Front view of the past-only world (prior), re-rendered from the candidate plan's camera at frame k."""
    q = {"z": np.load(d / prior / "front3.npz"), "M": np.load(d / prior / "rel_poses.npy"), "v": iio.imread(d / prior / "rgb.mp4")}
    c = {"z": np.load(d / cand / "front3.npz"), "M": np.load(d / cand / "rel_poses.npy")}
    with np.errstate(all="ignore"):
        img, _ = render(q["v"][k, :448, 800:1600], q["z"]["depth"][1, k], q["z"]["seg"][1, k], q["z"]["cam_k"][1, k],
                        q["z"]["cam2ego"][1, k], q["M"][k], c["z"]["cam_k"][1, k], c["z"]["cam2ego"][1, k], c["M"][k])
    return img


def frame(ax, w, tr, d, pl, k, col, arrow=False, gap_label=True, real_label=True, img=None):
    vid = iio.imread(d / pl / "rgb.mp4")
    z = np.load(d / pl / "front3.npz")
    M = np.load(d / pl / "rel_poses.npy")
    H, W = vid.shape[1] // 2, vid.shape[2] // 3
    y0, y1 = int(Y0 * H), int(Y1 * H)
    im = vid[k][:H, W:2 * W] if img is None else img
    ax.imshow(im[y0:y1], extent=(0, W, y1, y0), aspect="auto")
    t = np.array(tr["t"])
    lt = np.array(w["lead_track_t0frame"])
    cx = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 0])
    cy = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 1])
    C2E, K = z["cam2ego"][1, k].astype(np.float64), z["cam_k"][1, k].astype(np.float64)
    _, rear = box_faces(cx, cy, *w["lead_LW"])
    uv, dep = project(rear, M[k], C2E, K)
    if (dep > 0.3).all():
        ax.add_patch(Polygon(uv, closed=True, fc=GREEN, alpha=0.17, lw=0))
        ax.add_patch(Polygon(uv, closed=True, fill=False, ec=GREEN, lw=1.6))
    if arrow:
        g = gen_at(tr, pl, k)
        p, _ = project(np.array([[g + 0.3, cy, 1.0]]), M[k], C2E, K)
        u, v = p[0]
        ax.annotate("OmniNWM's car", xy=(u, v - 20), xytext=(X1 - 12, y1 - 22), fontsize=7.5, color="white",
                    ha="right", va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2, mutation_scale=8),
                    bbox=dict(fc=RED, ec="none", alpha=0.85, pad=1.2))
        if real_label:
            ax.text(X1 - 8, y0 + 46, "real car (green box)", fontsize=7.5, color="white", ha="right",
                    va="top", fontweight="bold", bbox=dict(fc=GREEN, ec="none", alpha=0.85, pad=1.2))
    if gap_label:
        e = np.array(tr["plans"][pl]["ego_front"])[k]
        gap = np.array(tr["real_rear"])[k] - e
        ax.text(X0 + 8, y0 + 10, (f"real gap {gap:.1f} m" if gap < 3 else f"real gap {gap:.0f} m") if gap > 0.05 else "real gap 0 m: crash", fontsize=7.2,
                color="white", va="top", bbox=dict(fc="black", alpha=0.6, lw=0, pad=1.0))
    ax.set_xlim(X0, X1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(col)
        sp.set_linewidth(2.0)


def teaser(windows, out_omni):
    W = {x["token"]: x for x in json.load(open(windows))}
    tr = TR[MAIN]
    t = np.array(tr["t"])
    k_star = contact(tr)
    ks = (0, 15, k_star - 3)                       # last column: 0.9 m before the real contact (re-render still clean)
    fig = plt.figure(figsize=(5.5, 3.3))
    gs = fig.add_gridspec(3, 3, wspace=0.03, hspace=0.07, left=0.05, right=1.0, top=0.95, bottom=0.0)
    d = Path(out_omni) / MAIN
    rows = (("logged_s0_n25", "logged plan\n(the driver waits)", "#5f5f5f", None),
            ("collide_s0_n25", "crashing plan\n(OmniNWM)", RED, None),
            ("collide_s0_n25", "crashing plan,\nrepaired (ours)", BLUE, repaired))
    for r, (pl, lab, col, fix) in enumerate(rows):
        for c, k in enumerate(ks):
            ax = fig.add_subplot(gs[r, c])
            frame(ax, W[MAIN], tr, d, pl, k, col, arrow=(r == 1 and c == 2), img=fix(d, k) if fix else None)
            if r == 0:
                ax.set_title(f"$t$ = {t[k]:.1f} s" + (f"  ({t[k_star] - t[k]:.2f} s before contact)" if c == 2 else ""), fontsize=8.5, pad=2.5)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8, color=col, labelpad=3, linespacing=1.05)
    fig.savefig(HERE / "teaser2.pdf", bbox_inches="tight", pad_inches=0.01, dpi=230)
    fig.savefig(HERE / "teaser2.png", bbox_inches="tight", pad_inches=0.01, dpi=160)


def dist():
    tr = json.load(open(HERE / "omni_tracks_cal.json"))[MAIN]      # OmniNWM depth corrected for its bias (App. A)
    t = np.array(tr["t"])
    real = np.array(tr["real_rear"])
    k_star = contact(tr)
    fig, ax = plt.subplots(figsize=(1.75, 1.95))
    for pl, col in (("logged_s0_n25", BLUE), ("collide_s0_n25", RED)):
        g = np.array([np.nan if v is None else v for v in tr["plans"][pl]["gen"]])
        ax.plot(t, tr["plans"][pl]["ego_front"], color=col, lw=1.6)
        ax.plot(t, g, color=col, lw=1.2, ls=(0, (3, 1.5)))
    ax.plot(t, real, color=GREEN, lw=2.2)
    ax.plot(t[k_star], real[k_star], marker="X", ms=6.5, color=INK, mec="white", mew=0.6, zorder=5)
    ax.text(t[k_star], real[k_star] - 1.4, "crash", fontsize=6.5, ha="center", va="top")
    kw = dict(fontsize=6.2, va="center", bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))   # direct labels
    ax.text(0.08, 2.4, "ego, waits", color=BLUE, **kw)
    ax.text(0.08, 5.4, "ego, drives in", color=RED, **kw)
    ax.text(0.08, 7.4, "real car (stopped)", color=GREEN, **kw)
    ax.text(0.08, 12.4, "OmniNWM's car:", color=INK, **kw)
    ax.text(0.08, 11.2, "dashed, one per plan", color=INK, **kw)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(1.6, 19)
    ax.set_yticks([5, 10, 15])
    ax.set_xlabel("time (s)", labelpad=1)
    ax.set_ylabel("position along the road (m)", labelpad=1)
    ax.grid(True, axis="both")
    fig.savefig(HERE / "dist.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / "dist.png", bbox_inches="tight", pad_inches=0.01, dpi=200)


if __name__ == "__main__":
    teaser(*sys.argv[1:3])
    dist()
