"""Fig. 1. OmniNWM, one held-out nuScenes scene (scene-0780: the lead car is stopped at a red light), same seed, two
plans. Left: generated front camera at three times under the logged plan (the driver waited) and under a counterfactual
plan that accelerates into the stopped car; the green box is the real car's annotated 3D box projected into the camera.
Right: distance ahead over time for the ego front, the real car and the car OmniNWM generates under each plan.
Usage: python teaser.py <windows.json> <out_omni> <omni_tracks.json> <out.pdf>"""
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from matplotlib.patches import Polygon

from style import BLUE, GREEN, INK, RED, plt

TOK = "ad26ae1c24e14afd9bc1e0bcc4e378b7"
FRONT = 1.3 + 4.1 / 2
CROP = (0.28, 0.98)          # vertical crop of the 448 x 800 front view (drop the sky)
XCROP = (110, 690)           # horizontal crop (centre the lane)


def box_faces(cx, cy, L, Wd, Hh=1.5):
    xs, ys = (cx - L / 2, cx + L / 2), (cy - Wd / 2, cy + Wd / 2)
    c = np.array([[x, y, z] for x in xs for y in ys for z in (0.0, Hh)])          # 8 corners, t0 frame
    rear = c[[0, 1, 3, 2]]                                                           # x = rear face
    return c, rear


def project(P_t0, M_k, cam2ego, K):
    """t0-frame points -> pixel (u, v) and depth in the front camera at frame k."""
    Ph = np.c_[P_t0, np.ones(len(P_t0))]
    pc = (np.linalg.inv(cam2ego) @ np.linalg.inv(M_k) @ Ph.T)[:3]
    uv = (K @ pc)[:2] / np.maximum(pc[2], 1e-3)
    return uv.T, pc[2]


def main(windows, out_omni, tracks, out):
    w = {x["token"]: x for x in json.load(open(windows))}[TOK]
    tr = json.load(open(tracks))[TOK]
    d = Path(out_omni) / TOK
    t = np.array(tr["t"])
    real = np.array(tr["real_rear"])
    ego_c = np.array(tr["plans"]["collide_s0_n25"]["ego_front"])
    k_star = int(np.nonzero(real - ego_c <= 0)[0][0])
    ks = (0, 15, k_star)
    L, Wd = w["lead_LW"]
    lt = np.array(w["lead_track_t0frame"])

    fig = plt.figure(figsize=(5.5, 2.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.035, hspace=0.05,
                          left=0.035, right=0.995, top=0.90, bottom=0.14)
    plans = (("logged_s0_n25", "logged plan\n(waits)", BLUE), ("collide_s0_n25", "counter-\nfactual plan\n(accelerates)", RED))
    for r, (pl, lab, col) in enumerate(plans):
        vid = iio.imread(d / pl / "rgb.mp4")
        z = np.load(d / pl / "front3.npz")
        M = np.load(d / pl / "rel_poses.npy")
        H, W = vid.shape[1] // 2, vid.shape[2] // 3
        y0, y1 = int(CROP[0] * H), int(CROP[1] * H)
        for c, k in enumerate(ks):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(vid[k][:H, W:2 * W][y0:y1], extent=(0, W, y1, y0), aspect="auto")
            cx = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 0])
            cy = np.interp(t[k], np.arange(len(lt)) * 0.5, lt[:, 1])
            _, rear = box_faces(cx, cy, L, Wd)
            uv, dep = project(rear, M[k], z["cam2ego"][1, k].astype(np.float64), z["cam_k"][1, k].astype(np.float64))
            if (dep > 0.3).all():
                ax.add_patch(Polygon(uv, closed=True, fill=True, fc=GREEN, alpha=0.18, lw=0))
                ax.add_patch(Polygon(uv, closed=True, fill=False, ec=GREEN, lw=1.3))
            ax.set_xlim(*XCROP)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(col)
                sp.set_linewidth(1.6)
            if r == 0:
                ttl = f"$t$ = {t[k]:.1f} s" if k != k_star else f"$t$ = {t[k]:.1f} s (contact)"
                ax.set_title(ttl, fontsize=7.5, pad=2)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7.5, color=col, labelpad=2)
            e_k = np.array(tr["plans"][pl]["ego_front"])[k]
            gap = real[k] - e_k
            ax.text(0.03, 0.05, f"real gap {gap:.0f} m" if gap > 0.05 else "real gap 0 m: crash",
                    transform=ax.transAxes, fontsize=6.3, color="white",
                    bbox=dict(fc="black", alpha=0.6, lw=0, pad=1.0))

    ax = fig.add_subplot(gs[:, 3])
    for pl, col, lab in (("logged_s0_n25", BLUE, "logged"), ("collide_s0_n25", RED, "counterfactual")):
        e = np.array(tr["plans"][pl]["ego_front"])
        g = np.array([np.nan if v is None else v for v in tr["plans"][pl]["gen"]])
        ax.plot(t, e, color=col, lw=1.6)
        ax.plot(t, g, color=col, lw=1.3, ls=(0, (3, 1.6)), marker="o", ms=1.8, markevery=3)
    ax.plot(t, real, color=GREEN, lw=2.2, solid_capstyle="butt")
    ax.plot(t[k_star], real[k_star], marker="X", ms=7, color=INK, mec="white", mew=0.6, zorder=5)
    ax.annotate("", xy=(t[-1] + 0.03, np.nanmax([v for v in tr["plans"]["collide_s0_n25"]["gen"][-4:] if v])),
                xytext=(t[-1] + 0.03, ego_c[-1]), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=INK, lw=1.4), Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 1.6))),
          Line2D([], [], color=GREEN, lw=2.2), Line2D([], [], color=BLUE, lw=3), Line2D([], [], color=RED, lw=3)]
    ax.legend(hs, ["ego front", "car in OmniNWM", "real car (stopped)", "logged plan", "counterfactual"], loc="upper left", fontsize=6.3,
              handlelength=1.8, borderpad=0.35, labelspacing=0.25)
    ax.text(t[k_star] + 0.02, real[k_star] - 1.9, "real\ncrash", fontsize=6.3, ha="center", va="top", color=INK,
            linespacing=1.0)
    ax.text(t[-1] - 0.05, 14.2, "model's\ngap", fontsize=6.3, ha="right", va="center", color=INK)
    ax.set_xlim(0, t[-1] + 0.12)
    ax.set_ylim(2, 25)
    ax.set_xlabel("time (s)", labelpad=1)
    ax.set_ylabel("distance ahead (m)", labelpad=1)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title("where the cars are", fontsize=7.5, pad=2)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.01, dpi=220)
    print("contact frame", k_star, "t", t[k_star])


if __name__ == "__main__":
    main(*sys.argv[1:5])
