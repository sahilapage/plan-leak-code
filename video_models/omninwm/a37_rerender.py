"""A37 training-free counterfactual video (descriptive / media): render the plan-marginal world (OmniNWM's own
generation under the past-only plan pm-1.5: RGB + metric depth, whose agents do not depend on the candidate plan) from
the candidate plan's camera poses (forward z-buffer splatting of every front-camera pixel, footprint = magnification d_q / z_c, 1-16 px:
ego_t(pm) -> t0 -> ego_t(cand);
sky at 1 km; remaining holes inpainted, cv2.INPAINT_TELEA). Same factorisation as the plan-marginal judge: agents from a
world that does not see the candidate, ego motion from the candidate. Panels: OmniNWM conditioned on the candidate |
the re-rendered plan-marginal world | bird's-eye strip with the true lead and the candidate ego box.
Usage: python src/a37_rerender.py <windows.json> <out_dir>/<token> <suffix> <cand plan> <media_dir> [prior=pm-1.5]"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_score as S  # noqa: E402

V = 1  # front camera (front3 index 1 = top-middle tile of rgb.mp4)


def render(rgb, depth, seg, Kq, Cq, Mq, Kc, Cc, Mc):
    """re-render one frame of the prior world (rgb/depth in camera q) into camera c; returns image, hole mask."""
    H, W = depth.shape
    d = depth.astype(np.float64).copy()
    d[(seg == 2) | (d <= 0.3)] = 1000.0
    vv, uu = np.mgrid[0:H, 0:W]
    pc = np.linalg.inv(Kq) @ np.stack([uu.ravel(), vv.ravel(), np.ones(H * W)]) * d.ravel()
    P = Cq @ np.vstack([pc, np.ones(H * W)])                          # ego_t(q)
    P = Mq @ P                                                        # t0
    P = np.linalg.inv(Mc) @ P                                         # ego_t(c)
    P = np.linalg.inv(Cc) @ P                                         # camera c
    z = P[2]
    ok = z > 0.2
    uv = (Kc @ P[:3, ok])[:2] / z[ok]
    col = rgb.reshape(-1, 3)[ok]
    zz = z[ok]
    out = np.zeros((H, W, 3), np.uint8)
    zb = np.full((H, W), np.inf)
    mag = np.clip(np.ceil(1.15 * d.ravel()[ok] / zz), 1, 16).astype(int)
    order = np.argsort(-zz)                                           # far first, near overwrite
    u0, v0, c0, z0, s0 = (np.floor(uv[0, order]).astype(int), np.floor(uv[1, order]).astype(int), col[order],
                          zz[order], mag[order])
    for sz in np.unique(s0):
        sel = s0 == sz
        for du in range(-(sz // 2), sz - sz // 2):
            for dv in range(-(sz // 2), sz - sz // 2):
                u, v = u0[sel] + du, v0[sel] + dv
                m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                u, v, c, zo = u[m], v[m], c0[sel][m], z0[sel][m]
                near = zo < zb[v, u] - 1e-6
                # among duplicates the last (nearest, given the far-first order) wins
                out[v[near], u[near]] = c[near]
                zb[v[near], u[near]] = zo[near]
    hole = ~np.isfinite(zb)
    out = cv2.inpaint(out, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return out, hole


def main(windows, wdir, suf, cand, mdir, prior="pm-1.5"):
    wdir, mdir = Path(wdir), Path(mdir)
    mdir.mkdir(parents=True, exist_ok=True)
    tok = wdir.name
    w = {x["token"]: x for x in json.load(open(windows))}[tok]
    t = np.array(json.load(open(wdir / f"logged{suf}" / "reward.json"))["t"])
    G = {}
    for pl in (cand, prior):
        d = wdir / f"{pl}{suf}"
        G[pl] = {"z": np.load(d / "front3.npz"), "M": np.load(d / "rel_poses.npy"), "v": iio.imread(d / "rgb.mp4")}
    tr = np.array(w["lead_track_t0frame"])
    L, Wd = w["lead_LW"]
    lx = np.interp(t, np.arange(len(tr)) * .5, tr[:, 0])
    ly = np.interp(t, np.arange(len(tr)) * .5, tr[:, 1])
    Mc = G[cand]["M"]
    gap = lx - L / 2 - (Mc[:, 0, 3] + S.FRONT0)
    hit = np.nonzero(gap <= 0)[0]
    k_hit = hit[0] if len(hit) else None
    tmp = Path(tempfile.mkdtemp())
    holes = []
    for k in range(len(t)):
        q, c = G[prior], G[cand]
        img, hole = render(q["v"][k, :448, 800 * V:800 * (V + 1)], q["z"]["depth"][V, k], q["z"]["seg"][V, k],
                           q["z"]["cam_k"][V, k], q["z"]["cam2ego"][V, k], q["M"][k],
                           c["z"]["cam_k"][V, k], c["z"]["cam2ego"][V, k], Mc[k])
        holes.append(hole.mean())
        fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
        for j, (im, title, col) in enumerate(((c["v"][k, :448, 800 * V:800 * (V + 1)],
                                               f"OmniNWM, conditioned on the plan (+{w['a_star']:.2f} m/s$^2$ into the lead)",
                                               "#cf222e"),
                                              (img, "fix: plan-independent world, seen from the same plan's camera",
                                               "#8250df"))):
            ax = fig.add_axes([0.01 + 0.495 * j, 0.36, 0.48, 0.58])
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(3)
            ax.set_title(title, fontsize=10, loc="left", color="#1f2328")
            msg = f"t = {t[k]:.2f} s" + ("   TRUE CONTACT" if k_hit is not None and k >= k_hit else "")
            ax.text(0.01, 0.04, msg, transform=ax.transAxes, color="white", fontsize=11,
                    bbox=dict(facecolor="#cf222e" if "CONTACT" in msg else "black", alpha=0.6, lw=0))
        ax = fig.add_axes([0.07, 0.08, 0.86, 0.2])
        ax.set_xlim(-3, 30); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.add_patch(Rectangle((lx[k] - L / 2, ly[k] - Wd / 2), L, Wd, fc="#1a7f37", alpha=0.3, ec="#1a7f37", lw=2))
        ax.add_patch(Rectangle((Mc[k, 0, 3] + S.BOX[0] - S.BOX[1] / 2, Mc[k, 1, 3] - S.BOX[2] / 2), S.BOX[1], S.BOX[2],
                               fc="#cf222e", alpha=0.35, ec="#cf222e", lw=2))
        ax.text(lx[k], 1.9, "lead (real)", color="#1a7f37", ha="center", fontsize=9)
        ax.text(Mc[k, 0, 3] + S.BOX[0], -2.8, "ego (plan)", color="#cf222e", ha="center", fontsize=9)
        ax.set_xlabel("metres ahead of the ego at t0 (bird's-eye view, ground truth)", color="#6e7781")
        ax.tick_params(colors="#6e7781")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.savefig(tmp / f"{k:03d}.png")
        plt.close(fig)
        if k in (0, 16, len(t) - 1) or (k_hit is not None and k in (k_hit - 6, k_hit - 3, k_hit)):
            (mdir / f"{tok[:8]}_rerender_{cand}_f{k:02d}.png").write_bytes((tmp / f"{k:03d}.png").read_bytes())
    out = mdir / f"{tok[:8]}_{w['scene']}_rerender_{cand}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "6", "-i", str(tmp / "%03d.png"),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)], check=True)
    print("wrote", out, "| contact frame", k_hit, "| mean hole share %.3f, max %.3f" % (np.mean(holes), np.max(holes)))


if __name__ == "__main__":
    main(*sys.argv[1:])
