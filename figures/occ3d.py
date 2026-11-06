"""3D semantic occupancy of OmniNWM's generated world (its own metric depth + semantics, three front cameras, voxelised
at VOX m) at the counterfactual plan's real contact frame, logged vs counterfactual plan, in the t0 ego frame. Green
wireframe: the real car's annotated box; blue / red box: the ego under each plan.
Usage: python occ3d.py <windows.json> <out_omni> <token> <out.pdf> [frame]"""
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio  # noqa: F401  (kept for parity with teaser.py)
import numpy as np

from style import BLUE, GREEN, INK, RED, plt

VOX = 0.32
X0, X1, Y0, Y1, Z0, Z1 = -1.0, float(__import__("os").environ.get("OCC_X1", 23.0)), -6.5, 6.5, -0.35, 3.4
STRIDE = 2
# OmniNWM semantic ids: 0 structure, 1 road, 2 sky, 3 vehicle, 4 person, 6 lane line, 7 sidewalk/terrain, 8 vegetation
COL = {0: "#ddd3c2", 1: "#a7a7a7", 3: "#d9b8cf", 4: "#d9b8cf", 5: "#d9b8cf", 6: "#f4f4f4", 7: "#c3dbb5", 8: "#a9cf97",
       9: "#b0127a"}   # 9: the lead car (vehicle voxels in a 1.4 m corridor around the real lead)
FRONT = 1.3 + 4.1 / 2
DEPTH_A, DEPTH_B = 1.09, 0.90           # OmniNWM depth reads A * true + B against annotated cars (App. A); undone


def world_points(d, k):
    z = np.load(d / "front3.npz")
    M = np.load(d / "rel_poses.npy")
    P, S = [], []
    for v in range(3):
        dep = (z["depth"][v, k, ::STRIDE, ::STRIDE].astype(np.float64) - DEPTH_B) / DEPTH_A
        seg = z["seg"][v, k, ::STRIDE, ::STRIDE]
        m = (dep > 0.5) & (dep < 40) & (seg != 2)
        yy, xx = np.nonzero(m)
        uv1 = np.stack([xx * STRIDE, yy * STRIDE, np.ones_like(xx)]).astype(np.float64)
        pc = np.linalg.solve(z["cam_k"][v, k].astype(np.float64), uv1) * dep[yy, xx]
        pe = z["cam2ego"][v, k].astype(np.float64) @ np.vstack([pc, np.ones((1, pc.shape[1]))])
        P.append((M[k] @ pe)[:3].T)
        S.append(seg[yy, xx])
    return np.concatenate(P), np.concatenate(S), M[k]


def voxelise(P, S):
    nx, ny, nz = (int(round((b - a) / VOX)) for a, b in ((X0, X1), (Y0, Y1), (Z0, Z1)))
    ijk = np.floor((P - [X0, Y0, Z0]) / VOX).astype(int)
    ok = ((ijk >= 0) & (ijk < [nx, ny, nz])).all(1)
    ijk, S = ijk[ok], S[ok]
    lab = np.full((nx, ny, nz), -1, int)
    cnt = np.zeros((nx, ny, nz, 9), int)
    np.add.at(cnt, (ijk[:, 0], ijk[:, 1], ijk[:, 2], S), 1)
    tot = cnt.sum(-1)
    veh = cnt[..., 3:6].sum(-1)
    lab[tot >= 2] = cnt.argmax(-1)[tot >= 2]
    lab[(veh >= 2) & (veh >= 0.3 * tot)] = 3          # vehicles win ties: they are what the figure is about
    return lab


def mark_lead(lab, y_lead):
    yc = Y0 + (np.arange(lab.shape[1]) + 0.5) * VOX
    corr = np.abs(yc - y_lead) <= 1.4
    lab[:, corr][...] = lab[:, corr]
    sub = lab[:, corr]
    sub[sub == 3] = 9
    lab[:, corr] = sub
    # keep only the vehicle component in the corridor nearest to the ego (the lead); others stay "other vehicle"
    from scipy import ndimage
    comp, n = ndimage.label(lab == 9, structure=np.ones((3, 3, 3)))
    best, bx = 0, 1e9
    for c in range(1, n + 1):
        ii = np.nonzero(comp == c)[0]
        if len(ii) >= 12 and ii.min() < bx:
            best, bx = c, ii.min()
    lab[(lab == 9) & (comp != best)] = 3
    return lab


def solid(ax, x0, x1, y0, y1, z0, z1, col, alpha):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    F = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([c[f] for f in F], facecolor=col, alpha=alpha, edgecolor="none"))


def wire(ax, x0, x1, y0, y1, z0, z1, col, lw):
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        for zz in (z0, z1):
            ax.plot([a[0], b[0]], [a[1], b[1]], [zz, zz], color=col, lw=lw, zorder=10)
    for x in (x0, x1):
        for y in (y0, y1):
            ax.plot([x, x], [y, y], [z0, z1], color=col, lw=lw, zorder=10)


def panel(ax, lab, M, lead, ego_col, title):
    fill = lab >= 0
    fc = np.empty(lab.shape, object)
    for c, h in COL.items():
        fc[lab == c] = h
    fill &= fc != None  # noqa: E711
    xg, yg, zg = np.meshgrid(*(np.arange(n + 1) * VOX + o for n, o in zip(lab.shape, (X0, Y0, Z0))), indexing="ij")
    ax.voxels(xg, yg, zg, fill, facecolors=fc, edgecolors="#00000018", linewidth=0.1, shade=True)
    ex, ey = M[0, 3] + 1.3, M[1, 3]
    solid(ax, ex - 2.05, ex + 2.05, ey - 0.9, ey + 0.9, 0, 1.5, ego_col, 0.55)
    wire(ax, ex - 2.05, ex + 2.05, ey - 0.9, ey + 0.9, 0, 1.5, ego_col, 1.4)
    cx, cy, L, Wd = lead
    solid(ax, cx - L / 2, cx + L / 2, cy - Wd / 2, cy + Wd / 2, 0, 1.6, GREEN, 0.18)
    wire(ax, cx - L / 2, cx + L / 2, cy - Wd / 2, cy + Wd / 2, 0, 1.6, GREEN, 1.6)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_zlim(Z0, Z1 + 2)
    ax.set_box_aspect((X1 - X0, Y1 - Y0, Z1 - Z0 + 2), zoom=1.0)
    ax.view_init(elev=38, azim=-168)
    ax.set_axis_off()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main(windows, out_omni, tok, out, frame=None):
    w = {x["token"]: x for x in json.load(open(windows))}[tok]
    d = Path(out_omni) / tok
    t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
    tr = np.array(w["lead_track_t0frame"])
    L, Wd = w["lead_LW"]
    Mc = np.load(d / "collide_s0_n25" / "rel_poses.npy")
    x_rear = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 0]) - L / 2
    k = int(frame) if frame else int(np.nonzero(x_rear - (Mc[:, 0, 3] + FRONT) <= 0)[0][0])
    lead = (np.interp(t[k], np.arange(len(tr)) * 0.5, tr[:, 0]), np.interp(t[k], np.arange(len(tr)) * 0.5, tr[:, 1]), L, Wd)
    import io
    from PIL import Image
    imgs = []
    for pl, col in (("logged_s0_n25", BLUE), ("collide_s0_n25", RED)):
        P, S, M = world_points(d / pl, k)
        f3 = plt.figure(figsize=(5, 4))
        ax = f3.add_subplot(111, projection="3d")
        panel(ax, mark_lead(voxelise(P, S), lead[1]), M, lead, col, "")
        f3.subplots_adjust(0, 0, 1, 1)
        buf = io.BytesIO()
        f3.savefig(buf, dpi=350, format="png")
        plt.close(f3)
        im = np.asarray(Image.open(buf).convert("RGB"))
        nz = np.nonzero((im < 250).any(-1))
        imgs.append(im[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1])
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw=dict(wspace=0.02))
    for ax, im, ttl, col in zip(axs, imgs, (os.environ.get("OCC_T0", "logged plan: the ego waits"), os.environ.get("OCC_T1", "crashing plan: the ego drives into the car")),
                                (BLUE, RED)):
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(ttl, fontsize=10, color=col, pad=2)
    from matplotlib.patches import Patch
    hs = [Patch(fc=BLUE, ec=BLUE, alpha=0.7), Patch(fc=RED, ec=RED, alpha=0.7), Patch(fc="#d6ecdc", ec=GREEN, lw=1.2),
          Patch(fc=COL[9]), Patch(fc=COL[3]), Patch(fc=COL[1])]
    fig.legend(hs, ["ego, waiting", "ego, driving into the car", "real car (annotated box)", "car ahead, as OmniNWM generates it",
                    "other generated cars", "generated road"], loc="lower center", ncol=3, fontsize=8.6,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.3, columnspacing=1.2, borderpad=0.4)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.235)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.01, dpi=220)
    print("frame", k, "t", round(float(t[k]), 2))


if __name__ == "__main__":
    main(*sys.argv[1:6])
