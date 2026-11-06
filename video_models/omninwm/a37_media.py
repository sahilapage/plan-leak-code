"""A37 media (local, CPU): per window, a side-by-side video of OmniNWM under the logged plan vs the collide plan
(front camera, generated) over a bird's-eye view in the t0 frame: the ego box of each plan, the TRUE lead box
(annotation) and the generated lead (tracked vehicle component from OmniNWM's own depth + semantics), plus the verdict
of OmniNWM's own released collision rule. Also a 4-panel keyframe PNG for the paper.
With "fix": a third panel shows the past-only plan's world (pm-1.5) and the BEV adds its lead and the verdict of the
plan-marginal judge (src/a37_score.py, pm) on the counterfactual plan.
Usage: python src/a37_media.py <windows.json> <out_dir>/<token> <suffix e.g. _s0_n25> <media_out_dir> [fix]"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_score as S  # noqa: E402

INK, MUTED = "#1f2328", "#6e7781"
C_LOG, C_COL, C_TRUE, C_GEN, C_PM = "#0969da", "#cf222e", "#1a7f37", "#bc4c00", "#8250df"


def frames(mp4, n):
    """front camera (top-middle tile of the 6-view panorama) frames as uint8 arrays."""
    import imageio.v3 as iio
    v = iio.imread(mp4)
    H, W = v.shape[1] // 2, v.shape[2] // 3
    return [v[k, :H, W:2 * W] for k in range(min(n, len(v)))]


def main(windows, wdir, suf, mdir, fix=None):
    wdir, mdir = Path(wdir), Path(mdir)
    mdir.mkdir(parents=True, exist_ok=True)
    tok = wdir.name
    w = {x["token"]: x for x in json.load(open(windows))}[tok]
    t = np.array(json.load(open(wdir / f"logged{suf}" / "reward.json"))["t"])
    tr = np.array(w["lead_track_t0frame"])
    L, Wd = w["lead_LW"]
    lx = np.interp(t, np.arange(len(tr)) * .5, tr[:, 0])
    ly = np.interp(t, np.arange(len(tr)) * .5, tr[:, 1])
    D = {}
    for pl in ("logged", "collide") + (S.PRIORS if fix else ()):
        d = wdir / f"{pl}{suf}"
        z = np.load(d / "front3.npz")
        M = np.load(d / "rel_poses.npy")
        dyn = S.backproject(z["depth"], z["seg"], z["cam_k"], z["cam2ego"], S.DYN)
        dyn_rel = S.backproject(z["depth"], z["seg"], z["cam_k"], z["cam2ego"], S.DYN, dmin=1e-6)
        rel = S.released_flags(dyn_rel, M)
        D[pl] = {"M": M, "dyn": dyn, "rear": S.track_lead(dyn, M, w), "img": frames(d / "rgb.mp4", len(t)),
                 "rel_any": bool(rel.any()), "self_any": any(S.in_box(P).sum() >= S.NMIN for P in dyn),
                 "rew": json.load(open(d / "reward.json"))["reward_released"]}
    if fix:
        Mc = D["collide"]["M"]
        share = np.zeros(len(t))
        for q in S.PRIORS:
            Q = D[q]
            share += np.array([S.in_box(S.from_t0(S.to_t0(Q["dyn"][k], Q["M"][k]), Mc[k])).sum() >= S.NMIN
                               for k in range(len(t))]) / len(S.PRIORS)
        pm_hit = share >= 0.5
    rs = [np.asarray(D[q]["rear"], float) for q in ("logged", "collide") + (("pm-1.5",) if fix else ())]
    xmax = max(30.0, np.nanmax(np.concatenate(rs + [np.asarray(lx, float) + L / 2])) + 8)   # keep every lead in view
    tmp = Path(tempfile.mkdtemp())
    for k in range(len(t)):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
        panels = [("logged", C_LOG, "logged plan (what the driver did)"),
                  ("collide", C_COL, f"counterfactual: +{w['a_star']:.2f} m/s$^2$ into the lead")]
        if fix:
            panels.append(("pm-1.5", C_PM, "fix: past-only plan (no future leak)"))
        wpan = 0.96 / len(panels)
        for j, (pl, col, lab) in enumerate(panels):
            ax = fig.add_axes([0.02 + wpan * j, 0.47, wpan - 0.02, 0.47])
            ax.imshow(D[pl]["img"][k])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(3)
            ax.set_title(f"OmniNWM, {lab}", color=INK, fontsize=11 if not fix else 9, loc="left")
            ax.text(0.01, 0.03, f"t = {t[k]:.2f} s", transform=ax.transAxes, color="white", fontsize=11,
                    bbox=dict(facecolor="black", alpha=0.5, lw=0))
        ax = fig.add_axes([0.07, 0.07, 0.86, 0.33])
        ax.set_xlim(-3, xmax); ax.set_ylim(-4, 4); ax.set_aspect("equal")
        ax.add_patch(Rectangle((lx[k] - L / 2, ly[k] - Wd / 2), L, Wd, fc=C_TRUE, alpha=0.25, ec=C_TRUE, lw=2))
        ax.text(lx[k], ly[k] + Wd / 2 + 0.3, "lead: real position", color=C_TRUE, ha="center", fontsize=9)
        for pl, col in (("logged", C_LOG), ("collide", C_COL)):
            M = D[pl]["M"]
            ax.add_patch(Rectangle((M[k, 0, 3] + S.BOX[0] - S.BOX[1] / 2, M[k, 1, 3] - S.BOX[2] / 2), S.BOX[1], S.BOX[2],
                                   fc=col, alpha=0.35, ec=col, lw=2))
            r = D[pl]["rear"][k]
            if np.isfinite(r):
                ax.plot([r, r], [-2.5, 2.5], color=col, lw=2.5, ls="--")
                ax.text(r + 0.2, 2.7 if pl == "collide" else -3.2, f"lead rear as generated ({pl})", color=col, fontsize=8)
        if fix:
            r = D["pm-1.5"]["rear"][k]
            if np.isfinite(r):
                ax.plot([r, r], [-2.5, 2.5], color=C_PM, lw=2.5, ls=":")
                ax.text(r + 0.2, -3.8, "lead rear, past-only world", color=C_PM, fontsize=8)
        ax.set_xlabel("metres ahead of the ego at t0", color=MUTED)
        ax.tick_params(colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        f = lambda b: "crash" if b else "no crash"  # noqa: E731
        ttl = (f"bird's-eye view (t0 frame). Truth: counterfactual CRASH | judge on its own world: "
               f"{f(D['collide']['self_any'])}")
        if fix:
            ttl += f" | fix (plan-marginal judge): {'CRASH' if pm_hit[:k + 1].any() else 'no crash yet'}"
        else:
            ttl = ttl.replace("Truth:", "Truth: logged no crash,").replace("own world: ",
                              f"own world: logged {f(D['logged']['self_any'])}, counterfactual ")
        ax.set_title(ttl, color=INK, fontsize=10, loc="left")
        fig.savefig(tmp / f"{k:03d}.png")
        plt.close(fig)
        if k in (0, 12, 24, len(t) - 1):
            (mdir / f"{tok[:8]}{'_fix' if fix else ''}_frame{k:02d}.png").write_bytes((tmp / f"{k:03d}.png").read_bytes())
    out = mdir / f"{tok[:8]}_{w['scene']}_{'fix' if fix else 'logged_vs_collide'}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "6", "-i", str(tmp / "%03d.png"),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
