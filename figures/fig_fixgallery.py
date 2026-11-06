"""App. figure: the test-time repair on more scenes. Per scene (row): OmniNWM under the logged plan, under the crashing
plan, and the repair (OmniNWM's world generated under a past-only plan, pm-1.5, re-rendered with its own depth from the
crashing plan's cameras), all at the last frame where the real gap is still at least 1 m (later frames put the camera
inside the reconstructed car). Green: the real car's annotated box.
Usage: python fig_fixgallery.py <windows_merged.json> <out_omni>"""
import json
import sys
from pathlib import Path

import numpy as np

from style import BLUE, RED, plt
from teaser2 import TR, X0, Y0, frame, repaired

HERE = Path(__file__).resolve().parent
SCENES = (("6b3c851413064994bb0babae7e74efad", "scene-0097, stopped car"),
          ("8d67c4024af6470faada45dc41ba9a16", "scene-0556, moving car"),
          ("918b8f22899b460abb6903efe15d663f", "scene-1070, moving car, night"))
GAP = 1.0


def main(windows, out_omni):
    W = {x["token"]: x for x in json.load(open(windows))}
    fig = plt.figure(figsize=(5.5, 3.3))
    gs = fig.add_gridspec(len(SCENES), 3, wspace=0.03, hspace=0.07, left=0.05, right=1.0, top=0.95, bottom=0.0)
    cols = (("logged_s0_n25", "logged plan", "#5f5f5f", None), ("collide_s0_n25", "crashing plan (OmniNWM)", RED, None),
            ("collide_s0_n25", "crashing plan, repaired", BLUE, repaired))
    for r, (tok, ttl) in enumerate(SCENES):
        tr = TR[tok]
        d = Path(out_omni) / tok
        gap = np.array(tr["real_rear"]) - np.array(tr["plans"]["collide_s0_n25"]["ego_front"])
        k = int(np.nonzero(gap >= GAP)[0][-1])       # last frame with the real gap still at least 1 m
        t = np.array(tr["t"])[k]
        for c, (pl, lab, col, fix) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            frame(ax, W[tok], tr, d, pl, k, col, arrow=False, gap_label=(c > 0), img=fix(d, k) if fix else None)
            if r == 0:
                ax.set_title(lab, fontsize=8.5, pad=2.5, color=col)
            if c == 0:
                ax.text(X0 + 8, int(Y0 * 448) + 10, f"{ttl}, $t$ = {t:.1f} s", fontsize=7.0, color="white", va="top",
                        bbox=dict(fc="black", alpha=0.6, lw=0, pad=1.0))
    fig.savefig(HERE / "app_fixgallery.pdf", bbox_inches="tight", pad_inches=0.01, dpi=230)
    fig.savefig(HERE / "app_fixgallery.png", bbox_inches="tight", pad_inches=0.01, dpi=160)


if __name__ == "__main__":
    main(*sys.argv[1:3])
