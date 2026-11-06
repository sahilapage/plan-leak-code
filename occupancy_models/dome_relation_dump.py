"""DOME forecasts on real-car windows (src/realhazard.py selection) under the LOGGED and the COLLIDE plan, saved in the
src/ii_realhazard.py layout so audit3/ii_relation_leak.py / ii_plan_leak.py read them unchanged:
  <dir>/windows.pkl        token -> selection record (scene, idx, a, lead_track, ...)
  <dir>/plan/<tok>_maps.npz  veh (2, 1, 4, 200, 200) float16: [logged, collide] vehicle probability (max over the
                           z band), h = 0..3 s, each in its own plan's frame (pi.Dome output).
Usage (server, vla env, GPU): python src/dome_relation_dump.py <fold> <n> <dir>"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_insert as pi  # noqa: E402
import realhazard as rh  # noqa: E402


def main(fold, n, out, bs=4):
    out = Path(out)
    (out / "plan").mkdir(parents=True, exist_ok=True)
    infos, sel = rh.select(fold, n)
    pickle.dump({s["token"]: s for s in sel}, open(out / "windows.pkl", "wb"))
    print(f"{len(sel)} windows", flush=True)
    net = pi.Dome()
    for b0 in range(0, len(sel), bs):
        batch = sel[b0: b0 + bs]
        frs = [infos[s["scene"]][s["idx"]: s["idx"] + 10] for s in batch]
        logged = [[di.pose_of(f) for f in fr] for fr in frs]
        cand = [lp[:4] + [rh.shift_pose(lp[k], lp[3], 0.5 * s["a"] * (0.5 * (k - 3)) ** 2) for k in range(4, 10)]
                for s, lp in zip(batch, logged)]
        occ = [np.stack([di.load_occ(s["scene"], f["token"]) for f in fr[:4]]) for s, fr in zip(batch, frs)]
        Fl, Fc = net(occ, logged, seed=b0), net(occ, cand, seed=b0)
        for i, s in enumerate(batch):
            veh = np.stack([Fl[i][:4], Fc[i][:4]])[:, None].astype(np.float16)
            np.savez_compressed(out / "plan" / f"{s['token']}_maps.npz", veh=veh)
        print(f"{b0 + len(batch)}/{len(sel)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
