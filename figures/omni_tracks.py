"""Per-frame position of the generated lead (nearest car surface in a 1.2 m corridor around the real lead, t0 frame;
same readout as audit_final/fresh_omninwm.py) and the ego front, for every rendered plan of the given windows.
Usage: python omni_tracks.py <windows.json> <out.json> <token-prefix,...>   (run from <out_omni>)"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "work/drive/audit_final"))
import os
import fresh_omninwm
from fresh_omninwm import FRONT, cars_t0, lead_x
# DEPTH_CAL="A,B": undo OmniNWM's depth bias, depth = A * true + B (fitted against the annotated lead, App. A)
if os.environ.get("DEPTH_CAL"):
    A, B = map(float, os.environ["DEPTH_CAL"].split(","))
    _load = np.load

    def _cal_load(p, *a, **k):
        z = _load(p, *a, **k)
        if str(p).endswith("front3.npz"):
            z = {n: z[n] for n in z.files}
            z["depth"] = (z["depth"].astype(np.float32) - B) / A
        return z
    fresh_omninwm.np.load = _cal_load
W = {w["token"]: w for w in json.load(open(sys.argv[1]))}
res = {}
for tok in sys.argv[3].split(","):
    tok = [k for k in W if k.startswith(tok)][0]
    w, d = W[tok], Path(tok)
    t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
    tr = np.array(w["lead_track_t0frame"])
    x_ctr = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 0])
    y_lead = np.interp(t, np.arange(len(tr)) * 0.5, tr[:, 1])
    r = {"t": t.tolist(), "real_rear": (x_ctr - w["lead_LW"][0] / 2).tolist(), "y": y_lead.tolist(), "plans": {}}
    for p in sorted(d.iterdir()):
        if not (p / "front3.npz").exists():
            continue
        P, _, M = cars_t0(p)
        s = M[:, 0, 3]
        gen = [lead_x(P[k], y_lead[k], s[k] + FRONT - 1.0) for k in range(len(t))]
        r["plans"][p.name] = {"ego_front": (s + FRONT).tolist(), "gen": [None if not np.isfinite(g) else g for g in gen]}
        print(tok[:8], p.name, "gen@0 %.1f" % (gen[0] if np.isfinite(gen[0]) else -1), flush=True)
    res[tok] = r
json.dump(res, open(sys.argv[2], "w"))
