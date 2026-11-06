"""Re-run the fresh OmniNWM beta with OmniNWM's depth corrected by the fit omni = A * true + B (or uncorrected)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "work/drive/audit_final"))
import fresh_omninwm as F
A, B = float(sys.argv[5]), float(sys.argv[6])
_load = np.load


class Cal(dict):
    pass


def load(p, *a, **k):
    z = _load(p, *a, **k)
    if str(p).endswith("front3.npz"):
        z = {n: z[n] for n in z.files}
        z["depth"] = ((z["depth"].astype(np.float32) - B) / A)
    return z


F.np.load = load
F.main(*sys.argv[1:5])
