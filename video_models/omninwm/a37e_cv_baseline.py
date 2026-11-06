"""A37e (exploratory, post-hoc 2026-10-18): constant-velocity lead baseline (t0 lead state from annotations).
Usage: python src/a37e_cv_baseline.py <windows.json> <out_dir>"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "<repo>/src")
import a37_score as S
W = {w["token"]: w for w in json.load(open(sys.argv[1]))}; out = Path(sys.argv[2])
print("tok      stop  vlead  | lead-rear error at t* (m): CV-from-past  pm0-world  pm-1.5-world | CV judge: collide logged")
for tok in sorted(p.name for p in out.iterdir() if p.is_dir()):
    w = W[tok]; t = np.array(json.load(open(out/tok/"logged_s0_n25"/"reward.json"))["t"])
    tr = np.array(w["lead_track_t0frame"]); L = w["lead_LW"][0]
    x_true = np.interp(t, np.arange(len(tr))*0.5, tr[:,0]) - L/2
    G = {pl: S.load(out/tok/f"{pl}_s0_n25") for pl in ("logged","collide","pm0","pm-1.5")}
    s_c = G["collide"]["M"][:,0,3]; s_l = G["logged"]["M"][:,0,3]
    hit = np.nonzero(x_true - (s_c + S.FRONT0) <= 0)[0]; k = hit[0] if len(hit) else len(t)-1
    x_cv = x_true[0] + w["lead_speed"]*t                       # lead keeps its t0 speed (past only)
    r = {q: S.track_lead(G[q]["dyn"], G[q]["M"], w)[k] for q in ("pm0","pm-1.5")}
    cv = [int((x_cv - (s + S.FRONT0) <= 0).any()) for s in (s_c, s_l)]
    print(f"{tok[:8]}  {int(w['lead_speed']<0.5)}  {w['lead_speed']:5.2f}  |  {x_cv[k]-x_true[k]:6.2f}  {r['pm0']-x_true[k]:6.2f}  {r['pm-1.5']-x_true[k]:6.2f} | {cv[0]} {cv[1]}")
