"""A37e (exploratory, post-hoc 2026-10-18): does the lead in the past-only worlds echo the prior plan?
Usage: python src/a37e_prior_echo.py <windows.json> <out_dir> <pilot.txt>"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "<repo>/src")
import a37_score as S
W = {w["token"]: w for w in json.load(open(sys.argv[1]))}
out = Path(sys.argv[2]); pilot = set(open(sys.argv[3]).read().split())
print("tok       stop  ds_prior  dlead  beta_prior | err_pm0 err_pm-1.5 (true-lead readout, m, at t*) | judge pm0 pm-1.5 on collide/logged")
for tok in sorted(p.name for p in out.iterdir() if p.is_dir()):
    w = W[tok]; G = {pl: S.load(out/tok/f"{pl}_s0_n25") for pl in ("logged","collide","pm0","pm-1.5")}
    t = np.array(json.load(open(out/tok/"logged_s0_n25"/"reward.json"))["t"])
    tr = np.array(w["lead_track_t0frame"]); L = w["lead_LW"][0]
    x_true = np.interp(t, np.arange(len(tr))*0.5, tr[:,0]) - L/2
    s_c = G["collide"]["M"][:,0,3]; gap = x_true - (s_c + S.FRONT0); hit = np.nonzero(gap <= 0)[0]
    k = hit[0] if len(hit) else len(t)-1
    r = {q: S.track_lead(G[q]["dyn"], G[q]["M"], w) for q in ("pm0","pm-1.5")}
    s = {q: G[q]["M"][:,0,3] for q in r}
    dsp = s["pm0"][k]-s["pm-1.5"][k]; dl = r["pm0"][k]-r["pm-1.5"][k]
    v = []
    for pl in ("collide","logged"):
        P = G[pl]
        for q in ("pm0","pm-1.5"):
            Q = G[q]; v.append(int(any(S.in_box(S.from_t0(S.to_t0(Q["dyn"][j],Q["M"][j]),P["M"][j])).sum()>=S.NMIN for j in range(len(t)))))
    print(f"{tok[:8]}{'*' if tok in pilot else ' '} {int(w['lead_speed']<0.5)}  {dsp:6.2f}  {dl:6.2f}  {dl/dsp if abs(dsp)>=0.5 else float('nan'):6.2f}  | {r['pm0'][k]-x_true[k]:6.2f} {r['pm-1.5'][k]-x_true[k]:6.2f} | C {v[0]} {v[1]}  L {v[2]} {v[3]}")
