"""A37 run lists (registered rule): up to 3 windows per lead instance, >= 3 s apart (earliest first), every one of the
6 cameras x 33 frames of the 12 Hz clip on disk, the pilot window excluded, fixed random order (seed 0).
Writes <pack>/main_omni.txt (first N_MAX) and <pack>/main_vista.txt (the same list; Vista scores beta only).
Usage (pod, omni env): python src/a37_list.py /workspace/pack /workspace/pilot.txt"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

N_MAX, N_FR, PER_LEAD, GAP_US = 30, 33, 3, 3_000_000
PKL = "/workspace/pretrained/omninwm/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl"
ROOT = Path("/workspace/data/nuscenes")


def main(pack, pilot):
    pack = Path(pack)
    W = json.load(open(pack / "windows.json"))
    pil = set(open(pilot).read().split())
    infos = sorted(pickle.load(open(PKL, "rb"))["infos"], key=lambda e: e["timestamp"])
    tok2i = {f["token"]: i for i, f in enumerate(infos)}
    seen, keep = {}, []
    for w in sorted(W, key=lambda w: w["t0"]):
        prev = seen.get(w["lead_instance"], [])
        if len(prev) >= PER_LEAD or (prev and w["t0"] - prev[-1] < GAP_US):
            continue
        if w["token"] in pil or w["token"] not in tok2i:
            continue
        s = tok2i[w["token"]]
        clip = infos[s:s + N_FR]
        ok = len(clip) == N_FR and all(
            (ROOT / c["data_path"].replace("../data/nuscenes/", "")).exists() for f in clip for c in f["cams"].values())
        if ok:
            keep.append(w)
            seen[w["lead_instance"]] = prev + [w["t0"]]
    order = np.random.default_rng(0).permutation(len(keep))
    keep = [keep[i] for i in order]
    (pack / "main_omni.txt").write_text("\n".join(w["token"] for w in keep[:N_MAX]) + "\n")
    vis = keep[:N_MAX]
    (pack / "main_vista.txt").write_text("\n".join(w["token"] for w in vis) + "\n")
    print(f"{len(W)} windows -> {len(seen)} lead instances -> {len(keep)} windows with full clips on disk (pilot excluded); "
          f"OmniNWM list {min(N_MAX, len(keep))}, Vista list {len(vis)}")
    for w in keep[:N_MAX]:
        print(f"  {w['scene']} {w['token'][:8]} lead_x0 {w['lead_track_t0frame'][0][0]:5.1f} v_lead "
              f"{w['lead_speed']:4.1f} v0 {w['v0_past']:4.1f} a* {w['a_star']:.2f} contact +{w['h_contact']} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
