"""A37 side-lane control, detector version (descriptive, amendment 2). Same question and groups as src/a37_control.py
(is the leak specific to the in-lane lead, whose future is coupled to the ego plan in the data?), but each annotated
vehicle is followed in 2D: torchvision Faster R-CNN v2 (COCO car / motorcycle / bus / truck, score >= 0.4) on the
generated front-left / front / front-right images; frame 0: the detection with the highest IoU (>= 0.3) with the
projected annotated box, in the view where it is most visible; then IoU tracking (>= 0.3, up to 2 missed frames).
Position per frame: median ego-frame point of the generated depth inside the (20 %-shrunk) box on vehicle-class pixels
(the generated ego frame: no commanded pose involved). At the last frame k <= true contact tracked in both plans with
commanded extra advance >= 1 m: d_gap = x_collide - x_logged; beta_world = (d_gap + d_s_hat) / d_s_hat, d_s_hat = PnP
ctrl * d_s (commanded where PnP is unmeasurable). Truth 0 for every vehicle ahead.
Usage: python src/a37_control2.py <windows.json> <out_dir> <scores.csv> <pnp.csv> <pilot_tokens.txt> <out.csv>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_score as S  # noqa: E402
from a37_control import quat_R  # noqa: E402

VEH = (3, 4, 6, 8)
GROUPS = ("lead", "same_lane", "adjacent_moving", "side_stopped", "far_moving")


def detector():
    import torch
    import torchvision
    dev = __import__("os").environ.get("DET_DEV", "cpu")             # DET_DEV=cuda on a GPU host
    torch.set_num_threads(6)
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").to(dev).eval()

    def run(imgs):
        out = []
        for b in range(0, len(imgs), 6):
            x = [torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255).to(dev) for im in imgs[b:b + 6]]
            with torch.no_grad():
                for o in net(x):
                    lab, sc, bx = o["labels"].cpu().numpy(), o["scores"].cpu().numpy(), o["boxes"].cpu().numpy()
                    out.append(bx[np.isin(lab, VEH) & (sc >= 0.4)])
        return out
    return run


def iou(a, B):
    if len(B) == 0:
        return np.zeros(0)
    x1, y1 = np.maximum(a[0], B[:, 0]), np.maximum(a[1], B[:, 1])
    x2, y2 = np.minimum(a[2], B[:, 2]), np.minimum(a[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)


def project(center, size, yaw, K, C):
    """3D box (t0 = ego frame at frame 0) -> 2D bbox in a view, or None if behind / outside."""
    w_, l_, h_ = size
    c, s = np.cos(yaw), np.sin(yaw)
    cs = np.array([[dx, dy, dz] for dx in (-l_ / 2, l_ / 2) for dy in (-w_ / 2, w_ / 2) for dz in (-h_ / 2, h_ / 2)])
    P = center + cs @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    pc = (np.linalg.inv(C) @ np.vstack([P.T, np.ones(8)]))[:3]
    if (pc[2] <= 0.5).any():
        return None
    uv = (K @ pc)[:2] / pc[2]
    b = np.array([uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max()])
    vis = np.array([max(b[0], 0), max(b[1], 0), min(b[2], 800), min(b[3], 448)])
    if vis[2] - vis[0] < 8 or vis[3] - vis[1] < 8:
        return None
    return vis, (vis[2] - vis[0]) * (vis[3] - vis[1]) / ((b[2] - b[0]) * (b[3] - b[1]))


def track(dets, box0):
    v = iou(box0, dets[0])
    if not len(v) or v.max() < 0.3:
        return None
    tr, last, miss = [dets[0][v.argmax()]], dets[0][v.argmax()], 0
    for D in dets[1:]:
        v = iou(last, D)
        if len(v) and v.max() >= 0.3:
            last, miss = D[v.argmax()], 0
            tr.append(last)
        else:
            miss += 1
            tr.append(None)
            if miss > 2:
                break
    return tr + [None] * (len(dets) - len(tr))


def position(z, v, k, b):
    x1, y1, x2, y2 = b
    dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    u0, u1, w0, w1 = int(x1 + dx), int(x2 - dx), int(y1 + dy), int(y2 - dy)
    d = z["depth"][v, k, w0:w1, u0:u1].astype(np.float32)
    m = np.isin(z["seg"][v, k, w0:w1, u0:u1], S.DYN) & (d >= 1.0)
    if m.sum() < 20:
        return np.nan
    yy, xx = np.nonzero(m)
    uv1 = np.stack([xx + u0, yy + w0, np.ones_like(xx)], 0).astype(np.float64)
    pc = np.linalg.inv(z["cam_k"][v, k]) @ uv1 * d[yy, xx]
    pe = (z["cam2ego"][v, k] @ np.vstack([pc, np.ones((1, pc.shape[1]))]))[:3]
    return float(np.median(pe[0]))


def main(windows, out, scores, pnp, excl, dst):
    import imageio.v3 as iio
    W = {w["token"]: w for w in json.load(open(windows))}
    sc = pd.read_csv(scores).set_index("token")
    pn = pd.read_csv(pnp).set_index("tok")
    ex = set(open(excl).read().split())
    det = detector()
    rows = []
    for tok in sc.index:
        if tok in ex or not np.isfinite(sc.loc[tok, "t_star"]):
            continue
        w = W[tok]
        d = Path(out) / tok
        t = np.array(json.load(open(d / "logged_s0_n25" / "reward.json"))["t"])
        K = int(np.argmin(np.abs(t - sc.loc[tok, "t_star"])))
        Z = {pl: np.load(d / f"{pl}_s0_n25" / "front3.npz") for pl in ("logged", "collide")}
        Mp = {pl: np.load(d / f"{pl}_s0_n25" / "rel_poses.npy") for pl in Z}
        vid = {pl: iio.imread(d / f"{pl}_s0_n25" / "rgb.mp4") for pl in Z}
        ds = Mp["collide"][:, 0, 3] - Mp["logged"][:, 0, 3]
        p = pn.loc[tok[:8]] if tok[:8] in pn.index else None
        ctrl = p.ds_chain / p.ds_cmd if p is not None and np.isfinite(p.ds_chain) else np.nan
        f0 = w["frames"][0]
        R0, T0 = quat_R(f0["ego_q"]), np.array(f0["ego_t"])
        yaw0 = np.arctan2(R0[1, 0], R0[0, 0])
        cache = {}
        for veh in w["vehicles"]:
            b = veh["track"][0]
            ctr = R0.T @ (np.array(b["t"]) - T0)
            tr = np.array([(R0.T @ (np.array(x["t"]) - T0)) for x in veh["track"]])
            x0, y0 = ctr[:2]
            if not (2 <= x0 <= 25 and abs(y0) <= 8):
                continue
            spd = np.hypot(*(tr[1, :2] - tr[0, :2])) / 0.5 if len(tr) > 1 else 0.0
            grp = ("lead" if veh["instance"] == w["lead_instance"] else "same_lane" if abs(y0) <= 1.5 else
                   "adjacent_moving" if abs(y0) <= 5.5 and spd > 0.5 else "side_stopped" if spd <= 0.5 else "far_moving")
            q = b["q"]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)) - yaw0
            best = None
            for v in range(3):
                pr = project(ctr, b["size"], yaw, Z["logged"]["cam_k"][v, 0], Z["logged"]["cam2ego"][v, 0])
                if pr is not None and (best is None or pr[1] > best[2]):
                    best = (v, pr[0], pr[1])
            base = {"token": tok, "scene": w["scene"], "group": grp, "x0": x0, "y0": y0, "speed": spd}
            if best is None:
                rows.append(base | {"tracked": False, "why": "not in view"})
                continue
            v, box0, _ = best
            xs = {}
            for pl in Z:
                if (pl, v) not in cache:
                    cache[(pl, v)] = det([vid[pl][k, :448, 800 * v:800 * (v + 1)] for k in range(len(t))])
                trk = track(cache[(pl, v)], box0)
                xs[pl] = (np.array([position(Z[pl], v, k, bb) if bb is not None else np.nan for k, bb in enumerate(trk)])
                          if trk is not None else np.full(len(t), np.nan))
            ok = [k for k in range(K + 1) if np.isfinite(xs["logged"][k]) and np.isfinite(xs["collide"][k]) and ds[k] >= 1.0]
            if not ok:
                rows.append(base | {"tracked": False, "why": "lost", "view": v})
                continue
            k = ok[-1]
            dgap = xs["collide"][k] - xs["logged"][k]
            dsh = (ctrl if np.isfinite(ctrl) else 1.0) * ds[k]
            rows.append(base | {"tracked": True, "view": v, "t": t[k], "ds_cmd": ds[k], "ds_hat": dsh,
                                "ctrl_measured": np.isfinite(ctrl), "x_logged": xs["logged"][k], "x_collide": xs["collide"][k],
                                "dgap": dgap, "beta_world": (dgap + dsh) / dsh, "beta_cmd": (dgap + ds[k]) / ds[k]})
        del Z, vid, cache
        print(f"{tok[:8]} done ({len(rows)} vehicles so far)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(dst, index=False)
    T = df[df.tracked == True]  # noqa: E712
    print(f"vehicles {len(df)} (tracked in both plans {len(T)}) in {df.token.nunique()} windows")
    rng = np.random.default_rng(0)
    for g in GROUPS:
        x = T[T.group == g]
        if x.empty:
            continue
        sc_ = x.scene.unique()
        bs = [np.median(np.concatenate([x[x.scene == s].beta_world.to_numpy() for s in rng.choice(sc_, len(sc_))]))
              for _ in range(2000)]
        print(f"  {g:16s} n {len(x):3d} ({len(sc_)} scenes) beta_world median {x.beta_world.median():+.2f} "
              f"[{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}] | beta_cmd {x.beta_cmd.median():+.2f} | "
              f"untracked {(df[df.group == g].tracked == False).sum()}")  # noqa: E712


if __name__ == "__main__":
    main(*sys.argv[1:])
