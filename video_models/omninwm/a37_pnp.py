"""PnP odometry on OmniNWM generations: ORB matches (static classes) frame A -> frame B; 3D from A's generated depth
(ego frame A), 2D in B; solvePnPRansac per view (camera intrinsics, cam2ego) -> ego x of B in A's ego frame."""
import sys, json, numpy as np, cv2
from pathlib import Path
import imageio.v3 as iio
STAT = (0, 1, 6, 7, 8)
orb = cv2.ORB_create(4000); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
def gray(vid, k, v):
    return cv2.cvtColor(vid[k][:448, 800 * v:800 * (v + 1)], cv2.COLOR_RGB2GRAY)
def rel_x(za, va, ka, zb, vb, kb, views=(1, 0, 2)):
    """ego-x of camera B (frame kb of video vb) in ego frame of A (frame ka of video va); median over views."""
    ests = []
    for v in views:
        A, B = gray(va, ka, v), gray(vb, kb, v)
        pa, da = orb.detectAndCompute(A, None); pb, db = orb.detectAndCompute(B, None)
        if da is None or db is None: continue
        m = [p[0] for p in bf.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(m) < 12: continue
        ua = np.array([pa[x.queryIdx].pt for x in m]); ub = np.array([pb[x.trainIdx].pt for x in m])
        u, w = ua[:, 0].astype(int).clip(0, 799), ua[:, 1].astype(int).clip(0, 447)
        d = za["depth"][v, ka][w, u].astype(np.float64); s = za["seg"][v, ka][w, u]
        ok = np.isin(s, STAT) & (d > 1.5) & (d < 40)
        if ok.sum() < 12: continue
        K = za["cam_k"][v, ka].astype(np.float64)
        Pc = (np.linalg.inv(K) @ np.vstack([ua[ok].T, np.ones(ok.sum())])) * d[ok]      # camera-A coords
        r = cv2.solvePnPRansac(Pc.T, ub[ok], zb["cam_k"][v, kb].astype(np.float64), None,
                               reprojectionError=3.0, iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
        if not r[0] or r[3] is None or len(r[3]) < 12: continue
        R, _ = cv2.Rodrigues(r[1]); tvec = r[2].ravel()
        camB_in_camA = -R.T @ tvec                                                       # B's centre in cam A
        C = za["cam2ego"][v, ka]
        ests.append(((C[:3, :3] @ camB_in_camA + C[:3, 3])[0] - C[0, 3], len(r[3])))   # ego-x shift of the rig
    if not ests: return np.nan, 0
    e = np.array(ests)
    return float(np.median(e[:, 0])), int(e[:, 1].sum())
def load(d):
    return np.load(d / "front3.npz"), iio.imread(d / "rgb.mp4"), np.load(d / "rel_poses.npy")
