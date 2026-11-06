"""Geometry-only plan conditioning for SparseWorld-TC (the leak law's fix by construction).
SWTC receives the ego plan twice: (1) geometrically (fut2cur maps future-frame query points to the current frame where
image features are sampled: exact, causal) and (2) through a learned embedding (transformer.pe_mln of the flattened
4x4 future pose), which can carry the plan's observational coupling to other agents (the leak). install() makes the
embedding read a PAST-ONLY reference pose instead (constant velocity v0 from the previous keyframe, straight ahead),
so no input but geometry depends on the future plan. Instance-level patch: checkpoint keys are unchanged."""
import numpy as np
import torch


def install(net):
    mln = next(m.pe_mln for m in net.modules() if hasattr(m, "pe_mln"))
    if getattr(mln, "_ref_installed", False):
        return mln
    orig = mln.forward
    mln._ref = None

    def forward(x, pos):
        if mln._ref is not None:
            bf, q, _ = pos.shape
            pos = mln._ref.to(device=pos.device, dtype=pos.dtype).reshape(-1, 16)
            assert bf == pos.shape[0], "batch size 1 only (frame-major fut2cur order)"
            pos = pos.unsqueeze(1).expand(bf, q, 16)
        return orig(x, pos)

    mln.forward = forward
    mln._ref_installed = True
    return mln


def cv_ref(v0, times=(0.0, 1.0, 2.0, 3.0)):
    """(len(times), 4, 4) fut2cur of a straight constant-velocity plan at speed v0 (m/s): past-only."""
    out = torch.eye(4).repeat(len(times), 1, 1)
    for k, t in enumerate(times):
        out[k, 0, 3] = float(v0) * t
    return out


def _quat_mat(t, q):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def past_speed(infos):
    """token -> forward ego speed (m/s) over the last keyframe interval (previous pose -> current), past only
    (same as src/swtc_cpc_v2.py past_speed; duplicated to keep the verifier free of trainer imports)."""
    by = {e["token"]: e for e in infos}
    out = {}
    for e in infos:
        p = by.get(e.get("prev") or "")
        if p is None:
            continue
        dt = (e["timestamp"] - p["timestamp"]) / 1e6
        if not 0.2 < dt < 1.0:
            continue
        rel = np.linalg.inv(_quat_mat(e["ego2global_translation"], e["ego2global_rotation"])) @ \
            _quat_mat(p["ego2global_translation"], p["ego2global_rotation"])
        out[e["token"]] = max(0.0, -float(rel[0, 3]) / dt)
    return out
