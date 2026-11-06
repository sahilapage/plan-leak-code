"""Evaluate a COME checkpoint (released or fine-tuned by src/come_ft.py) with the frozen test code, weights swapped only.
  probe  <ckpt> <fold> <n> <out.parquet>   insertion test (src/probe_insert.py, A14 protocol)
  hazard <ckpt> <fold> <n> <out.parquet>   inserted-hazard decision test (src/probe_hazard.py)
  dump   <ckpt> <name> [folds]              soft dump runs/soft/<name>.npy (N, 2, 4, 200, 200) over val folds A+B, the format
                                            of src/come_infer.py, plus mIoU / IoU at +1/2/3 s against Occ3D in each future
                                            frame (argmax warped into the future frame, cells outside the source grid = free)
<ckpt> = "released" for the original checkpoint.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dome_infer as di  # noqa: E402
import probe_insert as pi  # noqa: E402
from common import RUNS  # noqa: E402


def come_with(ckpt):
    if ckpt.startswith("flow:"):                                  # COME-T (src/come_flow.py): transport pathway
        import come_flow as cfl

        class ComeT(cfl.ComeTEval):
            def __init__(self):
                super().__init__(ckpt[5:])
        return ComeT

    class ComeCk(pi.Come):
        def __init__(self):
            super().__init__()
            if ckpt != "released":
                sd = torch.load(ckpt, map_location="cpu", weights_only=False)
                r = self.net.load_state_dict(sd["state_dict"], strict=False)
                print("fine-tuned weights: missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys),
                      "iter", sd.get("iter"), flush=True)
            self.net.eval()
    return ComeCk


def dump(ckpt, name, folds="01"):
    m = come_with(ckpt)()
    net, ci = m.net, m.ci
    infos, wins = di.windows({int(f) for f in folds})
    out = np.lib.format.open_memmap(str(RUNS / "soft" / f"{name}.npy"), "w+", np.float16, (len(wins), 2, 4, 200, 200))
    vh = torch.tensor(di.VEHICLE, device="cuda")
    ob = torch.tensor(di.OBSTACLE, device="cuda")
    inter, union, ii_, iu_ = np.zeros((3, 17)), np.zeros((3, 17)), np.zeros(3), np.zeros(3)
    toks = []
    bs = 8
    with torch.no_grad():
        for b0 in range(0, len(wins), bs):
            batch = wins[b0: b0 + bs]
            src, poses_all, frs = [], [], []
            for scene, idx in batch:
                fr = infos[scene][idx: idx + 10]
                frs.append((scene, fr))
                src.append(np.stack([di.load_occ(scene, f["token"]) for f in fr[:4]]).transpose(0, 3, 2, 1))
                poses_all.append([di.pose_of(f) for f in fr])
                toks.append(fr[3]["token"])
            if hasattr(m, "probs"):                                      # COME-T: the mixture distribution
                p, _ = m.probs([s_.transpose(0, 3, 2, 1) for s_ in src], poses_all)
                pv, po, lab = p[:, :, vh].sum(2), p[:, :, ob].sum(2), p.argmax(2)
                B = p.shape[0]
            else:
                pv = None
            inputs = {"source_occs": torch.from_numpy(np.stack(src).astype(np.int64)),
                      "source_metas": [{"ego2global": ps[:4]} for ps in poses_all],
                      "metas": [{"scene_range": ci.SCENE_RANGE}] * len(batch)}
            if pv is None:
              x, _ = net.pre_process(inputs)
              B, T, C, Z, H, W = x.shape
              x = x.reshape(B, -1, H, W)
              enc = []
              for e in net.encoder:
                  x = e(x)
                  enc.append(x)
              for i in reversed(range(len(net.decoder))):
                  x = net.decoder[i](enc[i], x)
              tT, tZ, tH, tW = net.target_seq_size
              x = net.temporal_stem(x).reshape(B * tT, -1, H, W)
              logits = net.occ_sem_conv(net.temporal_shared_branch(x)).reshape(B, tT, -1, tZ, H, W)[..., :tH, :tW]
              p = logits.float().softmax(dim=2)
              pv, po, lab = p[:, :, vh].sum(2), p[:, :, ob].sum(2), p.argmax(2)
            for b in range(B):
                ps = poses_all[b]
                res = np.zeros((2, 4, 200, 200), np.float32)
                cur = torch.from_numpy(src[b][3]).cuda()
                res[0, 0] = torch.isin(cur, ob)[di.ZLO:di.ZHI + 1].any(0).float().T.cpu().numpy()
                res[1, 0] = torch.isin(cur, vh)[di.ZLO:di.ZHI + 1].any(0).float().T.cpu().numpy()
                scene, fr = frs[b]
                for k, f in enumerate((1, 3, 5)):
                    for ch, vol in ((0, po[b, f]), (1, pv[b, f])):
                        res[ch, k + 1] = ci.warp(vol, ps[3], ps[4 + f])[di.ZLO:di.ZHI + 1].amax(0).T.cpu().numpy()
                    w_ = ci.warp(lab[b, f].float(), ps[3], ps[4 + f])
                    valid = ci.warp(torch.ones_like(lab[b, f].float()), ps[3], ps[4 + f]) > 0.5
                    pred = torch.where(valid, w_, torch.full_like(w_, 17.0)).round().long().cpu().numpy().transpose(2, 1, 0)
                    gt = di.load_occ(scene, fr[4 + f]["token"])
                    for c in range(17):
                        inter[k, c] += np.sum((pred == c) & (gt == c))
                        union[k, c] += np.sum((pred == c) | (gt == c))
                    ii_[k] += np.sum((pred != 17) & (gt != 17))
                    iu_[k] += np.sum((pred != 17) | (gt != 17))
                out[b0 + b] = res.astype(np.float16)
            if b0 % 800 == 0:
                print(f"{b0}/{len(wins)}", flush=True)
    out.flush()
    json.dump(toks, open(RUNS / "soft" / f"{name}_tokens.json", "w"))
    miou = (inter / np.maximum(union, 1)).mean(1)
    print("mIoU 1/2/3 s", np.round(100 * miou, 2), "avg", round(100 * miou.mean(), 2), "| IoU",
          np.round(100 * ii_ / iu_, 2), "avg", round(100 * (ii_ / iu_).mean(), 2), flush=True)
    print("DUMP DONE", name, len(toks))


def main():
    mode, ckpt = sys.argv[1], sys.argv[2]
    if mode == "dump":
        return dump(ckpt, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "01")
    pi.Come = come_with(ckpt)
    rest = sys.argv[3:]
    if mode == "probe":
        sys.argv = ["probe_insert.py", "come"] + rest
        pi.main()
    elif mode == "hazard":
        import probe_hazard as ph
        sys.argv = ["probe_hazard.py", "come"] + rest
        ph.main()


if __name__ == "__main__":
    main()
