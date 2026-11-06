# Amendment 24 - counterfactual-plan consistency (CPC) training of SWTC (registered before training starts)

Date: 2026-10-14, ~10:45 local time. No CPC or control checkpoint exists yet; training uses fold A frames only (fold != 1).

## Question
SWTC, a trajectory-conditioned camera world model, erases REAL lead cars when the plan drives into them (exploratory,
both folds: kept +2 s 0.72 under the logged plan vs 0.08 under a colliding plan). Its own collision check therefore
catches 5.5 % of colliding candidates, and neutral-reference judges fail on it (other traffic stops with the ego).

Hypothesis: the clearing is learned from logs in which the plan and the future are always consistent. Training on
COUNTERFACTUAL plans with the real future re-expressed into their frames teaches "agents do not vanish because of my
plan", with no collision data and no labels.

## Method (frozen; src/swtc_cpc.py)
- **Head fine-tune:** SWTC head, backbone and neck frozen; AdamW lr 2e-5 (sampling offsets x 0.1), warmup 100, cosine;
  accum 4; 6,000 frames; seed 0. Data: fold A real frames (2,564).
- **CPC arm (--p-cf 0.5):** with probability 0.5 the future poses are replaced by a counterfactual plan:
  - x += 0.5 a t^2, y += c t^2, with a ~ U(-4, 4) m/s^2 and c ~ U(-0.4, 0.4);
  - the target is the real future occupancy re-expressed into the counterfactual frames (unit-tested: identity exact;
    a 2 m shift = 5 cells);
  - teacher-anchored: the frozen model's non-vehicle prediction under the same plan, plus the true vehicle voxels.
- **Control arm (--p-cf 0):** identical, logged plan only.

## Evaluation (fold B, once)
- **Real-car path clearing:** <work>/swrh_f1 windows (115 windows, 11 scenes; real moving lead car ahead; plans
  logged / collide / brake / still). Run src/swtc_verifier.py --soft-all with each checkpoint; score with
  audit3/swrh_full.py.
- **Accuracy guard:** audit3/swtc_repro.py on 300 random fold-B frames (prep 300 <dir> 1), logged path, Occ3D with the
  camera mask.
- **Models compared:** original, control, CPC.

## Endpoints (scene-cluster bootstrap B = 10,000, seed 0; paired by window; Holm over P1, P2)
- **P1 (clearing removed):** the clearing effect, kept(logged) - kept(collide) at +2 s, is smaller for CPC than for the
  control: CPC - control < 0.
- **P2 (the model's own judge works):** the self judge's recall on the collide candidate (each plan in its own forecast,
  any of +1 / +2 / +3 s), CPC - control > 0.
- **P3 (no false-alarm cost):** self-judge false alarm on the logged path, CPC - control; upper 95 % bound <= +0.05.
- **Guard:** fold-B mIoU (avg 1-3 s) of CPC within 1.5 points of control.
- **Descriptive:** everything for the original model; kept per plan and horizon; brake / still judges.
