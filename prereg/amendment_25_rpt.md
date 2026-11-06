# Amendment 25 - randomized-plan training (RPT) of SWTC (registered before training starts)

Date: 2026-10-15, ~20:30 local time. No RPT checkpoint exists. The A24 arms (original / control / CPC) and their fold-B outputs
exist and are reused unchanged.

## Question and theory
SWTC reads the real lead car's future off the ego plan (exploratory, both folds):
- under the colliding plan, the lead keeps its gap to the ego (lead beta ~0.9-1.0) while the background stays put
  (~0.05);
- in the logs, ego and lead accelerations correlate +0.63 (car-following: the follower's timing is caused by the lead).

So the plan's TIMING is evidence about the lead's future. A24's counterfactual plans (CPC, as in AD-R1) are the LOGGED
plan plus an offset; they still encode the logged timing, hence the lead's future. A24 dented the leak only slightly
(lead beta under collide at +2 s: 0.84 control -> 0.77 CPC; descriptive, computed 2026-10-15 before this
registration).

**Prediction:** counterfactual plans whose timing comes from the PAST ONLY (an RCT-style randomizer) carry no
information about the future, so at the optimum the lead does not move with the plan.

## Method (frozen; src/swtc_cpc.py --cf-mode retime; unit-tested)
- Identical to A24 (head fine-tune, fold A frames, 6,000 frames, seed 0, lr, schedule, teacher-anchored targets with
  true vehicle voxels re-expressed into the counterfactual frames), except the counterfactual plan:
  - arc length s(t) = v0 t + a t^2 / 2, a ~ U(-4, 4) m/s^2, stopping at zero speed;
  - v0 = the ego speed over the last keyframe interval (previous pose -> current; available for 5,869 / 6,019 frames;
    frames without it keep the logged plan);
  - placed along the logged path SHAPE (heading = tangent), plus the same lateral c t^2, c ~ U(-0.4, 0.4).
- **Arms:**
  - **RPT-1.0** (--p-cf 1.0, primary): every frame with v0 uses a randomized plan.
  - **RPT-0.5** (--p-cf 0.5, secondary): same mixing as CPC; isolates the anchoring.

## Evaluation (fold B, once; identical to A24)
- <work>/swrh_f1 windows (115 / 11 scenes), plans logged / collide / brake / still; src/swtc_verifier.py
  --soft-all.
- mIoU guard: audit3/swtc_repro.py on the A24 300 frames.
- Statistics: audit3/a25_stats.py.

## Endpoints (scene-cluster bootstrap B = 10,000, seed 0, paired by window)
- **P1 (leak removed):** lead beta at +2 s under the collide plan, median paired difference RPT-1.0 - CPC, upper 95 %
  bound < 0.
- **P2 (clearing):** clearing +2 s, RPT-1.0 - CPC, upper bound < 0.
- **P3 (own judge):** self recall on the collide candidate, RPT-1.0 - CPC, lower bound > 0.
- **P4 (false alarms):** self false alarm on the logged path, RPT-1.0 - CONTROL, upper bound <= +0.05.
- **Guard:** mIoU (avg 1-3 s) of RPT-1.0 within 1.5 points of control. Moving-agent accuracy may drop because the plan
  no longer leaks the future; reported, not an endpoint.
- **Descriptive:** RPT-0.5 vs CPC (anchoring alone); background beta (must stay ~0); all arms' kept / recall / FA.
