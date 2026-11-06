# Amendment 21 - path clearing on I2-World (registered blind, before its fold-A development result exists)

Date: 2026-10-12, ~10:35 local time. The fold-A development run (results/probe/iiplan_dev_fA.parquet) is in progress and not
inspected. Seen so far: only the 3-window self-test of the harness. It shows that overriding with the LOGGED plan
reproduces the un-overridden output exactly (max |diff| 0), that the frame-transform convention matches the logged meta
(error 0.0), and that the stop plan changes the output.

## Question
A19 showed causally that DOME, conditioned on the plan it judges, removes a parked car because that plan drives into it.
Does I2-World, the state of the art and also conditioned on the future ego poses, do the same?

## Method (frozen)
- **The A19 2 x 2 on I2-World:** in-path vs adjacent parked real-shape car; logged plan vs stop plan.
- **Harness:** src/ii_plan.py (prep / score; selection = src/plan_cond.py, windows need the frame 2 s before t0) and
  src/ii_plan_run.py (5-frame stream t0-2 .. t0).
- **Plan override at t0:** curr_ego_to_global, ego_to_global_translation / _rotation, curr_to_future_ego_rt and
  gt_ego_lcf_feat are all rebuilt from the stop plan's poses; the command is kept.
- **Statistics:** analysis/pc_stats.py. Hashes in amendment_21_hashes.txt.
- **Windows:** fold B, selection seed 0, n = 300 before the 2-s-history filter; windows perceived at t0 in all four
  cells.

## Endpoints (I2-World, fold B, once; scene-cluster bootstrap B = 10,000, seed 0; Holm over P1, P2)
- **P1:** interaction at +2 s (contact) > 0.
- **P2:** interaction at +1 s (before contact) > 0.
- **Descriptive:**
  - the four cells per horizon;
  - the footprint outside the ego box;
  - the matched-gap control (audit3/pc_gap.py logic).
