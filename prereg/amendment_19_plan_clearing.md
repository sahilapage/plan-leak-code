# Amendment 19 - does a plan-conditioned world model clear the path it evaluates? (registered blind)

Date: 2026-10-12, 01:45 server time. Registered BEFORE the fold-A development result exists: the fold-A run
(results/probe/pc_dome_fA.parquet) is in progress and not inspected. The only output seen is an 8-window COME smoke run, in
which the plan had no effect, as expected for a plan-blind model.

## Question
A18 development found that DOME, which is conditioned on the ego's planned path, misses in-path cars at every lead,
including at 1 s. Its in-path cars are kept far less often than adjacent-lane cars at the same speed.

AD-R1 (CVPR 2026, arXiv 2511.20325) reports that DOME and I2-World "hallucinate an implausibly safe future" (obstacles
vanish) under unsafe trajectories, measured on its RFB benchmark (correlational: IoU on 450 unsafe clips).

We test the effect causally and separate it from A14's carrying failure, using a PARKED car, which needs no carrying. The
test is a 2 x 2 within each window, with the same scene, the same car, the same diffusion seed and the same past frames
(src/plan_cond.py):
- **Car position:** in the ego's lane, where the logged path drives 1.5 m into it at 2 s, vs the same car one lane over.
- **Plan given to the model:** the logged path vs a stop plan (same heading, constant deceleration <= 6 m/s^2, stopping
  3 m short of the car).

Kept at h means q50 >= 0.2 on the car's footprint, in the frame the plan puts the ego in at h.

## Method (frozen; code committed in 7f682d3)
- **Code:** src/plan_cond.py (sha256 76a93261e821910b3fe723fcbe4877ffb06b842d0c30a373d6adf87e260479e3) and
  analysis/pc_stats.py (sha256 130874f657172172df0ee093b0342634065129ccc6b4454ca68919d2eb2e9b89).
- **Models:** released DOME (plan-conditioned) and released COME (plan-blind negative control).
- **Windows:** fold B, selection seed 0, all valid windows up to 300; windows perceived at t0 in all four cells.

## Endpoints (DOME, fold B, once; scene-cluster bootstrap B = 10,000, seed 0; Holm over P1 and P2)
- **P1:** interaction at +2 s (contact),
  [kept(in-path, stop) - kept(in-path, logged)] - [kept(adjacent, stop) - kept(adjacent, logged)] > 0, with the 95 %
  interval above 0.
- **P2:** the same interaction at +1 s (before the ego reaches the car; the footprint lies outside the ego box) > 0.
- **Negative control (descriptive):** the COME interaction at +1 / +2 / +3 s; expected near 0.
- **Descriptive:**
  - the four cells per horizon;
  - the footprint outside the ego box at +2 / +3 s;
  - the no-car coverage (base_q50);
  - the stop plan's deceleration.

## Interpretation fixed in advance
- **If P1 and P2 are supported:** the plan-conditioned judge removes an obstacle because the plan it is asked to judge
  drives into it. This is AD-R1's optimism, measured causally. It is a second cause, separate from carrying (A14),
  which COME shows without any plan.
- **If not supported:** DOME's in-path misses are attributed to carrying and dilution alone, and the paper says so.
