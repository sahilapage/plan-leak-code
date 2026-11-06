# Amendment 22 - plan-swap judging, a test-time fix for path clearing (registered before any fold-B output exists)

Date: 2026-10-12, ~11:05 local time.

## Question
A19 showed causally that DOME removes an obstacle because the plan it is conditioned on drives into it. Principle: the
world a candidate is judged in must not be conditioned on that candidate. Plan-swap judging forecasts ONCE under a fixed
reference plan, independent of the scene (brake from the t0 speed at 4 m/s^2 along the t0 heading). It then checks each
candidate geometrically: the candidate's ego box at h, placed in the reference forecast's frame at h. It needs no
training and makes one forecast for all candidates.

## Disclosure (fold A only)
- **pc mode (A19 windows, parked in-path car, 150 windows, 33 scenes):** collision noticed at contact, self-conditioned
  0.573 vs plan-swap 0.893 (+0.320 [+0.222, +0.414]); car kept 0.560 vs 0.947; false alarms without the car 0 / 0.
- **Not inspected at registration:** la mode (A18 windows, moving lead car; running) and the real-traffic false-alarm
  mode (queued); neither output exists.
- **Smoke run:** 8 windows.
- **Related development (disclosed):** I2-World shows no plan effect in the A19 2 x 2 (fold A), so plan-swap is
  registered for DOME only.

## Method (frozen)
- **Code:** src/planswap.py (modes pc / la / real; A_REF = 4.0 m/s^2) and analysis/ps_stats.py; hashes in
  amendment_22_hashes.txt.
- **Model:** released DOME; the self-conditioned and plan-swap forecasts share the diffusion seed.
- **Fold B runs:**
  - `planswap.py pc 1 300`;
  - `planswap.py la 1 300` (all valid windows, 3 leads);
  - `planswap.py real 1 300` (300 random windows, no insertion).

## Endpoints (fold B, once; scene-cluster bootstrap B = 10,000, seed 0, paired by window; Holm over P1-P3)
- **P1 (pc):** collision noticed at contact, swap - self > 0.
- **P2 (la):** noticed at h = L pooled over leads, swap - self > 0.
- **P3 (real traffic, non-inferiority):** false alarm on the collision-free logged path (any of +1 / +2 / +3 s),
  swap - self; upper 95 % bound <= +0.03.
- **Descriptive:** per-lead noticed; car kept; false alarms without the inserted car.
