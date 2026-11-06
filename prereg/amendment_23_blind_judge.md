# Amendment 23 - the blind judge (registered before its I2-World development result and before any fold-B output)

Date: 2026-10-12, ~15:25 local time. At commit time results/probe/iiblind_pc_fA.parquet does not exist (the run is in
progress); no I2-World blind output has been inspected beyond a 3-window smoke run that only checked that the
stationary-ego stream perceives the t0 scene (485 vs 481 vehicle cells).

## Question and principle
A world-model judge erases what the ego would hit:
- DOME, because the plan it is conditioned on drives into it (A19);
- I2-World, because the ego approaches it, whatever the plan (A21 not supported; in-lane loss under both plans).

Principle: a judge must not know what the ego will do. The blind judge forecasts an EGO-NEUTRAL world: past occupancy
re-expressed in the t0 frame, a stationary ego at every stream step, a stationary plan. It then checks every candidate
geometrically against that one forecast. It needs no training and makes one forecast for all candidates.

## Disclosure (fold A only)
- **DOME, parked car on a collision course (150 windows, 33 scenes; src/blind.py pc):** noticed at contact
  self 0.573 / plan-swap 0.893 / blind 0.953.
  - Blind - self: +0.380 [+0.271, +0.485].
  - Blind - swap: +0.060 [-0.005, +0.127].
  - Car kept: 0.560 / 0.947 / 0.953.
- **DOME, real traffic (300 random windows; the logged path is collision-free):** false alarm self 0.000 / swap 0.063
  [0.019, 0.121] / blind 0.033 [0.013, 0.057].
  - The blind false alarms are mostly confident vehicles in the ego box at +3 s with no true vehicle there.
  - Self stays at 0 at every threshold because it zeroes its own path (parked hazards noticed 0.573 even at
    tau 0.005).
- **DOME, plan-swap on moving lead hazards (A22 development):** 0.057 -> 0.570 pooled.
- **I2-World, plan-swap development:** look-ahead self 0.185 / 0.021 / 0.014 -> swap 0.555 / 0.493 / 0.582; parked
  in-lane 0.527 -> 0.534.
- **Not inspected:** DOME blind la and I2-World blind (pc, real) are running.

## Method (frozen)
- **Code:** src/blind.py (DOME; judges self / swap / blind, same seed), src/ii_blind.py + src/ii_blind_run.py (I2-World;
  judges self / blind), analysis/blind_stats.py; hashes in amendment_23_hashes.txt.
- **Fold B, selection seed 0:**
  - DOME: `blind.py pc 1 300`, `blind.py la 1 300`, `blind.py real 1 300`.
  - I2-World: `ii_blind.py` pc 300, la 300 and real 300 (windows with the frame 2 s before t0), each run with
    ii_blind_run.py, then scored.

## Endpoints (fold B, once; scene-cluster bootstrap B = 10,000, seed 0; Holm over P1-P4)
- **P1 (DOME, pc):** noticed at contact, blind - self > 0.
- **P2 (DOME, la):** noticed pooled over leads, blind - self > 0.
- **P3 (I2-World, pc):** noticed at contact, blind - self > 0.
- **P4 (I2-World, la):** noticed pooled over leads, blind - self > 0.
- **P5 (real traffic, per model; stated tolerance):** blind false-alarm rate on the collision-free logged path; claimed
  acceptable if the upper 95 % bound is <= 0.08. Reported either way, with self (and swap for DOME).
- **Descriptive:** blind - swap (DOME); per-lead noticed; car kept; false alarms without the car.
