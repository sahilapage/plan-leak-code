# Amendment 20 - the intervention test (A14) on I2-World, the state of the art (registered before any fold-B output)

Date: 2026-10-12.

## Question
A14 showed on COME and DOME that a car the model has just seen is kept when parked and lost when it moves at constant
speed. Does the newest model, I2-World (ICCV 2025; streaming tokenizer + world model conditioned on the future ego poses),
share the failure?

## Disclosure (fold A only; development)
- **Run:** src/ii_probe.py prep / src/ii_probe_run.py / ii_probe.py score; 120 windows, 32 scenes. A first run died in
  the 2026-10-11 /home I/O errors; it was rerun unchanged.
- **Kept at +3 s, real-shape cars:** parked 0.617 / 4 m/s 0.125 / 8 m/s 0.167 / 12 m/s 0.108.
- **P1 (parked minus 4 m/s):** +0.492 [+0.359, +0.617].
- **P2 (slope over 4-12 m/s):** -0.0021 [-0.0159, +0.0120], not supported (floor).
- **Perceived at t0:** 0.99.

## Method (frozen)
- **Code:** src/ii_probe.py, src/ii_probe_run.py, src/probe_insert.py (selection and insertion as A14),
  analysis/probe_stats.py (hashes in amendment_20_hashes.txt).
- **Windows:** fold B, selection seed 0, 300 windows (as A14).
- **Data:** prepared in <work>/iiprobe_fB.

## Endpoints (fold B, once; scene-cluster bootstrap B = 10,000, seed 0)
- **P1:** kept(parked) - kept(4 m/s), real-shape cars, +3 s > 0, one-sided. Supported if the 95 % lower bound is > 0.
- **Descriptive:**
  - P2 (slope; expected at floor);
  - kept by condition and horizon;
  - where the mass went (stopped / continued position, corridor mass, progress of its centroid).
