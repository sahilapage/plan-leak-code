# Deviations from prereg/prereg.md (dated, with reason)

(none yet)

## 2026-10-01 (before any confirmatory result) - baseline perception fixes
1. SparseDrive box layout is [x, y, z, length, width, height, yaw, ...] (verified: index 3 mean 4.44 m vs annotated car
   length 4.42 m; yaw error median 1.1 deg). checkers.py and plcg.py had length/width swapped; fixed.
2. Shared-perception checkers (checkers.py) now use every SparseDrive class covered by the at-fault truth (vehicles,
   two-wheelers, pedestrians, barriers; cones excluded) instead of vehicles only (vehicles-only was an unfair strawman:
   at-fault collisions also involve pedestrians, cyclists and barriers). PLCG permanence / inflation stays defined on
   vehicles (as registered); its SWTC-danger component covers all classes.
3. Gate G2 triggered (Adv adversary well_rendered 0.065 vs real 0.593): mechanism claims rest on real data and within-Adv
   contrasts; the "observable & well-rendered" Adv subsets are reported with their (small) counts.

## 2026-10-01 18:50 - fine-tuning outcome under the registered rules
All five confirmatory runs (FT-full and FT-mask x seeds 0, 1; FT-real seed 0; fold 0) were STOPPED by the registered
stopping rule at 2,000 frames (vehicle recall @3 s on 200 held-out real frames 0.31-0.35 vs frozen 0.449 / 0.452 at two
consecutive checkpoints). Gate G5: no arm has >= 2 seeds that finished training -> H4 is NOT RUN (dropped, as
registered); fine-tuning is reported as one preliminary paragraph. Stopped checkpoints were evaluated (descriptive only).
No post-hoc learning-rate variant has been run.

## 2026-10-01 20:45 - amendment 1 outcome
E1 not supported (-0.12 pts [-0.56, +0.33]). Calibration drift: PG's real fold-B false-alarm rate 0.077 vs the 0.05
target fit on fold A (ECDF union of three scores); comparisons at the registered thresholds are reported as is.

## 2026-10-02 01:20 - amendment 2 outcome
P1 (attacker recall) and P2 (Adv collision) NOT SUPPORTED on fold B (Holm). Calibration transferred (real-B FA 0.034 at the
5% fold-A threshold). Seeds 1 and 2 (registered robustness) are reported as secondary when finished.

## 2026-10-02 13:45 - amendment 4 outcome
P1 (real-road AUPRC, WM+TH vs CV-det) SUPPORTED: +0.300 [+0.184, +0.399]. P2 (attacker AUPRC) NOT SUPPORTED: +0.018
[-0.054, +0.125]. No deviations from the registered protocol.

## 2026-10-02 14:40 - amendment 3 outcome
P1 (real vanish, perm - ctrl) and P2 (adversary vanish, perm - frozen) SUPPORTED. No deviations from the protocol.
A hallucination check (false vehicle mass) is added as an exploratory robustness analysis after seeing the result.

## 2026-10-02 18:10 - post-hoc robustness of amendment 3 (smear check), disclosed before amendment 5
results/smear_check.txt (held-out fold-A drives): the registered dyn-weight-10 model (perm) raises false vehicle mass
(p >= 0.2 farther than ~3 m from any true vehicle) from 112.5 (frozen) to 295.8 per frame-moment (false share 0.160 ->
0.267); near-vehicle mass 670 -> 812. Amendment 3's vanish reductions therefore partly reflect added probability mass
around vehicles. Amendment 5 registers the anchored dyn-weight-3 model with a no-hallucination (non-inferiority) primary.

## 2026-10-02 19:15 - amendment 5 outcome
P1 (real vanish, tperm3 - tctrl) and P2 (false-mass ratio tperm3 / frozen < 1.10) SUPPORTED. No deviations.

## Amendment 7 outcome (2026-10-03, fold B, run once; results/a7_decisions.txt)
- First launch crashed with CUDA OOM on the first frame (GPU shared with a DOME debug job); no fold-B output existed; relaunched
  unchanged (same hashes).
- P1 Adv chosen-path collision @5%: frozen 0.0276, tperm3 0.0223; diff -0.0054 [-0.0119, +0.0000], Holm p 0.055 ->
  superiority not supported; non-inferiority (upper < +0.5 pts) met.
- P2 real AUPRC: frozen 0.483, tperm3 0.355; diff -0.128 [-0.218, -0.024] -> non-inferiority (lower > -0.10) NOT met; worse.
- Registered interpretation applies: "the fix trades checker ranking for permanence" (limitation).
- Note: the frozen model's 5% threshold, calibrated on 514 held-out fold-A frames, realised 10.6% false alarms on fold B
  (tperm3 5.5%; CV-det 9.6%).

## Amendment 6 (DOME) - not registered
Drafted (prereg text not committed; analysis code d9ffcaa / dome_confirm.py) but not registered: the fold-A dev result of the
v2 permanence model failed the no-hallucination bound we would have registered (false-mass ratio 1.36 [1.32, 1.42] vs
1.10; results/dome_dev and runs/dome/stage4.log). DOME fold B was never opened for fine-tuned models.

## Documentation note (2026-10-03)
The training commands quoted in amendment_5_anchored_permanence.md omit `--weights ckpt/swtc/sw-tc-small.pth` (the frozen
SWTC checkpoint, required by src/wm_finetune_v2.py). The exact arguments of every registered fine-tune are stored next to
the checkpoint (ckpt/perm/<name>.json, e.g. tperm3.json) and match the registered hashes.

## Amendment 8 - pre-output bug fix (2026-10-04 ~11:45, before ANY fold-B output of the DOME transport head)
src/th_prep.py built the target map Y only for mode "path"; mode "pathdome" crashed at the first frame (np.packbits of
None). Fix: build Y for "pathdome" as for "path" (same Occ3D targets). No test logic changed. New hash of src/th_prep.py:
68182574e3c6ef4220209bca953ccd217ec8250c329acc952bcd2a1bdf44ce4f  src/th_prep.py

## Amendment 10 outcome (2026-10-04 ~23:50, fold B, run once; results/a10_improved.txt)
- P1 lateral, truth-centred: SWTC 0.182 -> 0.166, -0.016 [-0.034, +0.005], p 0.156; DOME 0.349 -> 0.283, -0.066 [-0.114,
  -0.005], p 0.042; Holm 0.156 / 0.084 -> NOT supported on either model.
- P2 non-regression (all moving): SWTC -0.008 [-0.017, 0.000]; DOME -0.051 [-0.071, -0.028] -> met on both.
- Wording: the amendment text says "largest threshold whose false mass does not exceed the reference"; the committed code
  (and its docstring, committed before registration) uses the threshold with the LARGEST FALSE MASS not exceeding the
  reference (i.e. the smallest such threshold). The code is binding; both readouts are read the same way.
- Dev-to-test shrinkage (SWTC lateral 0.155 -> 0.069 on 78 dev lateral movers vs 0.182 -> 0.166 on fold B): selection on a
  small development split overstated the gain.

## 2026-10-12 01:26 - A17 fold-B run: dump step failed (CUDA OOM), rerun with identical commands
The first A17 fold-B run completed the probe and hazard steps, then both dumps failed with CUDA out-of-memory (three of
our jobs shared the GPU); every later step failed on missing files, so no endpoint value was produced. The probe / hazard
outputs were not inspected. The dump and analysis steps are rerun unchanged, alone on the GPU (scripts in <work> on the
server: q_a17.sh, q_a17_dumps.sh). Dumps are written to <work>/soft and symlinked into runs/soft (root disk full).

## 2026-10-12 01:27-01:54 - queue sentinel error (A18 / A19 fold B), no endpoint affected
A stale "A17 DONE" line from the failed first A17 attempt released the A18 queue early (01:27), after A18's registration
(7f682d3). Its fold-B look-ahead runs completed once for released COME (01:36) and augmw (01:40); DOME's is still running
(the same process, not restarted). A second chain started the same way at 01:45: all its model runs (A18 and A19) hit CUDA
OOM before writing any output, so nothing was overwritten. Its statistics step ran on the two completed A18 files; that
text was not read and is superseded by the final chain (q_night2.sh), which waits for A17's result content, reuses the
completed A18 model outputs without rerunning, and runs A19 (registered d9bf7fd, no output existed before) once. The
user-approved cleanup (scripts/cleanup_oct12.sh) was run at the user's request at 01:52 and freed 20 GB.

## 2026-10-13 - A22 computed from the A23 runs (GPU rerun skipped); A23 DOME la rerun after crash
- A22 (plan-swap, registered a567ee7) was not rerun with src/planswap.py on fold B. Its endpoints are computed from the
  plan-swap judge inside the registered A23 DOME runs (src/blind.py imports ref_poses / ego_box_in from src/planswap.py;
  the same selection functions, n and per-batch seeds). On fold A the two produced identical numbers (pc 0.893; la
  0.718 / 0.595 / 0.395). Script: audit3/a22_from_a23.py.
- The A23 DOME la run died in the 2026-10-12 ~18:31 server crash at 668/819 before writing any output; it was rerun
  unchanged from scratch (q_resume_a23.sh).

## A26 (2026-10-15, before the fold-B result)
- P1 needs a paired interval (CV - self recall) that src/sw_cvjudge.py score does not print. audit3/a26_paired.py
  recomputes the identical per-window vetoes and bootstraps the paired difference by scene (B 10,000, seed 0). It was
  checked on fold-A dev only (+0.650 [+0.434, +0.835]) before the fold-B run finished.

## A25 (2026-10-15 ~22:00)
- The primary arm's registered evaluation was run early, in parallel with RPT-0.5 training (eval_rpt10_early); it is
  identical to the chain's later run. The placeholder RPT-0.5 row in a25_early.txt is CPC (ignore).
- src/swtc_cpc.py gained --offset-lr-mult (default 0.1 = registered behaviour), after RPT-0.5 training had started
  (that process uses the registered file). It is used only by the post-hoc dev runs (RPT-v2), which are not part of
  A25.

## A32 deviation 1 (2026-10-17, after the first policy output, before any rerun)
Observed: every policy (incl. the truth-trained one) picks a = 8 at every registered lambda (2 / 4 / 8 / 16); P3 is
uninformative (difference 0.000). Two design flaws, both fixed identically for every arm:
1. Penalty scale: progress values reach 8, so lambda <= 16 makes a crash worth ~2 grid steps. Fix (rule fixed
   before seeing any eval number): lambda* = the smallest lambda on a log grid (2 .. 4096) at which the TRUTH-trained
   policy's crash rate on the TRAIN windows is <= 2 %; every WM uses lambda* (and 0.5 x / 2 x as sensitivity).
   P3 is evaluated at lambda* instead of 8.
2. Crash geometry: the A29 label checks overlap at +1 / +2 / +3 s only; at extreme accelerations the ego passes
   through the lead between checks (truth crash share falls from a = 7 to a = 8). Fix: for A32 only, crash =
   penetration (ego front beyond the rear of a laterally overlapping vehicle that was ahead at t0) >= 1 m at ANY of
   the six 0.5 s steps, for truth, GT-CV and every model alike. A29 labels elsewhere are unchanged.
The belief test (P1 / P2, already read) is re-run on the corrected geometry and both versions are reported.
