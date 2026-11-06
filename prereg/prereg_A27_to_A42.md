# Preregistration log (tests A27 to A42 and later checks)

Each entry was committed to version control, with the SHA-256 of its analysis code, before the outputs it
tests existed. Entries are verbatim except for the anonymization for review noted in the top-level README.
Amendments 19 to 26 (earlier tests in the paper's App. F) are in the amendment_*.md files.

## A27 (registered 2026-10-16 ~02:30 local time, before any fold-B AUROC): threshold-free crash discrimination
Fold B real-car windows (swrh_f1; CV forecasts swcv_f1 = A26's), original SWTC. Score per candidate = max obstacle
probability in the candidate's ego box over +1 / +2 / +3 s (audit3/roc.py). Judges: self (forecast conditioned on the
candidate), cvref (one forecast under the past-only constant-velocity plan, candidates placed by geometry = A26),
oracle (one forecast under the logged plan; privileged). Positives = collide, negatives = logged. Paired scene-cluster
bootstrap (B = 2000, seed 0) of AUROC differences on the same resamples.
- P1: AUROC(cvref) - AUROC(self) > 0 with the 95 % lower bound >= +0.20.
- P2 (non-inferiority to the privileged oracle, margin 0.10): lower bound of AUROC(cvref) - AUROC(oracle) > -0.10.
- Descriptive: recall at 5 % / 10 % false alarm per judge; physics det_cv / det_mode operating points.
Fold A (DEV, seen): self 0.507, cvref 0.884, oracle 0.928 (exploratory, computed before this registration).

## A28 (registered 2026-10-16 ~04:50 local time, before training the model it tests): speed-blind crash discrimination
Protocol (audit3/fixed_acc.py, fixed_acc_extra.py): every fold-B real-car window (swrh_f1) gets the logged path plus a
FIXED extra acceleration a in 1.0 .. 3.5 m/s^2 (0.25 steps); kinematic labels vs the lead's logged track (crash =
ego-box overlap >= 1 m; safe = gap >= 2 m; else excluded); speed-blind AUROC = only crash / safe pairs with the same a
(pair-weighted). Scene-cluster bootstrap B = 2000, seed 0.
Models: original SWTC; RPT-A = src/swtc_cpc_v2.py --fold 1 (trained on FOLD-A frames only) --p-cf 1.0 --cf-mode
retime --offset-lr-mult 1.0 --steps 12000 (the RPT-v2 recipe, mirrored fold). Judge = self (each plan in its own
forecast). Comparators: physics (detections at constant velocity, longitudinal overlap depth), heuristics ego speed
and t0 gap (reported, not tested).
- P1: speed-blind AUROC(RPT-A) - AUROC(original) > 0, paired bootstrap lower bound > +0.15.
- P2: speed-blind AUROC(RPT-A) > 0.5 (lower bound > 0.5).
- Descriptive: RPT-A vs physics, vs the heuristics; naive (pooled) AUROC; A25 binary recall / FA at tau 0.2.
Fold A DEV (RPT-v2, trained on fold B): original 0.195, RPT-v2 0.803, physics 0.740, cvref 0.528, oracle 0.708,
ego-speed 0.742, t0-gap 0.930.

## A29 (registered 2026-10-16 ~07:45 local time, after the fold-A windows were drawn, BEFORE any forecast on them): decision value on a heterogeneous benchmark
Why: on the moving-lead windows (A27 / A28) the safe extra acceleration is nearly the same in every window, so a
PLAN-BLIND planner (same a everywhere) beats every judge (audit3/planner_baseline.py). Benchmark src/sw_hetero.py:
fold-A windows (seed 0, <= 8 per scene; 336 windows, 60 scenes: 150 free road, 144 moving lead, 42 stopped lead);
candidates logged + ACCS; truth vs EVERY annotated vehicle ahead (crash overlap >= 1 m, safe gap >= 2 m). Planner =
audit3/planner_select.py (most aggressive candidate scored below a global threshold). Primary metric: mean crash
reduction vs the plan-blind planner over mean chosen a in {2.0, 2.5, 2.75, 3.0, 3.25} ("gain"); scene bootstrap
(B = 300, seed 0). Judges: original SWTC self, RPT-v2 self (trained on fold-B frames), control-v2 self, oracle
(original logged forecast), cvref (original CV forecast), physics.
- P1: physics gain > 0 (lower bound > 0): detections carry decision value here.
- P2: original self gain < 0 (point) and < physics gain (point): the leaking judge is worse than ignoring it.
- P3: RPT-v2 self gain > original self gain (point) and > control-v2 self gain (point).
- P4: speed-blind AUROC orig self < 0.5 (inverted, as on the moving-lead windows).
Fold B (held out) will follow with RPT-A (A28's model) under the same predictions P2-P4 (RPT-A for RPT-v2, no control).

## A30 (registered 2026-10-16 ~13:15 local time, before any forecast): plan-marginal judging
Theory: for agents the ego cannot affect, p(world | h, do(plan)) = p(world | h); a log-trained plan-conditioned model
estimates p(world | h, plan observed) instead (biased by the log coupling). Estimator (src/het_marg.py): the candidate
is judged in forecasts under K = 7 past-only prior plans (constant acceleration -2 .. +2 m/s^2 from the past speed),
score = mean over k. A29 benchmark, both folds, SWTC original and RPT (RPT-v2 on fold A, RPT-A on fold B).
- P1: gain(plan-marginal) >= gain(single CV reference) for both models (pooled, point estimate).
- P2: gain(plan-marginal, original weights) > gain(self, original) with the paired 95 % lower bound > 0.
- Descriptive: plan-marginal vs physics.

## A31 (registered 2026-10-16 ~20:00 local time, before any model is trained): causal-in-time plan conditioning
Theory: a logged multi-second ego plan is partly a consequence of the other agents' futures (the driver kept reacting
during the plan), so conditioning on the whole future plan is conditioning on a descendant of the outcome (time-varying
confounding). The g-formula estimator (one-step model fed the ego plan only up to L steps beyond the current step,
rolled out on its own predictions) identifies p(world | h, do(plan)) from plain logs when L is within the driver's
reaction delay. Instrument (src/obj_wm.py): object-level transformer WMs on annotated nuScenes tracks (train split;
10 % of train scenes for model selection), identical architecture / data / budget (20k steps, seed 0); they differ
only in plan conditioning: step0 / step1 / step2 / step3 / step6 (rolled out), chunk (whole 6-step plan at once),
marg (no plan). Test = nuScenes val (never used for training or selection).
- P1 (leak removed): lead-car beta at +1 s (median over leads x counterfactual plans a = +2 / -1.5 m/s^2): step0 < chunk,
  paired scene-bootstrap 95 % upper bound of (step0 - chunk) < 0, and step0 point <= 0.25.
- P2 (reaction kept, not a marginal): follower beta at +2 s for step0 has 95 % lower bound > 0.
- P3 (decisions): A29 benchmark (both folds pooled, 653 windows / 120 scenes; same labels, gain vs plan-blind):
  gain(step0 self) - gain(chunk self) > 0 with paired lower bound > 0.
- Descriptive: lead beta(+1 s) vs L (dose curve; where it departs from 0), step0 / marg vs same-input GT-CV physics
  and detector physics; rollout ADE under the logged plan (accuracy guard, reported for every model).
Deviation rule: if a model fails to train (val ADE worse than GT-CV), fix and report; no change to the predictions.

## A32 (registered 2026-10-17, before any belief / policy number exists): the leak law predicts reward hacking
Theory: a plan-conditioned model with lead leak beta shifts the lead by beta x the plan's extra ego displacement, so
the apparent closing under an extra constant acceleration a is (1 - beta) of the real one; the acceleration it
believes safe is the real limit / (1 - beta). A policy optimised inside it becomes 1 / (1 - beta) more aggressive
(unbounded at beta -> 1). Instruments: the A31 object WMs (beta measured in a31_leak.txt: step0 0.00, marg 0,
step6 0.38, step3 0.41, step2 0.44, chunk 0.68, step1 0.74 at +1 s) + GT-CV physics; frozen SWTC / DOME scores on
the A29 grid. src/obj_rl.py.
Belief test (A29 windows with a lead, both folds pooled; extra accel grid 0 .. 8 m/s^2, step 0.25, along the
logged path): a_believed = largest grid a before the model's first predicted crash (predicted overlap >= 1 m);
a_true = same on the annotated tracks (truth). Ratio r = a_believed / a_true over windows with a_true in (0, 8).
- P1: median r chunk >= 1.5 and median r step0 in [0.8, 1.25].
- P2: across the 7 object WMs, Spearman(median r, 1 / (1 - beta_lead+1s)) > 0 (point), and median r of chunk >
  step0 with paired scene-bootstrap lower bound > 0.
Policy test (train split windows: near-straight logged path, ego >= 3 m/s, logged path clear of annotated vehicles):
softmax policy over the grid, MLP of t0 features (ego speed, in-lane lead gap / speed / presence), maximising
E[a - lambda x predicted crash] inside each WM (lambda in 2, 4, 8, 16); evaluated on the A29 windows on truth.
- P3: at every lambda, the real crash rate of the chunk-trained policy > step0-trained, paired scene-bootstrap lower
  bound > 0 at lambda = 8 (primary).
- Descriptive: believed vs real crash ("sim-to-real gap") per WM; truth-trained and GT-CV-trained policies as
  references; SWTC self / cvref / cvmarg and DOME self / cvref policies trained on fold A A29 scores, tested on fold B
  truth (and swapped), grid 0 .. 3.5.
A32 amendment (2026-10-17, before any A32 belief / policy output was read): the training-free plan-marginal judge
(A30) as the reward model. pm_<m> = median over K = 7 past-only prior plans (constant accel -2 .. +2 from the past
speed, straight; src/het_marg.py PRIOR / mplan) of the candidate's overlap depth in the forecast under prior plan k,
for m = chunk and step1. Added to the same grid / belief / policy pipeline.
- P4: median belief ratio pm_chunk in [0.8, 1.25] (the fix removes the inflated belief).
- P5: real crash of the pm_chunk-trained policy - step0-trained <= +0.01 at lambda = 8 (point), i.e. the frozen
  leaky model with the training-free query trains a policy about as safe as the causal model.

## A33 (registered 2026-10-17 night, before any A33 number exists): reward hacking inside the real 3D camera WM
Same question as A32 on frozen SWTC (camera occupancy WM): a policy trained with a WM judge as its reward.
Data: the A29 windows (both folds), the existing per-candidate judge scores (src/sw_hetero.py / het_marg.py /
het_occ.py tables, candidates a in {0} + ACCS = 0 .. 3.5; a candidate absent from a table is not selectable),
truth = A32 penetration geometry on annotated tracks (grid_a29_v2), features = A32 past-only t0 features.
Reward = a - lambda x p, p = the judge score read as a collision probability (0 for a = 0); physics / truth p = 1 if
overlap / penetration >= 1 m. Softmax MLP policy trained on one fold, tested on the other, both directions pooled;
lambda swept 2^0 .. 2^8 in half-octaves, 3 policy seeds; real crash interpolated at MATCHED mean progress.
Judges: SWTC self (orig), SWTC cvref, SWTC plan-marginal (cvmarg_orig), RPT self / plan-marginal, DOME self / cvref,
COME, physics, truth.
- P1 (primary): at matched progress 2.0 m/s^2, real crash(SWTC self) - real crash(SWTC plan-marginal) > 0 with
  scene-bootstrap lower bound > 0.
- P2: at matched progress 2.0, SWTC self's believed collision rate (mean p at the chosen a) < half its real crash
  rate (sim-to-real gap); the plan-marginal judge's believed / real ratio is closer to 1 than self's.
- Descriptive: DOME self vs cvref, COME, physics, RPT; progress 1.5 / 2.5.

## A34 (registered 2026-10-17, before computing): does fading explain the remaining gap to physics?
On the A29 lead windows (moving / stopped lead with a full track), from SWTC's EXISTING plan-marginal forecasts
(marg_orig, prior plan a = 0 i.e. the past-speed CV plan, k = 3): the real lead is "kept" at +h if the max vehicle
probability inside its true box (in the forecast's frame) is >= 0.2 at +2 s AND +3 s; else "faded". Gain vs
plan-blind (src/sw_hetero.py gain) per stratum for the SWTC plan-marginal judge and physics; scene bootstrap.
- P1: physics - plan-marginal gain is larger in faded windows than in kept windows (difference of differences > 0,
  point estimate), and in kept windows the paired physics - plan-marginal difference has a 95 % CI including 0.
Exploratory follow-ups (not registered): the same split for SWTC self.

## A35 (registered 2026-10-17 ~20:15 local time, after generating the simulated logs and their coupling, BEFORE any model is trained on them): ground truth for the leak and the fixes
A world whose causal effects are known (src/causal_sim.py): straight road, 10 Hz, logged at 2 Hz in the obj_wm window
format. Lead exogenous (traffic factor + random hard braking); ego = the logging driver (IDM on the lead with reaction
delay 0.5-1.0 s, brake-light anticipation, noise); follower reacts to the ego (IDM, delay 0.4-0.8 s); two adjacent cars
share the lead's traffic factor (common cause, no effect); 8 parked cars. do(ego plan) from t0 re-simulates the
follower only, so the TRUE plan effect is 0 for lead / adjacent / parked and exact for the follower. Train 1200
episodes (seed 0, 45,595 windows), test 150 episodes (seed 1, 5,700 windows), robustness test with slower drivers
(delay 0.8-1.4 s; seed 2, 5,672 windows). Log coupling (+2 s, train): lead history-aware +0.52 / history-blind +0.91,
follower +0.59 / +0.93, adjacent +0.27 / +0.52, parked 0.00. Models: obj_wm chunk / step0 / step1 / marg, seed 0
(seeds 1-2 for chunk / step0 / step1 as robustness), trained on the sim train split exactly as A31 (20k steps);
fixes from src/gcomp.py: gc_<m> (test-time g-computation of the frozen model), pm_<m> (plan-marginal positions).
Leak beta = median over agents x counterfactual plans (logged + 0.5 a t^2, a = +2 / -1.5) at +2 s unless stated;
episode-bootstrap 95 % CI. Judge: candidates a in {-4..+4} (step 1); crash = penetration >= 1 m at any 0.5 s
keyframe, front (ego into lead) and rear (follower into ego), predicted vs truth under do().
- P1 (law): chunk lead beta (+2 s) lies in [0.8 x history-aware, 1.1 x history-blind] = [0.42, 1.00] while the true
  effect is 0; chunk adjacent beta > 0 (lower bound) while the true effect is 0.
- P2 (causal model is right, not plan-blind): step0 |lead beta| <= 0.10 and step0 follower beta within +-0.15 of the
  median TRUE follower beta (+2 s).
- P3 (training-free g-computation): gc_chunk and gc_step1: |lead beta| <= 0.10 and follower beta within +-0.15 of the
  true follower beta.
- P4 (plan-marginal is wrong exactly for reacting agents): if the median true follower beta (+2 s) >= 0.20, the rear
  judge error rate (false alarms + misses over braking candidates a <= -2) pm_chunk - gc_chunk > 0 with episode-
  bootstrap lower bound > 0. (If the true follower beta < 0.20 the test is reported as uninformative.)
- P5 (belief / reward hacking): median believed / true largest safe extra accel on lead windows: chunk >= 1.25 and
  step1 >= 1.25; gc_chunk and gc_step1 in [0.8, 1.25].
- Descriptive: everything above on the slow-driver test set and model seeds 1-2; marg; logged-plan ADE.

## A36 (registered 2026-10-17 ~20:15 local time, before any g-computation output on held-out data): test-time g-computation of the frozen nuScenes object WMs
Frozen A31 checkpoints (chunk, step1; seeds 0 / 1 / 2), wrapped by src/gcomp.py: gc_<m> = at each 0.5 s step the
model sees the ego only up to now, its next-step prediction is averaged over the 7 past-only prior continuations
(A30 PRIOR, uniform), the step is kept and the candidate's next ego step appended; nothing is trained. Test = nuScenes
val (A31 leak protocol) and the A29 windows (A32 grid / frontier protocol, train-split grid for policy training).
- P1: lead beta (+1 s) median <= 0.15 for gc_chunk and gc_step1, every seed (vs chunk 0.61-0.68, step1 0.66-0.74).
- P2 (reaction kept): follower beta (+2 s) of gc_chunk and gc_step1 (seed 0) has 95 % lower bound > 0.
- P3 (reward hacking removed): A32 frontier (3 WM seeds x 3 policy seeds), matched progress 3.0: step1 - gc_step1 > 0
  with lower bound > 0, and |gc_step1 - step0| <= 0.010 (point).
- P4 (the standard safeguard is blind; audit3/ens_blind.py on the EXISTING A32 grid_a29_seeds): among plans the
  3-seed step1 ensemble mean calls safe, AUROC of ensemble std for real crash vs safe <= 0.65, and AUROC of the
  label-free self vs plan-marginal discrepancy exceeds it by >= 0.10 (point).
- Descriptive: belief ratios of gc columns, logged-plan ADE (accuracy cost of feeding back own predictions), chunk
  ensemble for P4, pm_ positions vs gc_ on follower beta.

## A37 (registered 2026-10-18 ~01:45 local time, after two pilot windows, BEFORE any confirmatory generation is looked at): the plan leak in a frozen 4D video world model
Model: OmniNWM (HF Arlolo0/OmniNWM, released nuScenes inference config: 6 cameras, 33 frames at
12 Hz, 448 x 800, panoramic RGB + metric depth + semantics), frozen, one A100 80 GB, 25 sampling steps (pilot 1, logged
plan, same seed: 25 vs the shipped 50 steps differ by median 0.13 m in depth and agree on 97-99 % of semantic pixels),
seed 0 for every plan (common random numbers). The ego plan enters only where the logged trajectory does (per-frame
relative ego pose -> ray maps + traj_gt; src/a37_omni.py). Windows (src/a37_windows.py + src/a37_list.py): all 150
nuScenes val scenes; a real lead (moving or stopped) whose centre is within 1.0 m of the ego lane centre, 2-25 m ahead,
heading within 30 deg; near-straight logged path (any ego speed); collide = logged path + the smallest constant extra
acceleration a* <= 4 m/s^2 putting the ego front >= 1 m into the annotated lead box at +1.0 .. +2.5 s; up to 3 windows
per lead instance >= 3 s apart; all 6 x 33 clip images on disk -> 64 windows (29 leads); fixed random order (seed 0);
the first N = 30 (pack/main_omni.txt). Pilots (excluded): 6d0cc921 (lead mostly in the adjacent lane: motivated the
1.0 m rule) and 10cdc73a (lead instance not in the main list). Plans: logged, collide, pm0, pm-1.5 (past-only: straight
from the past speed at constant 0 / -1.5 m/s^2).
Scoring: src/a37_score.py as committed with this entry. Generated dynamic points = semantics car/person/bicycle,
1 <= depth < 25 m (the released reward uses 0 < depth < 25 m), three front cameras, stride 2. Lead = the vehicle
component nearest the annotated lead at t0, tracked by nearest centroid; rear x in the t0 frame. beta = (rear_collide -
rear_logged) / ds at t* = the first frame of true contact of the collide plan (ds >= 1 m, else dropped; pilot 2
 showed a 3 m-gap rule leaves slow creep-ins undefined); beta_realized uses the realized
progress difference instead of ds. Realized progress = chained frame-to-frame registration of static (object /
vegetation) points. Judges: released (OmniNWM's collision rule as shipped, any dynamic point), self (frame-consistent,
>= 10 points in the ego box at any frame), pm (>= 10 points of a past-only generation in the candidate's ego box; share
of the 2 past-only generations >= 0.5 at any frame). Scene-cluster bootstrap (2000).
Gates (on the confirmatory windows; else P1-P3 are reported as uninformative): G1 controllability (realized progress
difference / ds at t*) median >= 0.3; G2 logged-plan tracked lead rear error vs annotation at ~1 s median <= 3.0 m.
- P1 (the leak exists in a 4D model): median lead beta >= 0.30 with bootstrap 95 % lower bound > 0
  (truth ~0: a lead does not react to the car behind it within 2.5 s; exactly 0 for stopped leads).
- P2 (its own reward is blind to the crash): the released collision rule misses >= 30 % of collide plans, and the
  frame-consistent self judge misses >= 30 %.
- P3 (training-free fix): pm miss rate <= self miss rate - 0.20 with paired bootstrap upper bound < 0, and pm false-alarm
  rate on logged plans <= self false-alarm rate + 0.10 (point).
- P4 (Vista arm, secondary; same windows, src/vista_leak.py detector readout, 25 steps = SVD default): median lead beta
  >= 0.30 if the readout passes its real-keyframe validation (median |error| <= 3 m); otherwise uninformative.
- Descriptive: released vs self (the released rule places the plan pose (t0 frame) on ego_t-frame points), stopped vs
  moving leads, beta_realized, the generated videos.
- Stop rule: RunPod spend estimate <= $72 (hard user cap $80) and 2026-10-19 12:00 UTC, whichever first; windows run in
  the fixed order, so a budget stop truncates N without selection.
- Amendment 1 (2026-10-18 ~02:30 local time, from pilot-2 data only; no confirmatory output opened; 1 main generation existed):
  judge point threshold 10 -> 50 points (stride 2) for self and pm. Pilot 2: lens-adjacent artifact clusters reach
  9-13 points in the ego box on safe frames (pm judge on the logged plan hit exactly 10), while a real vehicle in the
  box gives 181-9,890 points. Pilot-2 observation recorded: the past-only constant-speed generation (pm0) also moves
  the stopped lead away (rear 10.6 -> 15.5 m) while the braking one (pm-1.5) keeps it in place (10.3 m).

## A37b (registered 2026-10-18 ~02:50 local time, before any dose generation): dose-response of the leak in OmniNWM
Same model / settings / readout as A37. Plans dA = logged path + constant extra acceleration A along the t0 heading
(never reversing): d-2 (brake), d1, d3, plus A37's collide and logged. OmniNWM: the first 6 windows of the A37 main list
(pack/dose_omni.txt); Vista arm: all 30 A37 windows (same plans, 25 steps). Scoring src/a37b_dose.py: at t_m = min(1.5 s,
first true contact of the plan), d_rear = generated lead rear (plan) - (logged), d_s = plan - logged progress; per-window
through-origin slope over plans with |d_s| >= 0.3 m (>= 3 plans). Truth slope 0.
- P1 (the leak is a dose-response law, not a binary OOD failure): median slope >= 0.3, scene-bootstrap lower bound > 0.
- P2 (bidirectional, which an out-of-distribution account does not predict: braking is in-distribution): the braking plan
  pulls the generated lead towards the ego (d_rear < 0) in >= 60 % of windows.
- Descriptive: pooled slope and R^2, per-window curves; the Vista arm by the A37 P4 readout.
- Budget: runs after A37 (OmniNWM + Vista), Vista dose first; the $74 guard truncates the fixed order if reached.

### A37 amendment 2 (2026-10-18 ~06:00 UTC): disclosed interim look, estimator replacement, N and budget reallocation
- **Interim look.** At the PI's request, the registered scorer was run on the first 10 confirmatory windows (40 of 120
  generations), before the run finished. Interim (registered metrics): beta median 1.05 [0.87, 1.24] (n 9); self miss
  10/10; pm miss 0.20 [0.00, 0.50], pm false alarm 0.10; released miss 0.00 with false alarm 0.60; readout error
  2.5 m @1 s; controllability (registered estimator) 0.39 [0.20, 0.77].
- **G1 estimator invalid.** The registered progress estimator (chained occupancy-grid correlation of static points)
  recovers only 0.09-0.45 of the REAL ego progress on logged plans with a moving ego (e.g. 9.7 m of 24.8 m), i.e. it
  is biased toward zero (amorphous "object"/vegetation clutter, night scenes). Replaced by PnP odometry
  (src/a37_pnp.py: ORB matches on static classes, 3D from the generated depth of frame A, 2D in frame B,
  solvePnPRansac, chained 0.25 s hops), validated on LOGGED plans against the real progress (e.g. 12.55 vs 11.26 m,
  3.50 vs 3.58 m, 0.00 vs 0.00 m). Disclosure: the collide-plan values were computed in the same pass as the
  validation. Night windows with < 30 PnP inliers per hop are reported as unmeasurable. New secondary metric
  beta_world = (d_gap + d_s_hat) / d_s_hat (lead displacement relative to the generated static world).
  G1 is evaluated with the PnP estimator; the registered estimator's value is reported alongside.
- **N reduced 30 -> 20 OmniNWM windows** (first 20 of main_omni.txt, unchanged order) to fund (a) A37b OmniNWM dose
  on the first 10 main windows (was 6), (b) seed-1 replicates (4 plans) of the first 5 main windows. Motivation:
  budget reallocation to the dose-response (the confounding-vs-OOD test) and seed robustness. Disclosure: this was
  decided after the interim look; all 20 windows are reported, P1-P3 thresholds unchanged; Vista stays at 30.

### A37 amendment 3 (2026-10-18 ~06:35 UTC, before any dose / Vista / seed-1 / 50-step generation exists)
Queue changes (/root/chain4.sh, replaces chain.sh tail, chain2.sh, chain3.sh):
- OmniNWM main ends after window 18 of main_omni.txt (windows 19-20 repeat scenes 0907 / 0781 already in the set).
  Confirmatory N = the first 18 windows minus none (pilots are not in the list).
- A37b OmniNWM dose list = the windows among the first 18 with v0_past >= 1.0 m/s (braking is a no-op for a stopped
  ego): bd608be2 d036b91e a131cf4a 8d67c402 01c84516 d53d5eb7 440691c4 bb08ec78 918b8f22 (9). A37b P1/P2 unchanged.
- Vista: plans logged, collide, d-2, d1, d3 on the 30 windows of main_vista.txt (pm0 / pm-1.5 dropped: the
  plan-marginal judge needs metric depth, which Vista does not produce). P4 unchanged.
- 50-step check (descriptive): windows 1-2 (ad26ae1c, bd608be2), logged + collide at the shipped 50 steps; report beta
  and the self / pm verdicts at 50 vs 25 steps.
- Seed-1 replicates: first 5 windows x {logged, collide, pm0, pm-1.5}; report beta and judge verdicts per seed.
**A37c (registered here): does the leak flip a planner's decision?** Dose windows (9) x candidates {d-2, logged, d1,
d3, collide}. TRUE crash of a candidate: its ego front penetrates the annotated lead rear by >= 0.5 m at any frame
<= 2.75 s (lead boxes interpolated from 2 Hz). Judges as registered (self: own-world ego box >= NMIN points at any frame;
pm: plan-marginal share >= 0.5 at any frame; pm worlds shared by all candidates). Planner(J): pick the candidate with
the largest final progress that J calls safe; if J calls none safe, pick d-2. Outcome per window: picked plan truly
crashes (0/1) and its progress.
- PL1: among windows with >= 1 truly crashing candidate, Planner(self) picks a truly crashing plan in >= 50 %.
- PL2: Planner(pm) crash-pick rate <= Planner(self) - 0.30 (point), and Planner(pm) picks a plan with progress >= the
  logged plan's in >= 50 % of windows (the fix does not simply freeze).
- Descriptive: per-candidate self / pm verdicts vs truth (confusion tables); OmniNWM's released rule as a third judge.

### A37 amendment 4 (2026-10-18 ~07:40 UTC, before any dose / Vista / seed-1 / 50-step generation exists)
- A37c planner test extended from the 9 moving-ego dose windows to all 18 main windows: the 9 stopped-ego windows
  (v0_past < 1 m/s: ad26ae1c 6b3c8514 fd534bf0 e30706e0 b4799bad e3722219 6a85bfb2 eec42a16 fbbad630) get d1 and d3;
  for them d-2 is the logged plan (braking from standstill is a no-op). PL1 / PL2 unchanged, evaluated on all 18 and
  reported separately for moving / stopped egos. A37b (dose slope, braking) stays on the 9 moving-ego windows.
- Seed-1 replicates reduced to {logged, collide} on the first 5 windows (beta and the self judge per seed; the pm judge
  for seed-1 candidates uses the seed-0 plan-marginal worlds, which do not depend on the candidate).
- Queue order: OmniNWM dose (moving) -> Vista -> OmniNWM d1,d3 (stopped) -> seed-1 -> 50-step (/root/chain5.sh).

### A37d (exploratory, registered 2026-10-18 ~08:10 UTC, before any generation): WHEN does OmniNWM read the plan's intent?
The plan enters OmniNWM only through camera ray maps, at every denoising step. Ray-map schedule (src/a37_omni.py
swK / wsK): candidate = collide; swK feeds the past-only pm-1.5 ray maps for the first K of 25 steps and the collide
ray maps afterwards; wsK the reverse. Windows (stopped ego and lead, daytime, PnP measurable): ad26ae1c fd534bf0
6b3c8514. Plans sw5, sw10, sw15, ws10 (seed 0, 25 steps), compared with the existing collide / pm-1.5 / logged.
Measures at the collide plan's true contact frame: realized ego advance (PnP odometry; parked-car anchored where
available), lead ego-frame gap change vs logged (beta_world), self judge (crash visible in the model's own world).
- D1 (a training-free generative fix exists): for some K, in >= 2 of 3 windows the realized advance is >= 0.6 x
  collide's AND beta_world <= 0.3 AND the self judge fires.
- D2 (mechanism, descriptive): the realized advance and the lead's beta_world as functions of K (and ws10): whether
  the model decides the ego's motion and the agents' motion at different noise levels.
Exploratory (n = 3): a positive D1 would be confirmed on held-out windows before any claim.

## Exploratory A37e (post-hoc, logged 2026-10-18, after the 15-window look; not confirmatory)
Prompted by a design critique: is the plan-marginal judge informative, or does it return the prior plan we feed?
(a) prior echo: beta_prior = (lead rear in pm0 world - in pm-1.5 world) / (ego pm0 - ego pm-1.5) at t*,
windows with |ego difference| >= 0.5 m; per-prior judge verdicts. (b) constant-velocity baseline: lead keeps its
annotated t0 speed; lead-rear error at t* and collide/logged verdicts. Scripts src/a37e_prior_echo.py,
src/a37e_cv_baseline.py. Reported as exploratory in the paper.

## A37f (registered 2026-10-18 ~10:55 UTC, before any A37f generation): is OmniNWM's leak forced by single-frame conditioning?
Observation that motivates it (code, not data): every OmniNWM training stage uses condition_config = {i2v_head}: the
model starts from ONE real frame and never sees motion history; multi-round rollouts also hand over one frame. From one
frame the lead's velocity is barely identifiable, so the logged-plan coupling it can use is the history-BLIND one
(nuScenes train, +2 s: lead 0.96 [0.89, 1.03]; history-aware residual 0.67 [0.46, 0.84]); A37's measured 1.03 sits on
the blind bound. Intervention: give OmniNWM the real motion history. Plans (src/a37_omni.py): h-logged / h-collide see
the real frames 0-4 (0.33 s; 2 latent frames of the causal VAE) via the cond mask on both latents + RGB-latent
replacement at every denoising step; the plan equals the log through frame 4, then h-collide adds a* (t - t4)^2 / 2;
s-collide = the same shifted plan with the stock single-frame conditioning (matched control; paired with logged).
Windows: 01c84516, 8d67c402 (moving leads), fd534bf0 (stopped queue). Seed 0, 25 steps. Scored with src/a37_score.py
(beta at the true contact of the shifted collide plan, self judge) on renamed copies (h-logged -> logged, etc.).
- P1: median lead beta (h-collide vs h-logged) <= 0.80 AND median beta (s-collide vs logged) >= 0.90.
- Descriptive: per-window beta, self judge on h-collide, generated gap at contact; fidelity of frames 1-4.
- Extension rule (fixed now): if P1 holds, h-logged / h-collide on d53d5eb7, a131cf4a, 440691c4, bd608be2, 6b3c8514
  (no new s-collide; control = A37 collide / logged), same prediction for the pooled 8; budget permitting.
- A37d (sw / ws) is dropped before any generation (value superseded: the plan is the only motion source); the 50-step
  check is dropped (pilot 1 already: median depth diff 0.13 m, semantic agreement 0.97-0.99).

## A37g (registered 2026-10-18 ~11:40 UTC, before any Vista generation): motion history inside its training distribution
Vista's own sampler conditions later rounds on 3 frames (prediction_cond_mask [0, 1, 2], its "dynamic prior"), so a
3-frame start is in distribution (unlike OmniNWM, A37f). Plans (src/vista_leak.py): h-logged / h-collide condition on
the REAL t0 frame + the next two 12 Hz CAM_FRONT sweeps (initial_cond_indices [0, 1, 2]); plan = log through frame 2,
then h-collide adds a* (t - t2)^2 / 2; s-collide = same shifted plan, stock single frame (paired with logged).
Windows: main_vista.txt (30). Readout: the registered Vista read stage (detector, box-height distance), lead beta
as for Vista P4 (amendment 3) at the last frame before true contact.
- Precondition: Vista single-frame lead beta (s-collide vs logged) median >= 0.5; else reported as uninformative.
- P1: median lead beta (h-collide vs h-logged) <= median beta (s-collide vs logged) - 0.2, paired window bootstrap
  upper bound of the difference < 0.

## A38 (registered 2026-10-18 ~11:40 UTC, before any test-episode closed-loop output): closed-loop planning in the A35 world
src/a38_closed.py. World = src/causal_sim.py (A35): lead / adjacent / parked replay the log (exogenous), the follower
reacts to the planner's ego. Planner (fixed on smoke episodes, seeds 7 / 8 only): every 0.5 s, constant-accel plans
{-8, -6, -4 .. +4} over 3 s from the current speed; kept if predicted front gap >= 1 m + 1.0 s x plan speed and rear
gap >= 1 m at every 0.5 s keyframe; kept plan closest to clip(0.5 (v0_e - v), -2, 2); none kept -> front-first
fallback. Data regenerated with the A35 generator (sim_train 1200 ep seed 0, sim_test 150 seed 1, sim_slow 150 seed 2
slow); coupling must reproduce A35 (lead aware +0.52 / blind +0.91) or the run is void. Models retrained with the A31 /
A35 recipe on CPU (chunk, step0, step1, marg; seed 0; 20k steps; the A35 checkpoints sit on the 4090) -> report val
ADE. Judges: chunk, step0, step1, marg, gc_chunk, gc_step1, pm_chunk, cv, oracle (true do() future); logged driver.
Metrics per episode on sim_test (episode bootstrap 2000): front / rear crash, TTC < 1.5 s steps, tailgating share,
hard brakes, mean speed, min gap.
- P1 (the leak makes closed-loop driving unsafe): TTC-violation steps per episode chunk - step0 > 0 with lower bound > 0,
  and front-crash rate chunk >= step0 (point).
- P2 (training-free fix): TTC violations gc_chunk - chunk < 0 with upper bound < 0, and gc_chunk closes >= half of the
  chunk - step0 gap (point), at mean speed >= 0.95 x step0's (not bought by crawling).
- P3 (it is the leak, not the architecture): step1, which does not leak in this world (A35 mechanism), has TTC
  violations within +-50 % of step0's (point).
- Descriptive: pm_chunk / marg / cv / oracle / logged driver; sim_slow; secondary planner with 0.5 s headway.

## A37h (registered 2026-10-18 ~11:20 UTC, before any A37h generation; CONDITIONAL on A37g P1): video g-computation
src/vista_leak.py g-logged / g-collide: Vista rolled out in 4 rounds of 0.5 s; round r conditions on the 3 frames at
r/2 s (+1/12, +2/12) (round 0: real; later: its own kept frames), sees the candidate plan only up to the end of the 6
frames it keeps (r/2 + 8/12 s), continued past-only at constant velocity, keeps frames 3..8 (A36's g-computation with
one round of lookahead, the minimum for the camera to follow the candidate). Run on the 30 main_vista windows only if
A37g P1 holds (a 1-window pilot of h- / g- plans runs first as a code check; it is not analysed).
- P1: median lead beta (g-collide vs g-logged) <= 0.25 and <= median beta (h-collide vs h-logged) - 0.2.
- P2 (the model's own video shows the crash): share of windows whose generated lead camera distance at the true
  contact frame is <= 4 m: g-collide >= s-collide + 0.3.

### A38 exploratory follow-ups (logged 2026-10-18 ~22:40 local time (17:10 UTC), AFTER the registered P1 / P3 outputs of chunk, step0, step1, marg were seen; gc_ / pm_ / cv / oracle outputs not yet written; not confirmatory)
Seen: sim_test TTC violations 0.00 for every planner (floor: the 1 s headway + 0.5 s replanning absorbs the leak),
so P1 is NOT SUPPORTED and P3 is uninformative (0 / 0). Also seen: chunk brakes hard (a <= -3) +5.9 [+5.5, +6.3] times
per episode more than step0 (all three test sets), min gap -0.83 m, mean speed +0.08 m/s; 0.5 s headway: TTC
chunk - step0 +0.94 [+0.45, +1.45].
- (a) mechanism (audit3/a38_diag.py, chunk and step0 on sim_test, headway 1.0): at the judge's hard-brake decisions,
  the oracle on the same state keeps a gentler plan (a >= -2) in >= 50 % for chunk ("phantom braking": moderate brakes
  look unsafe because the lead is predicted to follow the ego's braking), lower share for step0.
- (b) headway frontier: A38_HEADWAY 0.25 / 0.75 / 1.5 for chunk, step0, step1, marg, cv, oracle on sim_test (plus the
  existing 0.5 / 1.0); per judge TTC violations, hard brakes and min gap vs mean speed; chunk - step0 at matched mean
  speed (linear interpolation across headways). Expectation: chunk has more hard brakes at every headway and at least
  as many TTC violations at matched speed. gc_chunk / pm_chunk added to the sweep if their registered runs finish in time.
- (c) logged 2026-10-18 ~23:00 local time, after (a) chunk output (phantom 0.70) and before any gc_chunk closed-loop output:
  audit3/a38_diag.py with gc_chunk (same chunk weights, test-time g-computation), headway 1.0, sim_test.
  Expectation (the closed-loop form of the fix): gc_chunk's phantom hard brakes per episode <= 0.25 x chunk's, and its
  judged front-safe share per candidate ordered like the oracle's (monotone in the plan's acceleration).

## A37 controls on the final 18 windows (logged 2026-10-18 ~23:40 local time, before any control2 / vanish output on the 18; descriptive tests with fixed readouts)
Code as committed: src/a37_control2.py (detector-tracked vehicles, beta_world with PnP ego advance), src/a37_vanish.py.
Inputs: results/a37_final (scores18.csv, pnp18.csv, windows_merged.json), OmniNWM outputs synced 16:55 UTC.
- C1 (selectivity: the leak is the plan-coupled lead, not a global scene shift): median beta_world lead - median
  beta_world side_stopped (parked / stopped off-lane vehicles, truth 0) >= 0.5; side_stopped median |beta_world| <= 0.3.
- V1 (two ways to clear the path): share of should-be-visible vehicles that vanish under collide - under logged
  >= 0.10 -> "flee vs delete" panel in the paper; otherwise one line ("the path is cleared by moving, not deleting").
- (d) logged 2026-10-18 ~23:55 local time (4090 free overnight), before any of these outputs: the headway frontier (b) is
  extended to gc_chunk, gc_step1, pm_chunk at headways 0.25 / 0.5 / 0.75 / 1.0 / 1.5 (plus chunk, step0, cv, oracle),
  run on the 4090 GPU with the pod's A38 checkpoints and sim_test (same code; device cuda). The registered A38 run on the
  pod CPU stays the confirmatory one; if device numerics differ, both are reported. Expectation: at every headway
  gc_chunk's hard brakes are within 2x step0's and its TTC violations <= chunk's; at matched mean speed gc_chunk lies
  on or below step0's TTC-vs-speed curve.

## A39 (registered 2026-10-19 ~00:35 local time, before any A39 closed-loop output; after the A38 exploratory results): closed loop on REAL traffic
src/a39_real_closed.py as committed with this entry. World: all nuScenes val scenes, the other vehicles replay their
annotated tracks (non-reactive log replay, as nuPlan's non-reactive closed loop), the ego follows its logged path
laterally and the planner (A38's, unchanged: candidates -8 .. +4 m/s^2, MARGIN 1 m, HEADWAY x plan speed, kept plan
closest to clip(0.5 (v_des - v), -2, 2), v_des = the scene's max logged ego speed) controls its progress, replanning
every 0.5 s. Models: the A31 seed-0 checkpoints (trained on nuScenes train): chunk, step0, step1, marg; gc_chunk,
gc_step1, pm_chunk (src/gcomp.py); cv; oracle (logged future). Headway 1.0 s primary, 0.5 s secondary. Episode
bootstrap (2000) over scenes. A smoke run (3 scenes, 2 judges) checks only that the code runs (no metric is read).
- R1 (phantom braking on real traffic): hard brakes / episode chunk - step0 > 0 with lower bound > 0, and >= 50 % of
  chunk's hard brakes are phantom (the oracle on the same state keeps a plan with a >= -2).
- R2 (the common per-step lookahead design, which leaks on real data, A31 beta 0.74): hard brakes / episode
  step1 - step0 > 0 with lower bound > 0.
- R3 (causal conditioning is smooth AND safe): step0's TTC < 1.5 s steps per episode <= those of gc_chunk, pm_chunk
  and cv (point estimates).
- R4 (test-time g-computation of the one-step model removes its phantom braking): hard brakes gc_step1 - step1 < 0
  with upper bound < 0.
- Descriptive: front crashes, speed, phantom shares, lead-subset (vehicle in path within 30 m) metrics, the 0.5 s
  headway run, the logged driver.

### Vista scorer (committed 2026-10-19 ~00:50 local time, before any Vista beta was computed; Vista frames were viewed as media only)
src/vista_score.py implements the registered Vista readout (amendment 3 / A37g / A37h): beta at the last frame before
the collide plan's true contact (frame 24 if contact is after the 2 s clip), stepping back to the last frame tracked in
both clips, ds >= 1 m; readout validity on real keyframes. src/vista_leak.py read gains VISTA_DEV (runs off the pod).
Queue plan: A37h (g-logged, g-collide) is queued right after A37g so the GPU never waits; if A37g's precondition or P1
fails when scored, the A37h generation is stopped and not analysed (disclosed).
Note: the A37h code-check pilot window ad26ae1c is the first window of main_vista.txt, and gen is resumable, so its
pilot h- / g- clips would enter A37g / A37h; ad26ae1c is therefore EXCLUDED from the A37g and A37h tests (n <= 29).

## A40 (registered 2026-10-19 ~01:20 local time, before any output; after A39 R1 / R2 failed): the leak regime law, tested by dosing beta
Post-hoc account of A38 vs A39 to be tested prospectively here: by the leak law a judge with lead leak beta sees
(1 - beta) of the true change in the gap. beta < 1: bounded optimism (braking still helps, just less) -> a margin
planner is not phantom-braking; beta ~ 1: blindness (every plan keeps the same gap; OmniNWM 1.02, A37); beta > 1:
INVERSION (braking predicted to shrink the gap) -> phantom braking (A35 sim chunk beta 1.59, A38). A39's real-data models
(beta 0.68 / 0.74) are in the first regime. Test (A35 world, src/a38_closed.py + audit3/a38_diag.py, 4090 A35
checkpoints, headway 1.0, sim_test): judge mix<k> = step0 + k (chunk - step0), whose lead leak is k x 1.59 by linearity:
k = 0.25 / 0.5 / 0.63 / 0.75 / 1.0 (beta 0.40 / 0.80 / 1.00 / 1.19 / 1.59) plus step0.
- L1: hard brakes / episode of mix0.25 and mix0.5 <= 1.5 x step0's; of mix0.75 and mix1.0 >= 3 x step0's.
- L2: the phantom share of hard brakes is >= 0.5 for mix0.75 and mix1.0 and < 0.5 for mix0.25 and mix0.5.
- L3 (inversion): the judged front-safe share of the a = -2 plan is below that of the a = 0 plan for mix0.75 / mix1.0
  and above it for mix0.25 / mix0.5.
- Descriptive: mix0.63 (beta ~ 1), TTC violations and mean speed along k, headway 0.5 for the same judges.

### FINAL RUN PLAN (2026-10-19 00:55 local time, user: RunPod only until 12:00 local time Oct 19; before any A37g / dose output)
Pod queue after Vista main: A37g (30 windows; stopped if Vista P4 / the A37g precondition fails when scored) -> A37b
OmniNWM BRAKING arm only (d-2 on the 9 moving-ego windows; the in-distribution direction, the OOD test) -> A37h (only if
A37g P1 holds). Dropped: OmniNWM d1 / d3, seed-1 replicates, stopped-ego d1 / d3 (A37c planner test). Consequence:
A37b P1 (slope over >= 3 plans) is uninformative for OmniNWM (2 non-logged plans) and is tested on the Vista arm only;
A37b P2 (braking pulls the generated lead towards the ego in >= 60 % of windows) is tested on both models. The pod is
stopped at 12:00 local time whatever is unfinished (truncation reported). The registered A38 pod-CPU run is reported from its
4090 replicate where it does not finish.

## A41 (registered 2026-10-19 ~01:15 local time, before any output): leak dose on REAL traffic + A40 across model seeds
(a) A39 world (real nuScenes val traffic, log replay, src/a39_real_closed.py unchanged except mixture judges), judge
mix<k> = step0 + k (chunk - step0) with the A31 seed-0 checkpoints: lead leak ~0.68 k (A31 +1 s), k = 0.5 / 1 / 1.5 /
2 / 3 (beta ~0.34 / 0.68 / 1.02 / 1.36 / 2.04; k > 1 amplifies the plan dependence beyond the trained model), plus
step0 (k = 0); headways 1.0 and 0.5.
- D1: hard brakes / episode rise with k: Spearman(k, mean hard brakes) >= 0.9 over the 6 doses, at both headways.
- D2: phantom share of hard brakes >= 0.5 at k = 3 and < 0.5 at k <= 1 (headway 1.0).
(b) A40 with the A35 seed-1 and seed-2 checkpoints (k = 0.25 / 0.5 / 0.75 / 1.0, headways 1.0 and 0.5): hard brakes
rise with k for each seed (Spearman over the 5 doses incl. step0 >= 0.9).

### Deviation (2026-10-19 ~02:30 local time, after the Vista P4 result and an unplanned interim A37g look, see notebook)
A37h is run UNCONDITIONALLY in the remaining pod time (the A37g gate was a budget gate; the budget now allows it):
queue = A37g (to completion) -> OmniNWM braking arm -> A37h; A37h is truncated in the fixed window order at the
12:00 local time pod stop. Its P1 / P2 are reported as registered, with this deviation. A37g / A37h exclude ad26ae1c.

## Vista controllability check (logged 2026-10-19 ~06:40 local time, BEFORE its output; after seeing a parked-car readout of beta_static ~1.05)
Concern: Vista's beta ~1 could mean the camera does not execute the counterfactual plan (then every object reads beta 1,
and the Vista leak, braking and history results are uninterpretable). The detector readout of side-parked cars is
ambiguous (beta_static > 1 in many windows: box truncation near the image edge). Independent test (audit3/vista_ctrl.py):
visual localization against the REAL 12 Hz CAM_FRONT sweeps. For each moving-ego window, the generated frame at +2.0 s
of the logged plan and of the collide plan are matched (ORB + RANSAC fundamental-matrix inliers, same resize as Vista)
to every real sweep from +1.0 to +4.0 s; the best-matching real time t* gives the camera's position on the logged path,
s_log(t*) (keyframe progress, constant-speed extrapolation past +3 s). Realized extra advance = s_log(t*_collide) -
s_log(t*_logged); ratio = realized / commanded extra advance at +2.0 s.
- K0 (localizer valid): logged-plan t* within 0.5 s of 2.0 s in >= 70 % of windows.
- K1 (Vista executes the plan): median ratio >= 0.5 -> Vista's beta is reported as a leak (as for OmniNWM, 0.86);
  median ratio < 0.5 -> Vista's beta, braking and history results are reported as UNINFORMATIVE (the camera does not
  follow the counterfactual plan), and the paper's Vista claims are withdrawn.
Fallback (logged ~06:50 local time, after the first windows of audit3/vista_ctrl.py showed the logged-plan frame localizing
at the earliest real frame, i.e. K0 likely failing because of the generated-vs-real domain gap): audit3/vista_ctrl2.py
localizes generated against generated (same seed, same renderer). For each moving-ego window (v0 >= 2 m/s) and each
collide frame k with t_k in [0.5, 1.5] s, the best-matching logged-plan frame k'' (ORB + RANSAC inliers) gives the
realized position s_logged(t_k''); ratio = (s_logged(t_k'') - s_logged(t_k)) / (s_collide(t_k) - s_logged(t_k)) over
frames with commanded difference >= 0.5 m, median per window. Plan ignored -> ratio ~0; executed -> ~1. Same K1 rule
(median over windows >= 0.5); validity K0': d1 / d-2 give ratios of the same sign as their commanded difference.

### Deviation (2026-10-19 ~07:50 local time, after the control audits): A37h stopped, OmniNWM d1 arm added
Vista only partly executes counterfactual plans (exploratory 0.44; registered checks inconclusive), so A37h (video
g-computation in Vista) cannot be interpreted; its generation is stopped (partial clips not analysed). The remaining pod
time runs A37b's d1 plan on the 9 moving-ego OmniNWM windows (fixed order, truncated at the 12:00 stop), restoring the
registered A37b P1 (slope over >= 3 plans: d-2, d1, collide) on the model whose control is validated (PnP).

## Vista guidance test (registered 2026-10-19 ~10:00 local time, before any output; replaces the seed-1 OmniNWM run)
Vista was sampled with classifier-free guidance 2.5 on its conditions (trajectory included in the unconditional drop);
it executed only 0.44 of the counterfactual (collide) extra advance (exploratory audit). Test: the same 11 moving-ego
windows, plans logged + collide, seed 0, cfg_scale 5.0 (src/vista_leak.py VISTA_CFG), output /workspace/out_vista_cfg5.
Measures: execution ratio (audit3/vista_ctrl3.py, same rule) and lead beta (src/vista_leak.py read + src/vista_score.py).
- G1: median execution ratio (collide) at cfg 5 >= 0.7 -> Vista's leak at cfg 5 is reported as controllability-validated
  (beta at cfg 5 is the Vista number in the paper); otherwise Vista stays demoted (partial control).
- Descriptive: beta at cfg 5 vs 2.5 on the same windows; readout validity.
Extension (logged 10:10 local time, before any cfg-5 output): after the 11 moving windows, the remaining 19 main_vista windows
are generated at cfg 5 (logged, collide) in fixed order until 11:20 local time (truncated, reported); beta at cfg 5 is
reported on all generated windows; G1 (execution) is measured on the moving windows as registered.

## A42 (logged 2026-10-19 ~11:00 local time, before any output): real-traffic closed loop across object-model seeds (robustness)
A39 / A41 world, headways 0.5 and 1.0, A31 seed-1 and seed-2 checkpoints: step0, step1, chunk, and mix<k> = step0 +
k (chunk - step0) for k = 2, 3. Expectation: at headway 0.5, chunk - step0 near-miss steps > 0 for both seeds (point),
and near-misses rise with k (k = 0 < 1 < 3) for both seeds.
