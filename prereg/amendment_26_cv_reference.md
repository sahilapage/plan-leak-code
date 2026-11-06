# Amendment 26 - past-only (constant-velocity) reference judge for SWTC (registered before the fold-B run)

Date: 2026-10-15, ~20:10 local time. Fold-B CV forecasts do not exist yet.

## Question
- Theory (docs/PAPER_NOTES.md, "Your plan is not evidence"): a reference plan used to judge candidates must carry no
  information about the future.
  - The logged plan encodes the real future (oracle).
  - Brake / still plans are informative in car-following logs ("ego stopped -> lead stopped"), hence the blind judge's
    failure on SWTC.
  - The constant-velocity plan from the past is uninformative.
- Dev (fold A, 140 windows / 10 scenes, src/sw_cvjudge.py, seen before this registration):

  | Judge | Recall | False alarm |
  |---|---|---|
  | self | 0.071 | 0.071 |
  | still | 0.943 | 0.964 |
  | logged (oracle) | 0.814 | 0.071 |
  | **CV** | **0.721** | **0.129** |

## Method (frozen; src/sw_cvjudge.py)
- **Reference plan:** straight, constant velocity, v0 = ego speed over the last keyframe interval.
- **Judging:** SWTC original weights; each candidate's ego box (logged, collide) placed in the CV forecast's frame;
  veto = obstacle p >= 0.2 in the box at any of +1 / +2 / +3 s.
- **Windows:** <work>/swrh_f1 (fold B, 115 windows / 11 scenes).

## Endpoints (scene-cluster bootstrap B = 10,000, seed 0; computed from src/sw_cvjudge.py score output)
- **P1:** recall of the collide candidate, CV judge - self judge: lower 95 % bound > 0.
- **P2:** CV judge false alarm on the logged path, upper 95 % bound <= 0.25.
- **Descriptive:** recall / false alarm of self, still, logged (oracle); real lead kept +2 s in the CV forecast.
