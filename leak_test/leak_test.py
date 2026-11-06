"""Model-agnostic plan-leak test (App. E of the paper).

Give a world model the logged plan and a counterfactual plan that differs only in the ego's motion, read where each
vehicle is in the two generated futures at the moment the counterfactual ego truly reaches the real car, and report

    beta = (shift of the vehicle between the two plans) / (shift of the ego between the two plans)

per relation to the ego (car ahead, parked, ...), as the median over scenes with a scene-bootstrap interval.
Truth for the car ahead and for parked cars within a few seconds is 0; at beta = 1 the gap to the car ahead never
closes, so every plan looks safe to a checker built on the model.

The test needs only black-box access. Two things are model-specific and left to the user:
  * a generator: (history, plan) -> generated future, and
  * a readout: generated future -> along-lane position (m) of each tracked vehicle and of the ego at time t.

Usage
  python leak_test.py demo                      # self-check on two synthetic models (leaky and causal)
  python leak_test.py score readouts.csv        # score your own readouts

readouts.csv columns: scene, vehicle, relation, plan (logged | counterfactual), ego_s, vehicle_s
  (positions along the lane, metres, at the real contact time of the counterfactual plan; one row per vehicle,
  scene and plan; optional column ego_s_commanded for the execution check)
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def counterfactual_plan(t, logged_s, lead_rear_s, ego_length_front=0.0, penetration=1.0,
                        grid=np.arange(0.25, 8.01, 0.25)):
    """Smallest constant extra acceleration a (m/s^2) such that logged_s(t) + a t^2 / 2 drives the ego front at least
    `penetration` metres past the real car's rear within the horizon. Returns (a, plan_s, contact index) or None."""
    t, logged_s, lead_rear_s = map(np.asarray, (t, logged_s, lead_rear_s))
    for a in grid:
        plan = logged_s + 0.5 * a * t ** 2
        gap = lead_rear_s - (plan + ego_length_front)
        if (gap <= -penetration).any():
            k = int(np.nonzero(gap <= 0)[0][0])            # real contact: the leak is read here
            return float(a), plan, k
    return None


def scene_betas(rows):
    """rows: dicts with scene, vehicle, relation, plan, ego_s, vehicle_s. Returns {relation: {scene: median beta}}."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["scene"], r["vehicle"], r["relation"])][r["plan"]] = (float(r["ego_s"]), float(r["vehicle_s"]))
    out = defaultdict(lambda: defaultdict(list))
    for (scene, _, rel), p in by.items():
        if "logged" not in p or "counterfactual" not in p:
            continue
        de = p["counterfactual"][0] - p["logged"][0]
        if abs(de) < 0.5:                                   # the plans must move the ego apart to read a slope
            continue
        out[rel][scene].append((p["counterfactual"][1] - p["logged"][1]) / de)
    return {rel: {s: float(np.median(v)) for s, v in d.items()} for rel, d in out.items()}


def bootstrap_median(values, n_boot=2000, seed=0):
    """Median over scenes with a 95% scene-bootstrap interval."""
    v = np.asarray(list(values), float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def execution(rows):
    """Realized / commanded extra ego advance (needs ego_s_commanded); a model that does not execute the plan cannot
    show a leak, so check this before reading beta."""
    by = defaultdict(dict)
    for r in rows:
        if r.get("ego_s_commanded") not in (None, ""):
            by[r["scene"]][r["plan"]] = (float(r["ego_s"]), float(r["ego_s_commanded"]))
    ratios = [(p["counterfactual"][0] - p["logged"][0]) / (p["counterfactual"][1] - p["logged"][1])
              for p in by.values() if "logged" in p and "counterfactual" in p
              and abs(p["counterfactual"][1] - p["logged"][1]) > 0.5]
    return float(np.median(ratios)) if ratios else None


def report(rows):
    betas = scene_betas(rows)
    print(f"{'relation':<12}{'scenes':>7}  beta (95% scene-bootstrap CI)")
    for rel in sorted(betas, key=lambda r: (r != "lead", r)):
        m, lo, hi = bootstrap_median(betas[rel].values())
        print(f"{rel:<12}{len(betas[rel]):>7}  {m:+.2f} ({lo:+.2f} to {hi:+.2f})")
    ex = execution(rows)
    if ex is not None:
        print(f"execution: the generated ego realizes {ex:.2f} of the commanded extra advance")
    return betas


def demo():
    """Two synthetic 'world models' on the same scenes: a leaky one that moves the car ahead with the ego (beta 1, the
    gap never closes) and a causal one that leaves it in place. Parked cars stay put in both."""
    rng = np.random.default_rng(1)
    t = np.arange(0, 3.01, 1 / 12)
    for name, leak in (("leaky (reads the world off the plan)", 1.0), ("causal", 0.0)):
        rows = []
        for i in range(18):
            v0 = rng.uniform(0, 8)                            # ego speed; the car ahead is stopped
            logged = v0 * t - 0.5 * min(v0 / 3, 2.0) * np.minimum(t, 3) ** 2 * (v0 > 0)
            rear = np.full_like(t, logged[-1] + rng.uniform(4, 10))
            cf = counterfactual_plan(t, logged, rear)
            if cf is None:
                continue
            _, plan, k = cf
            de = plan[k] - logged[k]
            for plan_name, ego in (("logged", logged[k]), ("counterfactual", plan[k])):
                shift = leak * de if plan_name == "counterfactual" else 0.0
                rows.append(dict(scene=f"s{i}", vehicle="lead", relation="lead", plan=plan_name, ego_s=ego,
                                 vehicle_s=rear[k] + shift + rng.normal(0, 0.3)))
                rows.append(dict(scene=f"s{i}", vehicle="p1", relation="parked", plan=plan_name, ego_s=ego,
                                 vehicle_s=rear[k] + 12 + rng.normal(0, 0.3)))
        print(f"\n== {name}")
        report(rows)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "score":
        with open(sys.argv[2]) as f:
            report(list(csv.DictReader(f)))
    else:
        demo()
