"""Measuring and reducing the tracking error of the planned paths (shapes and the path to the centre).

MEASURING (metrics.tracking_errors): two RMS errors between the planned and the actual 2-D path
    pointwise     distance between the ball and the planned position at the same moment (timing counts)
    cross-track   shortest distance between the ball and the planned path (timing does not count)

REDUCING: an iterative loop, as in the assignment. Start from the controller's default settings, then
repeat: change one setting (Kp, Ki, Kd, acceleration feedforward) a little, simulate, keep the change if
the error went down (coordinate search with shrinking steps). The path settings (lap time, leg time,
cruise speed) are then swept separately: slowing the path always lowers the error, so it is reported as
a trade-off, not folded into the search. The tuned settings are checked on a harder plant and other
noise seeds they were not tuned on, so the improvement is not just fitting this simulation.

    python ball_tracking/error_reduction.py            run everything (a few minutes)
    python ball_tracking/error_reduction.py --quick    fewer paths and rounds

Outputs go to ball_tracking/error_reduction/: ERROR_REDUCTION.md (tables), error_reduction.csv, figures.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trajectory as tj
import trajectory_study as ts
from metrics import tracking_errors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "error_reduction")

DEFAULT_GAINS = {"kp": 0.3, "ki": 0.05, "kd": 0.35}      # the controller's starting gains
DEFAULT_FF = 0.0                                          # acceleration feedforward off, as in the notes

TRAIN_PLANT = ts.Plant()
TRAIN_SEEDS = (0, 1, 2)
# A harder plant and different noise, never seen while tuning: slower camera, more friction, a tilted plate
HARD_PLANT = ts.Plant(friction=0.5, camera_delay_frames=4, noise_cm=0.08, tilt_bias=(0.15, -0.15))
HARD_SEEDS = (10, 11, 12)

PENALTY = 10.0    # cm added to the score when the ball leaves the plate


def path_cases(quick=False):
    """The paths to study: name -> dict(method settings, start, target, seconds, skip, path polyline)."""
    cases = {}
    names = ["triangle", "circle"] if quick else ["line", "triangle", "square", "circle", "infinity"]
    for shape in names:
        settings = {"shape": shape, "size": 4.0, "lap_time": 10.0, "duration": 3.0}
        if shape in tj.CONTINUOUS:
            path = tj.shape_outline(shape, 4.0, points=400)
        else:
            path = tj.shape_outline(shape, 4.0)
        cases[shape] = dict(settings=settings, start=(0.0, 0.0), velocity=(0.0, 0.0), target=(0.0, 0.0),
                            seconds=30.0, skip=8.0, path=path)
    # Going to the centre from 10 cm away, from rest: the planned path is the straight line
    cases["to centre"] = dict(settings={"cruise_speed": 8.0, "slow_distance": 4.0, "duration": 3.0},
                              start=(10.0, 0.0), velocity=(0.0, 0.0), target=(0.0, 0.0), seconds=8.0, skip=0.0,
                              path=np.array([[10.0, 0.0], [0.0, 0.0]]))
    return cases


def evaluate(case, method, gains, ff, plant, seeds, planner_overrides=None, keep_run=False):
    """Mean tracking errors of one configuration over several noise seeds. Returns a dict (and the first run)."""
    settings = {**case["settings"], **(planner_overrides or {})}
    errors, runs, fell = [], [], 0
    for seed in seeds:
        run = ts.simulate(method, case["start"], case["velocity"], case["target"], gains=gains,
                          planner_settings=settings, plant=plant, seconds=case["seconds"], seed=seed,
                          accel_feedforward=ff)
        fell += bool(run["fell"])
        if len(run["t"]) < 5:
            errors.append({"pointwise_rms_cm": 99.0, "pointwise_max_cm": 99.0,
                           "cross_track_rms_cm": 99.0, "cross_track_max_cm": 99.0})
        else:
            errors.append(tracking_errors(run["t"], run["x"], run["y"], run["ref_x"], run["ref_y"], case["path"],
                                          case["skip"]))
        runs.append(run)
    result = {key: float(np.mean([e[key] for e in errors])) for key in errors[0]}
    result["fell"] = fell
    result["score"] = result["pointwise_rms_cm"] + result["cross_track_rms_cm"] + PENALTY * fell / len(seeds)
    return (result, runs[0]) if keep_run else result


def reduce_error(case, method, rounds=10):
    """Coordinate search from the default settings. Returns (gains, ff, history of (score, gains, ff))."""
    gains, ff = dict(DEFAULT_GAINS), DEFAULT_FF
    best = evaluate(case, method, gains, ff, TRAIN_PLANT, TRAIN_SEEDS)["score"]
    history = [(best, dict(gains), ff)]
    shrink = 1.0
    for _ in range(rounds):
        candidates = []
        for name in ("kp", "kd"):
            for factor in (1 - 0.35 * shrink, 1 + 0.5 * shrink):
                candidates.append(({**gains, name: round(gains[name] * factor, 4)}, ff))
        candidates.append(({**gains, "ki": round(gains["ki"] + 0.05 * shrink, 4)}, ff))
        candidates.append(({**gains, "ki": round(max(0.0, gains["ki"] - 0.05 * shrink), 4)}, ff))
        if method != "step":                      # the feedforward comes from the plan: none in step mode
            for delta in (-0.25 * shrink, 0.25 * shrink):
                candidates.append((dict(gains), round(float(np.clip(ff + delta, 0.0, 1.0)), 3)))
        scored = [(evaluate(case, method, g, f, TRAIN_PLANT, TRAIN_SEEDS)["score"], g, f) for g, f in candidates]
        score, g, f = min(scored, key=lambda item: item[0])
        if score < best - 1e-4:
            best, gains, ff = score, g, f
            history.append((best, dict(gains), ff))
        else:
            shrink *= 0.5
            if shrink < 0.1:
                break
    return gains, ff, history


def fmt(value, digits=2):
    return "nan" if value != value else f"{value:.{digits}f}"


LABEL = {"step": "Step", "trajectory": "Trajectory", "update": "Trajectory + replanning",
         "center": "Trajectory to centre"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="two shapes and fewer rounds")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cases = path_cases(args.quick)
    rounds = 5 if args.quick else 10
    rows, tuned = [], {}

    for name, case in cases.items():
        methods = ["step", "center"] if name == "to centre" else ["step", "trajectory", "update"]
        for method in methods:
            before_train = evaluate(case, method, DEFAULT_GAINS, DEFAULT_FF, TRAIN_PLANT, TRAIN_SEEDS)
            before_hard = evaluate(case, method, DEFAULT_GAINS, DEFAULT_FF, HARD_PLANT, HARD_SEEDS)
            gains, ff, history = reduce_error(case, method, rounds)
            after_train = evaluate(case, method, gains, ff, TRAIN_PLANT, TRAIN_SEEDS)
            after_hard = evaluate(case, method, gains, ff, HARD_PLANT, HARD_SEEDS)
            tuned[(name, method)] = (gains, ff)
            print(f"{name:10s} {LABEL[method]:24s} pointwise {before_train['pointwise_rms_cm']:.2f} -> "
                  f"{after_train['pointwise_rms_cm']:.2f}, cross-track {before_train['cross_track_rms_cm']:.2f} -> "
                  f"{after_train['cross_track_rms_cm']:.2f}  (hard plant {before_hard['score']:.2f} -> "
                  f"{after_hard['score']:.2f}) gains {gains} ff {ff}", flush=True)
            rows.append(dict(path=name, method=method, gains=gains, ff=ff, steps=len(history) - 1,
                             before_train=before_train, after_train=after_train,
                             before_hard=before_hard, after_hard=after_hard))

    write_outputs(cases, rows, tuned)


def write_outputs(cases, rows, tuned):
    with open(os.path.join(OUT, "error_reduction.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "method", "kp", "ki", "kd", "feedforward", "setting", "plant",
                         "pointwise_rms_cm", "cross_track_rms_cm", "pointwise_max_cm", "cross_track_max_cm", "fell_runs"])
        for r in rows:
            for setting, plant, key in (("default", "training", "before_train"), ("tuned", "training", "after_train"),
                                        ("default", "harder", "before_hard"), ("tuned", "harder", "after_hard")):
                m = r[key]
                g = DEFAULT_GAINS if setting == "default" else r["gains"]
                ff = DEFAULT_FF if setting == "default" else r["ff"]
                writer.writerow([r["path"], r["method"], g["kp"], g["ki"], g["kd"], ff, setting, plant,
                                 f"{m['pointwise_rms_cm']:.3f}", f"{m['cross_track_rms_cm']:.3f}",
                                 f"{m['pointwise_max_cm']:.3f}", f"{m['cross_track_max_cm']:.3f}", m["fell"]])

    lines = ["# Reducing the tracking error of the planned paths", "",
             "Errors are RMS over the run (after the ball has reached the path), in cm. "
             "Pointwise = distance to the planned position at the same time; cross-track = distance to the planned path. "
             "For Step the planned position jumps from corner to corner, so its pointwise error is not meaningful; "
             "compare Step by cross-track.", "",
             "Default = the controller's starting gains (Kp 0.3, Ki 0.05, Kd 0.35, feedforward 0). "
             "Tuned = the result of the iterative search (Kp, Ki, Kd, acceleration feedforward).", "",
             "Training plant = the simulated platform the search ran on; harder plant = slower camera, more friction, "
             "a tilted plate and different noise, never used while tuning (its column is pointwise + cross-track "
             "RMS, plus 10 cm per run in which the ball fell).", ""]
    lines += ["## Before and after tuning", "",
              "| Path | Method | Pointwise default | Pointwise tuned | Cross-track default | Cross-track tuned "
              "| Harder plant default | Harder plant tuned | Tuned Kp / Ki / Kd / FF |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["gains"]
        lines.append(f"| {r['path']} | {LABEL[r['method']]} | {fmt(r['before_train']['pointwise_rms_cm'])} | "
                     f"{fmt(r['after_train']['pointwise_rms_cm'])} | {fmt(r['before_train']['cross_track_rms_cm'])} | "
                     f"{fmt(r['after_train']['cross_track_rms_cm'])} | {fmt(r['before_hard']['score'])} | "
                     f"{fmt(r['after_hard']['score'])} | {g['kp']:.2f} / {g['ki']:.2f} / {g['kd']:.2f} / {r['ff']:.2f} |")

    lines += ["", "## The other lever: how fast the path asks the ball to go", "",
              "Tuned settings, training plant. Slower paths always track better; this is paid for in time, "
              "so it is a trade-off and not a free improvement.", "",
              "| Path | Setting | Value | Pointwise RMS (cm) | Cross-track RMS (cm) |", "|---|---|---|---|---|"]
    for name in cases:
        if name == "to centre":
            method, setting, values = "center", "cruise speed (cm/s)", (4.0, 6.0, 8.0, 12.0)
        elif name in tj.CONTINUOUS:
            method, setting, values = "trajectory", "lap time (s)", (6.0, 8.0, 10.0, 14.0)
        else:
            method, setting, values = "trajectory", "time per side (s)", (1.5, 2.0, 3.0, 4.0)
        gains, ff = tuned[(name, method)]
        for value in values:
            if method == "center":
                overrides = {"cruise_speed": value, "slow_distance": value / 2}
            elif name in tj.CONTINUOUS:
                overrides = {"lap_time": value}
            else:
                overrides = {"duration": value}
            m = evaluate(cases[name], method, gains, ff, TRAIN_PLANT, TRAIN_SEEDS, overrides)
            lines.append(f"| {name} | {setting} | {value:g} | {fmt(m['pointwise_rms_cm'])} | "
                         f"{fmt(m['cross_track_rms_cm'])} |")

    with open(os.path.join(OUT, "ERROR_REDUCTION.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    make_figures(cases, rows)


def make_figures(cases, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(cases)
    columns = math.ceil(len(names) / 2)
    fig, axes = plt.subplots(2, columns, figsize=(5 * columns, 9))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        method = "center" if name == "to centre" else "trajectory"
        row = next(r for r in rows if r["path"] == name and r["method"] == method)
        case = cases[name]
        ax.plot(case["path"][:, 0], case["path"][:, 1], "k--", linewidth=2, label="planned path")
        for gains, ff, color, text in ((DEFAULT_GAINS, DEFAULT_FF, "tab:red", "default"),
                                       (row["gains"], row["ff"], "tab:green", "tuned")):
            m, run = evaluate(case, method, gains, ff, TRAIN_PLANT, TRAIN_SEEDS[:1], keep_run=True)
            keep = run["t"] >= case["skip"]
            ax.plot(run["x"][keep], run["y"][keep], color=color, linewidth=1.5,
                    label=f"{text}: cross-track {m['cross_track_rms_cm']:.2f}, pointwise {m['pointwise_rms_cm']:.2f} cm")
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Ball path before and after reducing the error (Trajectory; to centre = Trajectory to centre)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_paths_before_after.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
