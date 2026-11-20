"""Simulation study: three ways to move the ball to the target, and whether they need feedforward.

    step        the reference jumps to the target; one PID loop
    trajectory  a Hermite path is planned at the start and the PID follows it
    update      the Hermite path is replanned from the measured state every so often

Each trajectory method is also run with acceleration feedforward (the tilt that produces the planned
acceleration is added to the PID output). The PID-only versions are the method as described in the notes.

    python trajectory_study.py            run all experiments (several minutes)
    python trajectory_study.py --quick    smaller tuning grid, much faster
    python trajectory_study.py --csv tuning_runs/csv/run_xxx.csv   score a run recorded on the real platform

Plots, tables and results.csv are written to trajectory_study/ next to this script.

The simulation uses the same planner (trajectory.py), PID (pid.py), alpha-beta filter and metrics
(metrics.py) as the real controller. The plant is a model, not the hardware: a solid ball rolling on the
tilted plate (a = 5/7 g sin(tilt) minus friction), a servo that follows the tilt command with a lag, and a
camera that sees the ball a few frames late with some noise. Its numbers are assumptions (see Plant below),
so the real platform runs are what confirm or correct these results.
"""
import argparse
import csv
import itertools
import math
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import trajectory as tj
from metrics import PLATE_RADIUS_CM, compute_metrics
from simulation import DT, DEFAULT_GAINS, DEFAULT_PLANNER, Plant, simulate, metrics_of

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "trajectory_study")

# How much of the planned acceleration is fed forward in the "feedforward" configurations. Experiment 8
# sweeps this. 0.75 has the best score because it keeps the ball on the plate in every starting condition;
# 0.5 is gentler in routine moves (less overshoot) but loses the ball in the hardest start, and at 1.0
# steady tracking gets worse because the feedforward adds to what the PID already does about the lag.
FEEDFORWARD = 0.75

# The configurations compared: (planner method, acceleration feedforward gain)
CONFIGS = {
    "step": ("step", 0.0),
    "trajectory": ("trajectory", 0.0),
    "update": ("update", 0.0),
    "trajectory_ff": ("trajectory", FEEDFORWARD),
    "update_ff": ("update", FEEDFORWARD),
}
KEYS = list(CONFIGS)
NAMES = {"step": "Step", "trajectory": "Trajectory", "update": "Trajectory + replanning",
         "trajectory_ff": "Trajectory + feedforward", "update_ff": "Trajectory + replanning + feedforward"}
COLORS = {"step": "tab:red", "trajectory": "tab:blue", "update": "tab:green",
          "trajectory_ff": "tab:cyan", "update_ff": "tab:olive"}

# The conditions of the "initial position and velocity" experiment: (start x in cm, start velocity in cm/s).
# A negative velocity is towards the target, a positive one is away from it.
CONDITIONS = [(x0, v0) for x0 in (5, 10, 12) for v0 in (-15, 0, 15)]


def run_config(key, start, velocity=(0.0, 0.0), **kwargs):
    """simulate() for one of the configurations in CONFIGS."""
    method, feedforward = CONFIGS[key]
    return simulate(method, start, velocity, accel_feedforward=feedforward, **kwargs)


def circle_reference(radius, period):
    """Reference function for a circle: (position, velocity, acceleration) at time t."""
    omega = 2 * math.pi / period

    def reference(t):
        position, velocity = tj.circular_reference(t, radius, period)
        return position, velocity, -(omega ** 2) * position    # the acceleration is centripetal

    return reference


# --------------------------------------------------------------------------- helpers for reporting
ALL_RUNS = []   # every run's metrics, written to results.csv at the end
TABLES = []     # markdown tables, written to summary.md at the end


def log_run(experiment, scenario, key, metrics, gains=None, planner_settings=None):
    gains = {**DEFAULT_GAINS, **(gains or {})}
    settings = {**DEFAULT_PLANNER, **(planner_settings or {})}
    ALL_RUNS.append({"experiment": experiment, "scenario": scenario, "config": key, **gains,
                     "duration": settings["duration"], "replan_interval": settings["replan_interval"], **metrics})


def fmt(value, digits=2):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "never"
    return f"{value:.{digits}f}"


def fmt_move(r):
    """Settling time / overshoot of one run, as text."""
    settle = "never" if math.isnan(r["settling_time_s"]) else f"{r['settling_time_s']:.1f} s"
    return f"{settle} / {r['overshoot_cm']:.1f} cm" + (" LOST" if r["fell"] else "")


def table(title, header, rows):
    """Print a table and keep it for summary.md."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    print(f"\n{title}\n" + "\n".join(lines))
    TABLES.append(f"### {title}\n\n" + "\n".join(lines) + "\n")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=120)
    plt.close(fig)
    print(f"saved {name}")


def condition_label(x0, v0):
    return f"x0={x0}, v0={v0:+d}"


def grouped_bars(ax, values_by_key, width=None):
    """One group of bars per condition, one bar per configuration. values_by_key[key] is a list."""
    count = len(values_by_key)
    width = width or 0.8 / count
    for j, (key, values) in enumerate(values_by_key.items()):
        ax.bar(np.arange(len(values)) + (j - (count - 1) / 2) * width, values, width, color=COLORS[key],
               label=NAMES[key])


# --------------------------------------------------------------------------- experiment 1: the 10 cm move
def experiment_headline(seeds=10):
    """Move from 10 cm to 0, from rest, with each configuration and the default settings."""
    runs = {key: run_config(key, (10, 0)) for key in KEYS}

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for key, run in runs.items():
        c = COLORS[key]
        axes[0].plot(run["t"], run["x"], color=c, linewidth=2, label=NAMES[key])
        axes[0].plot(run["t"], run["ref_x"], color=c, linestyle=":", linewidth=1, alpha=0.8)
        axes[1].plot(run["t"], run["vx"], color=c, linewidth=2, label=NAMES[key])
        axes[2].plot(run["t"], run["tilt_x"], color=c, linewidth=2, label=NAMES[key])
        axes[3].plot(run["t"], run["y"], color=c, linewidth=2, label=NAMES[key])
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].axhspan(-0.5, 0.5, color="gray", alpha=0.15, label="settling band (0.5 cm)")
    axes[0].set_ylabel("x position (cm)")
    axes[0].set_title("Moving the ball from 10 cm to 0 (solid: ball, dotted: what each method asks for)")
    axes[1].set_ylabel("x velocity (cm/s)")
    axes[2].set_ylabel("x tilt command (deg)")
    axes[3].set_ylabel("y position (cm)")
    axes[3].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    save(fig, "fig1_headline_10cm_move.png")

    # Spread over several random noise seeds
    rows = []
    for key in KEYS:
        results = [metrics_of(run_config(key, (10, 0), seed=s)) for s in range(seeds)]
        for r in results:
            log_run("headline", "x0=10, v0=0", key, r)
        settle = np.array([r["settling_time_s"] for r in results])
        rows.append([NAMES[key],
                     f"{np.nanmean(settle):.2f} +- {np.nanstd(settle):.2f}" if (~np.isnan(settle)).any() else "never",
                     f"{np.mean([r['overshoot_cm'] for r in results]):.2f} +- {np.std([r['overshoot_cm'] for r in results]):.2f}",
                     f"{np.mean([r['rms_to_reference_cm'] for r in results]):.2f}",
                     f"{np.mean([r['final_error_cm'] for r in results]):.2f}",
                     f"{np.mean([r['peak_tilt_deg'] for r in results]):.1f}"])
    table(f"Experiment 1: 10 cm to 0 from rest, default settings, mean over {seeds} noise seeds",
          ["Method", "Settling time (s)", "Overshoot (cm)", "RMS error to its reference (cm)", "Final error (cm)",
           "Peak tilt (deg)"], rows)


# --------------------------------------------------------------------------- experiment 2: initial conditions
def experiment_initial_conditions():
    results = {key: [] for key in KEYS}
    for x0, v0 in CONDITIONS:
        for key in KEYS:
            r = metrics_of(run_config(key, (x0, 0), (v0, 0)))
            results[key].append(r)
            log_run("initial_conditions", condition_label(x0, v0), key, r)

    labels = [condition_label(*c) for c in CONDITIONS]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    grouped_bars(axes[0], {k: [10.0 if np.isnan(r["settling_time_s"]) else r["settling_time_s"] for r in results[k]] for k in KEYS})
    for i in range(len(CONDITIONS)):
        for j, key in enumerate(KEYS):
            if results[key][i]["fell"]:
                axes[0].text(i + (j - (len(KEYS) - 1) / 2) * 0.8 / len(KEYS), 10.2, "x", ha="center",
                             color=COLORS[key], fontsize=10, fontweight="bold")
    grouped_bars(axes[1], {k: [r["overshoot_cm"] for r in results[k]] for k in KEYS})
    grouped_bars(axes[2], {k: [r["peak_tilt_deg"] for r in results[k]] for k in KEYS})
    axes[0].set_ylabel("Settling time (s)\n(10 = never)")
    axes[0].set_title("Different starting positions and velocities (x = the ball left the plate)")
    axes[1].set_ylabel("Overshoot (cm)")
    axes[2].set_ylabel("Peak tilt (deg)")
    axes[2].axhline(10, color="k", linewidth=0.8, linestyle=":")
    axes[2].set_xticks(np.arange(len(CONDITIONS)))
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    save(fig, "fig2_initial_conditions.png")

    rows = []
    for i, (x0, v0) in enumerate(CONDITIONS):
        rows.append([condition_label(x0, v0)] + [fmt_move(results[key][i]) for key in KEYS])
    table("Experiment 2: settling time / overshoot for each starting condition (default settings; LOST = left the plate)",
          ["Start"] + [NAMES[k] for k in KEYS], rows)

    summary = []
    for key in KEYS:
        rs = results[key]
        settle = np.array([r["settling_time_s"] for r in rs])
        summary.append([NAMES[key], f"{np.sum(~np.isnan(settle))} of {len(settle)}", fmt(np.nanmean(settle)),
                        fmt(np.mean([r["overshoot_cm"] for r in rs])), fmt(np.max([r["overshoot_cm"] for r in rs])),
                        fmt(np.mean([r["peak_tilt_deg"] for r in rs]), 1), sum(r["fell"] for r in rs)])
    table("Experiment 2 summary over the 9 starting conditions",
          ["Method", "Settled within 10 s", "Mean settling of those (s)", "Mean overshoot (cm)",
           "Worst overshoot (cm)", "Mean peak tilt (deg)", "Ball lost"], summary)


# --------------------------------------------------------------------------- experiment 3: disturbances
def disturbances():
    """The plants the controllers are tested on: the assumed one, and the same with one thing made worse."""
    return {
        "nominal": Plant(),
        "plate bias -0.9 deg": Plant(tilt_bias=(-0.9, 0.0)),
        "noisy camera (0.3 cm)": Plant(noise_cm=0.3),
        "late camera (167 ms)": Plant(camera_delay_frames=5),
        "slow servo (0.15 s)": Plant(servo_tau=0.15),
        "high friction (1.0)": Plant(friction=1.0),
        "bump at 1 s (+10 cm/s)": Plant(bump=(1.0, 10.0, 0.0)),
    }


STARTS = [("x0=10, v0=0", (10, 0), (0, 0)), ("x0=10, v0=+15", (10, 0), (15, 0))]


def experiment_robustness():
    scenarios = disturbances()
    starts = STARTS
    fig, axes = plt.subplots(len(starts), 2, figsize=(14, 8), sharex=True)
    rows = []
    for row, (label, p0, v0) in enumerate(starts):
        settle = {k: [] for k in KEYS}
        final = {k: [] for k in KEYS}
        for name, plant in scenarios.items():
            for key in KEYS:
                r = metrics_of(run_config(key, p0, v0, plant=plant))
                log_run("robustness", f"{label}; {name}", key, r)
                settle[key].append(10.0 if np.isnan(r["settling_time_s"]) else r["settling_time_s"])
                final[key].append(min(r["final_error_cm"], 12.0))
                rows.append([label, name, NAMES[key], fmt_move(r), fmt(r["final_error_cm"]),
                             "yes" if r["stable"] else "NO"])
        grouped_bars(axes[row, 0], settle)
        grouped_bars(axes[row, 1], final)
        axes[row, 0].set_ylabel(f"{label}\nSettling time (s, 10 = never)")
        axes[row, 1].set_ylabel("Final error (cm)")
        for ax in axes[row]:
            ax.grid(True, alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("Settling time with a disturbed plant")
    axes[0, 1].set_title("Final error with a disturbed plant")
    for col in range(2):
        axes[-1, col].set_xticks(np.arange(len(scenarios)))
        axes[-1, col].set_xticklabels(list(scenarios), rotation=30, ha="right")
    save(fig, "fig3_robustness.png")
    table("Experiment 3: disturbed plant (default gains, no integral term)",
          ["Start", "Disturbance", "Method", "Settling / overshoot", "Final error (cm)", "Stable"], rows)


# --------------------------------------------------------------------------- experiment 4: tuning vs method
def score_setting(key, gains, planner_settings):
    """One number for how good a configuration with these gains/settings is over the 9 starting conditions."""
    settle, over, lost = [], [], 0
    for x0, v0 in CONDITIONS:
        r = metrics_of(run_config(key, (x0, 0), (v0, 0), gains=gains, planner_settings=planner_settings))
        settle.append(10.0 if np.isnan(r["settling_time_s"]) else r["settling_time_s"])
        over.append(r["overshoot_cm"])
        lost += r["fell"]
    return float(np.mean(settle) + 2 * np.mean(over) + 20 * lost), float(np.mean(settle)), float(np.mean(over)), lost


def refine_setting(key, gains, planner, score, rounds=8):
    """Hill-climb from a good setting: nudge each parameter, keep the best change, repeat until nothing helps.

    The grid search only finds the best of the grid points; this lets the optimum land between them (or
    beyond the edge of the grid).
    """
    method = CONFIGS[key][0]
    gains = {**DEFAULT_GAINS, **gains}
    for _ in range(rounds):
        candidates = []
        for kp in (gains["kp"] * 0.8, gains["kp"] * 1.25):
            candidates.append(({**gains, "kp": round(kp, 3)}, planner))
        for kd in (gains["kd"] * 0.7, gains["kd"] * 1.4):
            candidates.append(({**gains, "kd": round(kd, 3)}, planner))
        for ki in (0.0, 0.05, 0.1, 0.2):
            if ki != gains["ki"]:
                candidates.append(({**gains, "ki": ki}, planner))
        if method != "step":
            for duration in (planner["duration"] - 0.25, planner["duration"] + 0.25):
                if 0.5 <= duration <= 6.0:
                    candidates.append((gains, {**planner, "duration": round(duration, 2)}))
        scored = [(score_setting(key, g, p), g, p) for g, p in candidates]
        best = min(scored, key=lambda item: item[0][0])
        if best[0][0] >= score[0] * 0.99:    # no real improvement
            break
        score, gains, planner = best
    return score, gains, planner


def experiment_tuning(quick=False):
    """Does changing the method help more than tuning one method? Search the settings of each."""
    if quick:
        kps, kds, kis, durations, rounds = (0.4, 0.75), (0.3, 0.6), (0.0,), (2.0, 3.0), 2
    else:
        kps, kds, kis, durations, rounds = (0.25, 0.4, 0.6, 0.9), (0.15, 0.3, 0.5, 0.8), (0.0, 0.1), (1.0, 2.0, 3.0), 8
    gain_grid = [{"kp": kp, "kd": kd, "ki": ki} for kp, kd, ki in itertools.product(kps, kds, kis)]

    best, landscape = {}, {}
    for key in KEYS:
        method = CONFIGS[key][0]
        # Methods with a planned path also get to choose how long the move takes
        settings = [{"duration": d} for d in durations] if method != "step" else [{}]
        default_planner = {} if method == "step" else {"duration": 3.0}
        default = score_setting(key, None, default_planner)
        tried = []
        for gains, planner in itertools.product(gain_grid, settings):
            tried.append((score_setting(key, gains, planner), gains, planner))
            if gains["ki"] == 0.0 and planner in ({}, {"duration": durations[-1]}):
                landscape[(key, gains["kp"], gains["kd"])] = tried[-1][0][0]
        tried.sort(key=lambda item: item[0][0])
        tuned = refine_setting(key, tried[0][1], tried[0][2], tried[0][0], rounds)
        best[key] = {"default": (default, {}, default_planner), "tuned": tuned}
        print(f"  tuned {NAMES[key]}: {len(tried)} grid points, then refined", flush=True)

    rows = []
    for key in KEYS:
        for kind in ("default", "tuned"):
            (score, settle, over, lost), gains, planner = best[key][kind]
            g = {**DEFAULT_GAINS, **gains}
            rows.append([NAMES[key], kind, f"Kp {g['kp']}, Ki {g['ki']}, Kd {g['kd']}"
                         + (f", T {planner['duration']} s" if planner else ""),
                         fmt(settle), fmt(over), lost, fmt(score, 1)])
            log_run("tuning", kind, key, {"mean_settling_s": settle, "mean_overshoot_cm": over, "ball_lost": lost,
                                          "score": score}, gains, planner)
    table("Experiment 4: default vs best-tuned settings, scored over the 9 starting conditions "
          "(score = mean settling + 2 x mean overshoot + 20 per lost ball; lower is better)",
          ["Method", "Settings", "Gains / planner", "Mean settling (s)", "Mean overshoot (cm)", "Ball lost", "Score"], rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    labels, settle, over, colors = [], [], [], []
    for key in KEYS:
        for kind in ("default", "tuned"):
            (score, s, o, lost), _, _ = best[key][kind]
            labels.append(f"{NAMES[key].replace(' + ', chr(10) + '+ ')}\n{kind}")
            settle.append(s)
            over.append(o)
            colors.append(COLORS[key])
    for ax, values, title in ((axes[0], settle, "Mean settling time (s)"), (axes[1], over, "Mean overshoot (cm)")):
        bars = ax.bar(range(len(values)), values, color=colors)
        for index, bar in enumerate(bars):
            bar.set_hatch("//" if index % 2 else "")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(title + "   (hatched = best-tuned)")
        ax.grid(True, alpha=0.3, axis="y")
    grid = np.array([[landscape[("step", kp, kd)] for kp in kps] for kd in kds])
    image = axes[2].imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
    axes[2].set_xticks(range(len(kps)))
    axes[2].set_xticklabels(kps)
    axes[2].set_yticks(range(len(kds)))
    axes[2].set_yticklabels(kds)
    axes[2].set_xlabel("Kp")
    axes[2].set_ylabel("Kd")
    axes[2].set_title("Step method: score over the gain grid (Ki = 0)")
    fig.colorbar(image, ax=axes[2])
    save(fig, "fig4_tuning_vs_method.png")
    return best


# --------------------------------------------------------------------------- experiment 9: tuned, then disturbed
def experiment_tuned_robustness(best):
    """Each method with its own best settings (tuned on the undisturbed plant), on the disturbed plants."""
    scenarios = disturbances()
    results = {key: [] for key in KEYS}
    rows = []
    fig, axes = plt.subplots(len(STARTS), 2, figsize=(14, 8), sharex=True)
    for row, (label, p0, v0) in enumerate(STARTS):
        settle = {k: [] for k in KEYS}
        final = {k: [] for k in KEYS}
        for name, plant in scenarios.items():
            for key in KEYS:
                _, gains, planner = best[key]["tuned"]
                r = metrics_of(run_config(key, p0, v0, plant=plant, gains=gains, planner_settings=planner))
                results[key].append(r)
                log_run("tuned_robustness", f"{label}; {name}", key, r, gains, planner)
                settle[key].append(10.0 if np.isnan(r["settling_time_s"]) else r["settling_time_s"])
                final[key].append(min(r["final_error_cm"], 12.0))
                rows.append([label, name, NAMES[key], fmt_move(r), fmt(r["final_error_cm"])])
        grouped_bars(axes[row, 0], settle)
        grouped_bars(axes[row, 1], final)
        axes[row, 0].set_ylabel(f"{label}\nSettling time (s, 10 = never)")
        axes[row, 1].set_ylabel("Final error (cm)")
        for ax in axes[row]:
            ax.grid(True, alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("Tuned settings, disturbed plant: settling time")
    axes[0, 1].set_title("Tuned settings, disturbed plant: final error")
    for col in range(2):
        axes[-1, col].set_xticks(np.arange(len(scenarios)))
        axes[-1, col].set_xticklabels(list(scenarios), rotation=30, ha="right")
    save(fig, "fig9_tuned_robustness.png")
    table("Experiment 9: each method with its own tuned settings, on the disturbed plants",
          ["Start", "Disturbance", "Method", "Settling / overshoot", "Final error (cm)"], rows)

    summary = []
    for key in KEYS:
        rs = results[key]
        settle = np.array([r["settling_time_s"] for r in rs])
        summary.append([NAMES[key], f"{int(np.sum(~np.isnan(settle)))} of {len(rs)}", sum(r["fell"] for r in rs),
                        fmt(float(np.nanmean(settle))), fmt(float(np.mean([r["overshoot_cm"] for r in rs]))),
                        fmt(float(np.mean([min(r["final_error_cm"], 12.0) for r in rs])))])
    table("Experiment 9 summary over the 14 disturbed runs (7 disturbances x 2 starts)",
          ["Method (tuned)", "Settled within 10 s", "Ball lost", "Mean settling of those (s)", "Mean overshoot (cm)",
           "Mean final error (cm)"], summary)


# --------------------------------------------------------------------------- experiment 5: planner settings
def experiment_planner_settings():
    durations = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    intervals = [0.1, 0.25, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    rows = []
    metrics_names = ("settling_time_s", "overshoot_cm", "peak_tilt_deg")
    labels = ("Settling time (s)", "Overshoot (cm)", "Peak tilt (deg)")
    for key in ("trajectory", "update", "trajectory_ff", "update_ff"):
        values = {m: [] for m in metrics_names}
        for d in durations:
            r = metrics_of(run_config(key, (10, 0), planner_settings={"duration": d}))
            log_run("planner_duration", f"T={d}", key, r, planner_settings={"duration": d})
            for m in metrics_names:
                values[m].append(r[m])
            rows.append([NAMES[key], f"duration {d} s", fmt(r["settling_time_s"]), fmt(r["overshoot_cm"]),
                         fmt(r["peak_tilt_deg"], 1), fmt(r["rms_to_reference_cm"])])
        for ax, m, label in zip(axes[0], metrics_names, labels):
            ax.plot(durations, values[m], "o-", color=COLORS[key], label=NAMES[key])
            ax.set_ylabel(label)
            ax.set_xlabel("Planned duration T (s)")
    step_r = metrics_of(run_config("step", (10, 0)))
    for ax, m in zip(axes[0], metrics_names):
        ax.axhline(step_r[m], color=COLORS["step"], linestyle="--", label="Step method")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("Planned duration (move from 10 cm to 0)")

    for key in ("update", "update_ff"):
        values = {m: [] for m in metrics_names}
        for interval in intervals:
            settings = {"duration": 3.0, "replan_interval": interval}
            r = metrics_of(run_config(key, (10, 0), planner_settings=settings))
            log_run("replan_interval", f"interval={interval}", key, r, planner_settings=settings)
            for m in metrics_names:
                values[m].append(r[m])
            rows.append([NAMES[key], f"replan every {interval} s (T=3)", fmt(r["settling_time_s"]),
                         fmt(r["overshoot_cm"]), fmt(r["peak_tilt_deg"], 1), fmt(r["rms_to_reference_cm"])])
        for ax, m, label in zip(axes[1], metrics_names, labels):
            ax.plot(intervals, values[m], "o-", color=COLORS[key], label=NAMES[key])
            ax.set_ylabel(label)
            ax.set_xlabel("Replan interval (s)")
            ax.grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("Replan interval (T = 3 s)")
    save(fig, "fig5_planner_settings.png")
    table("Experiment 5: planner settings, move from 10 cm to 0 from rest",
          ["Method", "Setting", "Settling (s)", "Overshoot (cm)", "Peak tilt (deg)", "RMS error to reference (cm)"], rows)


# --------------------------------------------------------------------------- experiment 6: the circle
def experiment_circle():
    """Follow a moving reference (the circle from the notes). Does the reference velocity / feedforward help?"""
    radius = 5.0
    periods = [12.0, 8.0, 6.0, 4.0]
    variants = [("D on measured velocity only", False, 0.0, "tab:orange"),
                ("D on velocity error (reference velocity)", True, 0.0, "tab:blue"),
                (f"+ acceleration feedforward {FEEDFORWARD}", True, FEEDFORWARD, "tab:green"),
                ("+ acceleration feedforward 1.0", True, 1.0, "tab:red")]

    def circle(period):
        return circle_reference(radius, period)

    rows, rms, paths = [], {v[0]: [] for v in variants}, {}
    for period in periods:
        omega = 2 * math.pi / period
        for label, use_v, ff, _ in variants:
            run = simulate("trajectory", (radius, 0.0), (0.0, radius * omega), seconds=24.0,
                           use_reference_velocity=use_v, accel_feedforward=ff, reference_fn=circle(period))
            after = run["t"] >= 4.0    # skip the first seconds, while the ball settles onto the circle
            error = np.hypot(run["x"] - run["ref_x"], run["y"] - run["ref_y"])[after]
            value = float(np.sqrt(np.mean(error ** 2)))
            rms[label].append(value)
            paths[(period, label)] = run
            ALL_RUNS.append({"experiment": "circle", "scenario": f"period {period} s; {label}",
                             "config": "circle", "rms_to_reference_cm": value})
            rows.append([f"{period:g} s", label, fmt(value), fmt(float(np.max(error)))])
    table("Experiment 6: following a circle of radius 5 cm (error after the first 4 s)",
          ["Period", "Controller", "RMS error (cm)", "Worst error (cm)"], rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    theta = np.linspace(0, 2 * np.pi, 200)
    for ax, period in zip(axes[:2], (12.0, 6.0)):
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k--", linewidth=1, label="Reference circle")
        for label, _, _, color in variants:
            run = paths[(period, label)]
            after = run["t"] >= 4.0
            ax.plot(run["x"][after], run["y"][after], color=color, linewidth=1.5, label=label)
        ax.set_aspect("equal")
        ax.set_title(f"Circle, period {period:g} s")
        ax.legend(fontsize=6, loc="lower center")
        ax.grid(True, alpha=0.3)
    for label, _, _, color in variants:
        axes[2].plot(periods, rms[label], "o-", color=color, label=label)
    axes[2].set_xlabel("Period (s)  (shorter = faster)")
    axes[2].set_ylabel("RMS tracking error (cm)")
    axes[2].set_title("Tracking error vs speed")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)
    save(fig, "fig6_circle_tracking.png")


# --------------------------------------------------------------------------- experiment 7: replanner variants
def experiment_replanner_variants():
    """Why the replanner has its rules: compare it with a version that replans forever, and others."""
    variants = [
        ("No replanning (method b)", "trajectory", {}, 0.0),
        ("Replan forever (literal)", "update", {"final_approach": -1e9}, 0.0),
        ("Replan + final-approach rule (used)", "update", {}, 0.0),
        ("Replan only when 1 cm off the plan", "update", {"trigger": 1.0}, 0.0),
        ("No replanning + feedforward", "trajectory", {}, FEEDFORWARD),
        ("Replan (used) + feedforward", "update", {}, FEEDFORWARD),
        ("Replan only when 1 cm off + feedforward", "update", {"trigger": 1.0}, FEEDFORWARD),
    ]
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:cyan", "tab:olive", "tab:brown")
    scenarios = [
        ("nominal", (10, 0), (0, 0), Plant()),
        ("bump at 1 s (+10 cm/s)", (10, 0), (0, 0), Plant(bump=(1.0, 10.0, 0.0))),
        ("plate bias -0.9 deg", (10, 0), (0, 0), Plant(tilt_bias=(-0.9, 0.0))),
        ("outward start (+10 cm/s)", (10, 0), (10, 0), Plant()),
    ]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (scenario, p0, v0, plant) in zip(axes.flat, scenarios):
        for (label, method, settings, ff), color in zip(variants, colors):
            run = simulate(method, p0, v0, plant=plant, planner_settings=settings, accel_feedforward=ff)
            r = metrics_of(run)
            log_run("replanner_variants", f"{scenario}; {label}", method + ("_ff" if ff else ""), r,
                    planner_settings=settings)
            ax.plot(run["t"], run["x"], color=color, linewidth=1.5, label=label)
            rows.append([scenario, label, fmt_move(r), fmt(r["final_error_cm"]), fmt(r["final_jitter_cm"]),
                         "yes" if r["stable"] else "NO", getattr(run["planner"], "replan_count", 0)])
        ax.axhspan(-0.5, 0.5, color="gray", alpha=0.2)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(scenario)
        ax.set_ylabel("x position (cm)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    save(fig, "fig7_replanner_variants.png")
    table("Experiment 7: replanner variants (default gains)",
          ["Scenario", "Variant", "Settling / overshoot", "Final error (cm)", "Final jitter (cm)", "Stable", "Replans"], rows)


# --------------------------------------------------------------------------- experiment 8: how much feedforward
def experiment_feedforward_gain():
    """Sweep the feedforward gain: over the 9 starting conditions, and on a circle."""
    gains = [0.0, 0.25, 0.5, 0.75, 1.0]
    radius, period = 5.0, 6.0
    omega = 2 * math.pi / period
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rows = []
    for method, color in (("trajectory", "tab:blue"), ("update", "tab:green")):
        score, over, lost = [], [], []
        for g in gains:
            settle_g, over_g, lost_g = [], [], 0
            for x0, v0 in CONDITIONS:
                r = metrics_of(simulate(method, (x0, 0), (v0, 0), accel_feedforward=g))
                settle_g.append(10.0 if np.isnan(r["settling_time_s"]) else r["settling_time_s"])
                over_g.append(r["overshoot_cm"])
                lost_g += r["fell"]
            s = float(np.mean(settle_g) + 2 * np.mean(over_g) + 20 * lost_g)
            score.append(s)
            over.append(float(np.mean(over_g)))
            lost.append(lost_g)
            ALL_RUNS.append({"experiment": "feedforward_gain", "scenario": f"gain {g}", "config": method,
                             "mean_settling_s": float(np.mean(settle_g)), "mean_overshoot_cm": over[-1],
                             "ball_lost": lost_g, "score": s})
            rows.append([NAMES[method], g, fmt(float(np.mean(settle_g))), fmt(over[-1]), lost_g, fmt(s, 1)])
        axes[0, 0].plot(gains, score, "o-", color=color, label=NAMES[method])
        axes[0, 1].plot(gains, over, "o-", color=color)
        axes[1, 0].plot(gains, lost, "o-", color=color)
    axes[0, 0].set_title("Score over the 9 starting conditions (lower is better)")
    axes[0, 1].set_title("Mean overshoot (cm)")
    axes[1, 0].set_title("Balls lost (out of 9)")
    axes[0, 0].legend(fontsize=8)
    table("Experiment 8a: feedforward gain over the 9 starting conditions (default PID gains, T = 3 s; "
          "score = mean settling + 2 x mean overshoot + 20 per lost ball)",
          ["Method", "Feedforward gain", "Mean settling (s)", "Mean overshoot (cm)", "Ball lost", "Score"], rows)

    # The circle, on an ideal plant and on the default plant
    plants = {"ideal plant (no delay, no lag, no noise)": Plant(camera_delay_frames=0, servo_tau=1e-3, noise_cm=0.0,
                                                               friction=0.3, timestamp_jitter=0.0),
              "default plant (100 ms camera delay)": Plant()}
    circle_rows = []
    for (label, plant), color in zip(plants.items(), ("tab:purple", "tab:orange")):
        errors = []
        for g in gains:
            run = simulate("trajectory", (radius, 0.0), (0.0, radius * omega), seconds=24.0, plant=plant,
                           accel_feedforward=g, reference_fn=circle_reference(radius, period))
            after = run["t"] >= 4.0
            errors.append(float(np.sqrt(np.mean(np.hypot(run["x"] - run["ref_x"], run["y"] - run["ref_y"])[after] ** 2))))
        axes[1, 1].plot(gains, errors, "o-", color=color, label=label)
        circle_rows.append([label] + [fmt(e) for e in errors])
    axes[1, 1].set_title(f"Circle (radius {radius:g} cm, period {period:g} s): RMS error (cm)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Feedforward gain")
        ax.axvline(FEEDFORWARD, color="gray", linestyle=":")
        ax.grid(True, alpha=0.3)
    save(fig, "fig8_feedforward_gain.png")
    table("Experiment 8b: circle tracking error (cm) by feedforward gain",
          ["Plant"] + [f"gain {g}" for g in gains], circle_rows)


# --------------------------------------------------------------------------- scoring a real run
def score_csv(path):
    """Score a run recorded by basic_controller.py (tuning_runs/csv/*.csv)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def column(name):
        return np.array([float(r[name]) if r.get(name, "") != "" else np.nan for r in rows])

    if "reference_x_cm" not in rows[0]:
        raise SystemExit("This CSV has no reference columns: record it with the updated basic_controller.py")
    on = (column("enabled") == 1) & (column("found") == 1)
    t = column("time_s")[on]
    x, y = column("x_cm")[on], column("y_cm")[on]
    target = (column("setpoint_x_cm")[on][-1], column("setpoint_y_cm")[on][-1])
    start = (x[0], y[0])
    result = compute_metrics(t, x, y, target, start, column("reference_x_cm")[on], column("reference_y_cm")[on],
                             column("tilt_x_deg")[on], column("tilt_y_deg")[on])
    first = rows[int(np.where(on)[0][0])]    # the settings in force once control started
    print(f"{path}: method {first.get('method', '?')}, feedforward {first.get('feedforward', '?')}, "
          f"duration {first.get('duration_s', '?')} s, {len(t)} samples while controlling")
    for key, value in result.items():
        print(f"  {key:22s} {value if isinstance(value, bool) else fmt(value)}")
    return result


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller tuning grid")
    ap.add_argument("--csv", help="score a recorded run instead of running the simulation")
    args = ap.parse_args()
    if args.csv:
        score_csv(args.csv)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    tuned = {}

    def tuning():
        tuned.update(experiment_tuning(args.quick))

    for name, function in [("1: the 10 cm move", experiment_headline),
                           ("2: initial position and velocity", experiment_initial_conditions),
                           ("3: disturbances", experiment_robustness),
                           ("4: tuning vs method (this one takes a while)", tuning),
                           ("5: planner settings", experiment_planner_settings),
                           ("6: circle", experiment_circle),
                           ("7: replanner variants", experiment_replanner_variants),
                           ("8: how much feedforward", experiment_feedforward_gain),
                           ("9: tuned settings on disturbed plants", lambda: experiment_tuned_robustness(tuned))]:
        print(f"Experiment {name}", flush=True)
        function()

    # Everything to files
    columns = []
    for row in ALL_RUNS:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ALL_RUNS)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Simulation results (generated by trajectory_study.py)\n\n" + "\n".join(TABLES))
    print(f"\nwrote results.csv ({len(ALL_RUNS)} runs) and summary.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
