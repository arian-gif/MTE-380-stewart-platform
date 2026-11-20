"""Compare PID gains for a step to plate centre, or replay one recorded step trial."""
import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from metrics import compute_metrics
from pid import CONTROLLER_DEFAULT_GAINS
from simulation import DT, simulate, metrics_of

OUTPUT = Path(__file__).resolve().parent / "tuning_runs" / "overshoot"


def response(start, velocity, gains, seconds=10):
    return simulate("step", start, velocity, gains=dict(zip(("kp", "ki", "kd"), gains)), seconds=seconds)


def sweep(output):
    rows = []
    ranked = []
    for gains in product((0.3, 0.5, 0.75), (0.0, 0.05), (0.25, 0.35, 0.45, 0.55)):
        trials = []
        for x, vx in product((5, 10, 12), (-15, 0, 15)):
            score = metrics_of(response((x, 0), (vx, 0), gains))
            trials.append(score)
            rows.append(dict(kp=gains[0], ki=gains[1], kd=gains[2], x_cm=x, vx_cm_s=vx,
                             **{k: score[k] for k in ("overshoot_cm", "overshoot_pct", "settling_time_s", "fell")}))
        if all(not m["fell"] and np.isfinite(m["settling_time_s"]) for m in trials):
            ranked.append((max(m["overshoot_cm"] for m in trials),
                           max(m["settling_time_s"] for m in trials), gains))
    with (output / "gain_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if not ranked:
        raise ValueError("No candidate settled in every trial.")
    peak, settling, best = min(ranked)
    print(f"Selected Kp, Ki, Kd = {best}. Worst overshoot {peak:.3f} cm; worst settling {settling:.2f} s.")
    return best


def plot(output, series):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 4.5), tight_layout=True)
    ax = fig.subplots()
    ax.axhline(0, color="black", linewidth=0.8)
    for label, t, positions in series:
        start = positions[0]
        signed = -(positions @ start) / np.linalg.norm(start)
        ax.plot(t, signed, label=f"{label} (peak {max(0, signed.max()):.2f} cm)")
    ax.set(xlabel="Time (s)", ylabel="Travel past centre (cm)", title="Step response to plate centre")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output)


def recorded(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    enabled = [i for i, r in enumerate(rows) if r["enabled"] == "1"]
    if not enabled or enabled[-1] - enabled[0] + 1 != len(enabled):
        raise ValueError("Record one continuous enabled trial per CSV.")
    trial = rows[enabled[0]:enabled[-1] + 1]
    if any(r["method"] != "step" or float(r["setpoint_x_cm"]) != 0 or
           float(r["setpoint_y_cm"]) != 0 or r["found"] != "1" for r in trial):
        raise ValueError("Comparison requires a tracked step to centre for the whole enabled trial.")
    values = np.array([[float(r[k]) for k in ("time_s", "x_cm", "y_cm", "vx_cm_s", "vy_cm_s",
                                              "kp", "ki", "kd")] for r in trial])
    t = values[:, 0] - values[0, 0]
    if (len(t) < 2 or not np.isfinite(values).all() or np.any(np.diff(t) <= 0)
            or np.linalg.norm(values[0, 1:3]) <= 1e-9 or np.any(values[:, 5:] != values[0, 5:])):
        raise ValueError("Use a fixed-gain move starting away from centre with valid, increasing timestamps.")
    xy = values[:, 1:3]
    sample_count = int(np.floor(t[-1] / DT)) + 1
    seconds = np.nextafter(sample_count * DT, np.inf)
    model = response(xy[0], values[0, 3:5], values[0, 5:], seconds)
    measured = compute_metrics(t, *xy.T, (0, 0), xy[0])
    predicted = metrics_of(model)
    for label, score in (("Measured", measured), ("Model", predicted)):
        print(f"{label}: {score['overshoot_cm']:.3f} cm ({score['overshoot_pct']:.2f}%), "
              f"settling {score['settling_time_s']:.2f} s, fell={score['fell']}")
    return [("Measured", t, xy), ("Model", model["t"], np.column_stack((model["x"], model["y"])))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="compare one hardware trial against the same modelled move")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.csv:
        try:
            series = recorded(args.csv)
        except (KeyError, ValueError, OSError) as exc:
            parser.error(str(exc))
        plot(args.output / (args.csv.stem + "_comparison.png"), series)
    else:
        best = sweep(args.output)
        series = []
        for label, gains in (("Controller defaults", CONTROLLER_DEFAULT_GAINS), ("Selected gains", best)):
            run = response((10, 0), (0, 0), gains)
            series.append((label, run["t"], np.column_stack((run["x"], run["y"]))))
        plot(args.output / "centre_step.png", series)


if __name__ == "__main__":
    main()
