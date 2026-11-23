"""Run the digital twin from the command line, no GUI.

    python digital_twin/run_twin.py                                   defaults: trajectory to the centre from 10 cm
    python digital_twin/run_twin.py --shape circle --kp 0.5 --ff 0.75
    python digital_twin/run_twin.py --compare step,trajectory,update  the same settings with each method, overlaid
    python digital_twin/run_twin.py --sweep kp=0.2,0.4,0.6,0.8        one setting changed over a list of values
    python digital_twin/run_twin.py --friction 0.8 --delay 5          a different platform

Prints the two tracking errors (pointwise and cross-track, RMS) for each run and saves the plot and a CSV of
the last run to digital_twin/runs/. Add --no-show to skip the plot window.
"""
import argparse
import csv
import os
from dataclasses import fields, replace

from twin import HERE, METHODS, SHAPES, Settings, plot, run, summary


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, default="trajectory")
    ap.add_argument("--shape", choices=SHAPES, default="none")
    ap.add_argument("--size", type=float, default=4.0, help="cm, radius of the shape")
    ap.add_argument("--lap", type=float, default=10.0, help="s, lap time of circle / infinity")
    ap.add_argument("--duration", type=float, default=3.0, help="s, time of a move or of one side")
    ap.add_argument("--replan-error", type=float, default=3.0,
                    help="cm: the centre method replans when the ball is this far off its plan (0 = never)")
    ap.add_argument("--kp", type=float, default=0.3)
    ap.add_argument("--ki", type=float, default=0.05)
    ap.add_argument("--kd", type=float, default=0.35)
    ap.add_argument("--ff", type=float, default=0.0, help="acceleration feedforward, 0..1")
    ap.add_argument("--start", type=float, nargs=2, default=(10.0, 0.0), metavar=("X", "Y"))
    ap.add_argument("--velocity", type=float, nargs=2, default=(0.0, 0.0), metavar=("VX", "VY"))
    ap.add_argument("--target", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--friction", type=float, default=0.3)
    ap.add_argument("--servo-tau", type=float, default=0.05)
    ap.add_argument("--delay", type=int, default=3, help="camera delay in frames (33 ms each)")
    ap.add_argument("--noise", type=float, default=0.05, help="cm")
    ap.add_argument("--bias", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"), help="plate tilt bias, deg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", help="comma-separated methods to overlay, e.g. step,trajectory,update")
    ap.add_argument("--sweep", help="name=v1,v2,... e.g. kp=0.2,0.4,0.6 (any setting name from twin.Settings)")
    ap.add_argument("--no-show", action="store_true")
    return ap.parse_args()


def main():
    a = parse()
    base = Settings(method=a.method, shape=a.shape, size=a.size, lap_time=a.lap, duration=a.duration, replan_error=a.replan_error, kp=a.kp,
                    ki=a.ki, kd=a.kd, feedforward=a.ff, start=tuple(a.start), velocity=tuple(a.velocity),
                    target=tuple(a.target), seconds=a.seconds, friction=a.friction, servo_tau=a.servo_tau,
                    camera_delay_frames=a.delay, noise_cm=a.noise, tilt_bias=tuple(a.bias), seed=a.seed)

    if a.compare:
        variants = [(m, replace(base, method=m)) for m in a.compare.split(",")]
    elif a.sweep:
        name, values = a.sweep.split("=")
        kind = {f.name: f.type for f in fields(Settings)}[name]
        cast = int if kind is int else float
        variants = [(f"{name}={v}", replace(base, **{name: cast(v)})) for v in values.split(",")]
    else:
        variants = [(f"{base.method}" + (f", {base.shape}" if base.shape != "none" else ""), base)]

    results = []
    for label, settings in variants:
        result = run(settings)
        results.append((label, result))
        print(f"--- {label}\n{summary(result)}")

    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    png = os.path.join(HERE, "runs", "last_run.png")
    plot(results, path=png, show=not a.no_show)
    last = results[-1][1]
    with open(os.path.join(HERE, "runs", "last_run.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "x_cm", "y_cm", "reference_x_cm", "reference_y_cm", "tilt_x_deg", "tilt_y_deg"])
        for row in zip(last.t, last.x, last.y, last.ref_x, last.ref_y, last.tilt_x, last.tilt_y):
            writer.writerow([f"{v:.4f}" for v in row])
    print(f"saved {png} and runs/last_run.csv")


if __name__ == "__main__":
    main()
