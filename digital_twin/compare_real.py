"""Put a run recorded on the real platform next to the twin's simulation of the same settings.

    python digital_twin/compare_real.py                  the newest run in ball_tracking/tuning_runs/csv
    python digital_twin/compare_real.py ball_tracking/tuning_runs/csv/run_20261119_152750.csv
    python digital_twin/compare_real.py run.csv --friction 0.6 --delay 5 --from 12 --to 24

The twin starts from the ball's measured position and velocity at the beginning of the chosen time window and
uses the method, shape, gains and feedforward that were in force then (they are logged in the CSV). If the
simulated ball does not behave like the real one, change the platform numbers (--friction, --delay,
--servo-tau, --noise, --bias) until it does: that is how you fit the twin to your platform.

Use a run in which the settings did not change (stop between method or shape changes), or use --from/--to to
pick a stretch where they did not.
"""
import argparse
import csv
import os
from dataclasses import replace

import numpy as np

from twin import HERE, Settings, run, summary


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="a run from ball_tracking/tuning_runs/csv (default: the newest one)")
    ap.add_argument("--from", dest="t0", type=float, default=None, help="start of the window (s of the run)")
    ap.add_argument("--to", dest="t1", type=float, default=None, help="end of the window (s of the run)")
    ap.add_argument("--friction", type=float, default=0.3)
    ap.add_argument("--servo-tau", type=float, default=0.05)
    ap.add_argument("--delay", type=int, default=3, help="camera delay in frames")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--bias", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    if a.csv is None:
        folder = os.path.join(os.path.dirname(HERE), "ball_tracking", "tuning_runs", "csv")
        found = sorted((os.path.join(folder, n) for n in os.listdir(folder) if n.endswith(".csv")),
                       key=os.path.getmtime) if os.path.isdir(folder) else []
        if not found:
            raise SystemExit(f"no CSV given and none found in {folder}")
        # the newest run in which the platform was actually controlling the ball for a while
        for path in reversed(found):
            if sum(1 for r in load(path) if r["enabled"] == "1" and r["found"] == "1") >= 20:
                a.csv = path
                break
        else:
            raise SystemExit(f"none of the runs in {folder} has controlled samples")
        print(f"using the newest run with control on: {a.csv}")

    rows = [r for r in load(a.csv) if r["enabled"] == "1" and r["found"] == "1"]
    f = lambda r, k: float(r[k]) if r.get(k, "") != "" else float("nan")
    t_all = np.array([f(r, "time_s") for r in rows])
    keep = np.ones(len(rows), dtype=bool)
    if a.t0 is not None:
        keep &= t_all >= a.t0
    if a.t1 is not None:
        keep &= t_all <= a.t1
    rows = [r for r, k in zip(rows, keep) if k]
    if len(rows) < 20:
        raise SystemExit("not enough controlled samples in that window")

    first = rows[0]
    t = np.array([f(r, "time_s") for r in rows])
    t = t - t[0]
    x = np.array([f(r, "x_cm") for r in rows])
    y = np.array([f(r, "y_cm") for r in rows])
    shape = first.get("shape", "none") or "none"
    settings = Settings(
        method=first.get("method", "step"), shape=shape,
        size=f(first, "size_cm") if shape != "none" else 4.0,
        lap_time=f(first, "lap_time_s") if shape != "none" else 10.0,
        duration=f(first, "duration_s"), replan_interval=f(first, "replan_interval_s"),
        cruise_speed=f(first, "cruise_speed_cm_s") if "cruise_speed_cm_s" in first else 8.0,
        slow_distance=f(first, "slow_distance_cm") if "slow_distance_cm" in first else 4.0,
        kp=f(first, "kp"), ki=f(first, "ki"), kd=f(first, "kd"), feedforward=f(first, "feedforward"),
        start=(x[0], y[0]), velocity=(f(first, "vx_cm_s"), f(first, "vy_cm_s")),
        target=(f(rows[-1], "setpoint_x_cm"), f(rows[-1], "setpoint_y_cm")),
        seconds=float(t[-1]), friction=a.friction, servo_tau=a.servo_tau, camera_delay_frames=a.delay,
        noise_cm=a.noise, tilt_bias=tuple(a.bias))
    print(f"settings from the CSV: method {settings.method}, shape {settings.shape}, Kp {settings.kp}, "
          f"Ki {settings.ki}, Kd {settings.kd}, feedforward {settings.feedforward}")
    sim = run(settings)
    print("--- twin\n" + summary(sim))

    from metrics import distance_to_path, tracking_errors
    real_ref = np.column_stack([[f(r, "reference_x_cm") for r in rows], [f(r, "reference_y_cm") for r in rows]])
    ok = ~np.isnan(x)
    real = tracking_errors(t[ok], x[ok], y[ok], real_ref[ok, 0], real_ref[ok, 1], sim.path, sim.skip)
    print(f"--- real run\npointwise RMS {real['pointwise_rms_cm']:.2f} cm, cross-track RMS "
          f"{real['cross_track_rms_cm']:.2f} cm")

    import matplotlib
    if a.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(sim.path[:, 0], sim.path[:, 1], "k--", linewidth=2, label="planned path")
    axes[0].plot(x, y, color="tab:red", label="real ball")
    axes[0].plot(sim.x, sim.y, color="tab:blue", label="twin ball")
    axes[0].set_aspect("equal")
    axes[0].set_title("Path")
    axes[0].legend(fontsize=8)
    for ax, series, label in ((axes[1], (x, sim.x), "x"), (axes[2], (y, sim.y), "y")):
        ax.plot(t, series[0], color="tab:red", label="real ball")
        ax.plot(sim.t, series[1], color="tab:blue", label="twin ball")
        ax.plot(t, real_ref[:, 0 if label == "x" else 1], "k:", label="planned (real run)")
        ax.set_title(f"{label} position (cm)")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Real vs twin   (twin: friction {a.friction}, servo lag {a.servo_tau} s, camera delay {a.delay} frames)")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", "real_vs_twin.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
