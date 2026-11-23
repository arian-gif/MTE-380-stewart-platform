# Digital twin

A simulated Stewart-platform ball balancer. It runs the **same planners, PID and filter as the real
controller** (from `../ball_tracking/`) against a model of the plate, the servos and the camera, so anyone can
try settings, compare methods and see what the tracking error does, without touching the hardware.

Run everything from the project root with the project's virtual environment (`.venv\Scripts\activate`).

| Command | What it does |
|---|---|
| `python digital_twin/twin_gui.py` | Sliders for everything, live plots (path, pointwise and cross-track error, tilt). Tick "Keep previous runs" to overlay a change on the old runs |
| `python digital_twin/run_twin.py` | One run from the command line (defaults: Trajectory from 10 cm to the centre) |
| `python digital_twin/run_twin.py --compare step,trajectory,update` | The same settings with each method, overlaid |
| `python digital_twin/run_twin.py --sweep kp=0.2,0.4,0.6,0.8` | One setting over a list of values (any field of `Settings` in `twin.py`) |
| `python digital_twin/compare_real.py <run.csv>` | A run recorded on the real platform next to the twin's run of the same settings |

Plots and a CSV of the last run go to `digital_twin/runs/`.

## What you can change

* **How the ball is moved:** method (`step`, `trajectory`, `update` = replanning, `center` = slows down near the
  target), shape (`none`, line, triangle, square, circle, infinity), shape size, lap time, move time,
  replan interval, cruise speed, slow-down distance and the error above which `center` replans.
* **The controller:** Kp, Ki, Kd and the acceleration feedforward gain.
* **The ball at the start:** position, velocity, run length, and an optional knock (`bump`, from Python).
* **The platform (assumptions, not measurements):** rolling friction, servo lag, camera delay in frames
  (1 frame = 33 ms), camera noise, and a bias tilt of the plate.

## What it reports

* **Pointwise error:** distance between the ball and the planned position at the same moment (late counts as wrong).
* **Cross-track error:** shortest distance between the ball and the planned path (timing does not count).
* Both as RMS and maximum. For a shape, the first 8 s (the ball going to the path) are not scored.
* For a move to a target: settling time, overshoot, final error, and peak tilt.

## From Python

```python
import sys; sys.path.insert(0, "digital_twin")
from twin import Settings, run, plot, summary

a = run(Settings(method="trajectory", shape="circle", start=(0, 0), kp=0.4, feedforward=0.75, seconds=30))
b = run(Settings(method="step",       shape="circle", start=(0, 0), kp=0.4, seconds=30))
print(summary(a)); print(summary(b))
plot([("trajectory", a), ("step", b)])
```

## Questions it can answer

* Which method wins for which start position and velocity? (`--compare`, change `--start`/`--velocity`.)
* How much does tuning help versus changing the method? (`--sweep kp=...`, then `--compare`.)
* How much does camera delay or friction hurt, and which method copes best? (`--delay`, `--friction`.)
* Can the plate follow a circle this fast? (shorten `--lap` until the error blows up.)

## Fitting the twin to the real platform

The platform numbers are guesses, so by default the twin and the platform will not agree. To fit it:

1. Record a run on the real platform (change nothing while it runs; stop between method or shape changes).
2. `python digital_twin/compare_real.py ball_tracking/tuning_runs/csv/run_xxx.csv --from 10 --to 25`
3. Change `--friction`, `--delay`, `--servo-tau`, `--noise` and `--bias` until the twin's ball behaves like the
   real one. The camera delay (about 100 ms assumed) is the biggest unknown; measure it if you can.
4. Use the fitted numbers as the defaults in the `Settings` class in `twin.py`.

Anything the twin predicts is only as good as these numbers: confirm important results on the real platform.

## Files

* `twin.py` — `Settings`, `run()`, `plot()`, `summary()`: the library everything else uses
* `twin_gui.py`, `run_twin.py`, `compare_real.py` — the three ways to run it
