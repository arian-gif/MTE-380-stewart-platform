"""The digital twin: a simulated Stewart-platform ball balancer you can run without any hardware.

It reuses the real controller's code (the planners in ball_tracking/trajectory.py, the PID in
ball_tracking/pid.py, the alpha-beta filter and the metrics), and replaces the platform and the camera by a
model: a solid ball rolling on the tilted plate (a = 5/7 g sin(tilt) minus friction), a servo that follows the
tilt command with a lag, and a camera that sees the ball a few frames late with some noise.

    from twin import Settings, run, plot

    result = run(Settings(method="trajectory", shape="circle", kp=0.4, feedforward=0.75))
    print(result.metrics)
    plot([("my run", result)])

Everything is in `Settings`. The numbers of the plant (friction, camera delay, ...) are assumptions: the
point of the twin is that you can change them to match what you measure on the real platform
(compare_real.py puts a recorded run next to the simulation of the same settings).
"""
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ball_tracking"))

import trajectory as tj                                   # noqa: E402
import trajectory_study as study                          # noqa: E402  (its simulate() is the engine)
from metrics import compute_metrics, tracking_errors      # noqa: E402

relevant_settings = tj.relevant_settings        # which settings a method and shape use (for the GUI)
METHODS = ["step", "trajectory", "update", "center"]
SHAPES = tj.SHAPES


@dataclass
class Settings:
    # ---- how the ball is moved
    method: str = "trajectory"        # step, trajectory, update (replanning) or center (slows down near the target)
    shape: str = "none"               # none = go to the target, or line, triangle, square, circle, infinity
    size: float = 4.0                 # cm, radius of the shape
    lap_time: float = 10.0            # s, one lap of a circle or infinity
    duration: float = 3.0             # s, planned time of a move (and of one side of a polygon)
    replan_interval: float = 0.5      # s, how often "update" replans
    cruise_speed: float = 8.0         # cm/s, top speed of "center"
    slow_distance: float = 4.0        # cm from the target where "center" starts slowing down
    replan_error: float = 3.0         # cm: "center" replans when the ball is this far off its plan (0 = never)
    # ---- the controller
    kp: float = 0.3                   # deg of tilt per cm of error
    ki: float = 0.05
    kd: float = 0.35                  # deg of tilt per cm/s of velocity error
    feedforward: float = 0.0          # 0 = PID only, 1 = full acceleration feedforward
    # ---- the ball at the start
    start: tuple = (10.0, 0.0)        # cm from the plate centre
    velocity: tuple = (0.0, 0.0)      # cm/s
    target: tuple = (0.0, 0.0)        # cm (the centre of the shape, if there is one)
    seconds: float = 20.0
    # ---- the plant (the assumptions about the platform)
    friction: float = 0.3             # 1/s, rolling resistance
    servo_tau: float = 0.05           # s, how quickly the plate follows the tilt command
    camera_delay_frames: int = 3      # frames (1 frame = 33 ms) between the real ball and the measurement
    noise_cm: float = 0.05            # std of the measured position
    tilt_bias: tuple = (0.0, 0.0)     # deg, tilt of the plate when it is commanded level
    bump: tuple = None                # (time s, x velocity change, y velocity change cm/s): knock the ball
    seed: int = 0


@dataclass
class Result:
    settings: Settings
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ref_x: np.ndarray
    ref_y: np.ndarray
    tilt_x: np.ndarray
    tilt_y: np.ndarray
    fell: bool
    path: np.ndarray                  # the planned path as a polyline (n, 2)
    skip: float                       # seconds not scored (the ball is still going to the path)
    metrics: dict = field(default_factory=dict)

    @property
    def pointwise(self):
        return np.hypot(self.x - self.ref_x, self.y - self.ref_y)

    @property
    def cross_track(self):
        from metrics import distance_to_path
        return distance_to_path(np.column_stack([self.x, self.y]), self.path)


def planned_path(s):
    """The path the ball is asked to follow, as a polyline."""
    if s.shape != "none":
        return np.asarray(s.target, dtype=float) + tj.shape_outline(s.shape, s.size, points=400)
    return np.array([s.start, s.target], dtype=float)


def run(s=None, **changes):
    """Simulate one run. `run(Settings(...))` or `run(kp=0.5, shape="circle")` (changes to the defaults)."""
    s = replace(s or Settings(), **changes)
    plant = study.Plant(friction=s.friction, servo_tau=s.servo_tau, camera_delay_frames=int(s.camera_delay_frames),
                        noise_cm=s.noise_cm, tilt_bias=tuple(s.tilt_bias), bump=s.bump)
    planner_settings = {"duration": s.duration, "replan_interval": s.replan_interval, "shape": s.shape,
                        "size": s.size, "lap_time": s.lap_time, "cruise_speed": s.cruise_speed,
                        "slow_distance": s.slow_distance, "replan_error": s.replan_error}
    raw = study.simulate(s.method, s.start, s.velocity, s.target,
                         gains={"kp": s.kp, "ki": s.ki, "kd": s.kd}, planner_settings=planner_settings, plant=plant,
                         seconds=s.seconds, seed=s.seed, accel_feedforward=s.feedforward)
    path = planned_path(s)
    skip = 8.0 if (s.shape != "none" and s.seconds > 10) else 0.0
    result = Result(s, raw["t"], raw["x"], raw["y"], raw["ref_x"], raw["ref_y"], raw["tilt_x"], raw["tilt_y"],
                    bool(raw["fell"]), path, skip)
    if len(result.t) > 5:
        move = compute_metrics(result.t, result.x, result.y, s.target, s.start, result.ref_x, result.ref_y,
                               result.tilt_x, result.tilt_y)
        move["fell"] = bool(raw["fell"] or move["fell"])
        result.metrics = {**move, **tracking_errors(result.t, result.x, result.y, result.ref_x, result.ref_y,
                                                     path, skip)}
    return result


def summary(result):
    """A few lines of text: the two tracking errors, and the move metrics when there is no shape."""
    m = result.metrics
    if not m:
        return "The ball left the plate almost at once."
    lines = [f"pointwise RMS {m['pointwise_rms_cm']:.2f} cm (max {m['pointwise_max_cm']:.2f})",
             f"cross-track RMS {m['cross_track_rms_cm']:.2f} cm (max {m['cross_track_max_cm']:.2f})"
             + (f"   [scored after {result.skip:g} s]" if result.skip else "")]
    if result.settings.shape == "none":
        settle = m["settling_time_s"]
        lines.append(f"settling time {'never' if settle != settle else f'{settle:.2f} s'}, "
                     f"overshoot {m['overshoot_cm']:.2f} cm, final error {m['final_error_cm']:.2f} cm")
    lines.append(f"peak tilt {m['peak_tilt_deg']:.1f} deg" + ("   BALL LEFT THE PLATE" if result.fell else ""))
    return "\n".join(lines)


def draw(fig, runs):
    """Draw runs on a matplotlib Figure. runs = [(label, Result), ...]; several runs are overlaid."""
    fig.clear()
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    xy = fig.add_subplot(grid[:, 0])
    err = fig.add_subplot(grid[0, 1])
    tilt = fig.add_subplot(grid[1, 1], sharex=err)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    first = runs[-1][1]
    xy.plot(first.path[:, 0], first.path[:, 1], "k--", linewidth=2, label="planned path")
    xy.add_patch(__import__("matplotlib").patches.Circle((0, 0), study.PLATE_RADIUS_CM, fill=False, color="gray"))
    for i, (label, r) in enumerate(runs):
        color = colors[i % len(colors)]
        xy.plot(r.x, r.y, color=color, linewidth=1.3, label=label)
        err.plot(r.t, r.pointwise, color=color, linewidth=1.3, label=f"{label}: pointwise")
        err.plot(r.t, r.cross_track, color=color, linewidth=1.3, linestyle=":", label=f"{label}: cross-track")
        tilt.plot(r.t, r.tilt_x, color=color, linewidth=1.0)
        tilt.plot(r.t, r.tilt_y, color=color, linewidth=1.0, linestyle="--")
    xy.set_aspect("equal")
    xy.set_xlim(-study.PLATE_RADIUS_CM, study.PLATE_RADIUS_CM)
    xy.set_ylim(-study.PLATE_RADIUS_CM, study.PLATE_RADIUS_CM)
    xy.set_xlabel("x (cm)")
    xy.set_ylabel("y (cm)")
    xy.set_title("Ball path on the plate")
    xy.legend(fontsize=7, loc="upper right")
    xy.grid(True, alpha=0.3)
    err.set_ylabel("error (cm)")
    err.set_title("Tracking error (solid: pointwise, dotted: cross-track)", fontsize=10)
    err.legend(fontsize=6, loc="upper right")
    err.grid(True, alpha=0.3)
    tilt.set_ylabel("tilt command (deg)")
    tilt.set_xlabel("time (s)")
    tilt.set_title("Plate tilt (solid: x, dashed: y)", fontsize=10)
    tilt.grid(True, alpha=0.3)
    fig.tight_layout()


def plot(runs, path=None, show=True):
    """Plot and/or save runs: plot([("label", result), ...], path="out.png")."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6.5))
    draw(fig, runs)
    if path:
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig
