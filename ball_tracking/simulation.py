"""Ball, camera and servo model shared by the controller studies."""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

import trajectory as tj
from metrics import PLATE_RADIUS_CM, compute_metrics
from pid import AxisPID

DT = 1 / 30   # one control cycle: the camera runs at 30 fps

DEFAULT_GAINS = {"kp": 0.75, "ki": 0.0, "kd": 0.35}
DEFAULT_PLANNER = {"duration": 3.0, "replan_interval": 0.5, "final_approach": 0.5, "restart_distance": 3.0,
                   "trigger": 0.0}


@dataclass
class Plant:
    """The assumptions about the platform. Change these to match measurements."""
    friction: float = 0.3             # 1/s, rolling resistance (acceleration lost = friction * speed)
    servo_tau: float = 0.05           # s, how quickly the plate follows the tilt command
    camera_delay_frames: int = 3      # frames between the real ball and the measurement (3 = 100 ms)
    noise_cm: float = 0.05            # standard deviation of the measured position (about 0.6 pixel)
    tilt_bias: tuple = (0.0, 0.0)     # deg, tilt of the plate when it is commanded level
    timestamp_jitter: float = 0.003   # s, camera timestamps are not perfectly regular
    bump: tuple = None                # (time in s, x velocity change, y velocity change in cm/s): the ball is knocked


class AlphaBeta:
    """The same alpha-beta filter as BallTracker in ball_detection.py."""

    def __init__(self, alpha=0.5, max_gap=0.3):
        self.alpha = alpha
        self.beta = alpha ** 2 / (2 - alpha)
        self.max_gap = max_gap
        self.position = None
        self.velocity = np.zeros(2)
        self.last_time = None

    def update(self, measured, t):
        measured = np.asarray(measured, dtype=float)
        dt = None if self.last_time is None else t - self.last_time
        if self.position is None or dt is None or not (0 < dt <= self.max_gap):
            self.position = measured.copy()
            self.velocity = np.zeros(2)
        else:
            predicted = self.position + self.velocity * dt
            error = measured - predicted
            self.position = predicted + self.alpha * error
            self.velocity = self.velocity + (self.beta / dt) * error
        self.last_time = t
        return self.position, self.velocity


# --------------------------------------------------------------------------- the simulation
def simulate(method, start, velocity=(0.0, 0.0), target=(0.0, 0.0), gains=None, planner_settings=None,
             plant=None, seconds=10.0, seed=0, reference_fn=None, use_reference_velocity=True,
             accel_feedforward=0.0):
    """Move the simulated ball to the target with one of the methods. Returns a dict of arrays.

    reference_fn(t) -> (position, velocity) or (position, velocity, acceleration), if given, replaces the
    planner (used for the circle).
    accel_feedforward: 0 = the PID alone (as in the notes), 1 = also add the tilt that produces the
    planned acceleration of the path.
    """
    gains = {**DEFAULT_GAINS, **(gains or {})}
    settings = {**DEFAULT_PLANNER, **(planner_settings or {})}
    plant = plant or Plant()
    rng = np.random.default_rng(seed)

    planner = tj.make_planner(method, **settings)
    pids = [AxisPID(), AxisPID()]
    tracker = AlphaBeta()
    start = np.array(start, dtype=float)
    velocity = np.array(velocity, dtype=float)
    target = np.array(target, dtype=float)
    bias = np.array(plant.tilt_bias, dtype=float)

    # The camera sees the ball a few frames late, with noise
    history = deque(maxlen=plant.camera_delay_frames + 1)

    def measure(true_position):
        history.append(true_position.copy())
        return history[0] + rng.normal(0.0, plant.noise_cm, 2)

    # Warm-up: the ball rolls on a level plate and the tracker locks on, so the velocity estimate is
    # right when the control starts (on the real platform the tracker runs before you enable control)
    for k in range(-10, 0):
        tracker.update(measure(start + velocity * k * DT), k * DT)

    position, ball_velocity, tilt = start.copy(), velocity.copy(), np.zeros(2)
    keys = ["t", "x", "y", "vx", "vy", "ref_x", "ref_y", "tilt_x", "tilt_y"]
    record = {key: [] for key in keys}
    last_t = None
    fell = False
    bumped = False

    for k in range(int(seconds / DT)):
        t = k * DT + rng.uniform(-plant.timestamp_jitter, plant.timestamp_jitter)
        est_position, est_velocity = tracker.update(measure(position), t)

        # Where should the ball be now? (the method decides)
        ref_acceleration = np.zeros(2)
        if reference_fn is not None:
            reference = reference_fn(t)
            ref_position, ref_velocity = reference[0], reference[1]
            if len(reference) > 2:
                ref_acceleration = reference[2]
        else:
            ref_position, ref_velocity = planner.reference(t, est_position, est_velocity, target)
            ref_acceleration = planner.acceleration(t)
        if not use_reference_velocity:
            ref_velocity = np.zeros(2)

        # PID on each axis -> tilt command (plus the feedforward tilt for the planned acceleration)
        dt = DT if last_t is None else t - last_t
        last_t = t
        command = np.array([pids[axis].update(gains["kp"], gains["ki"], gains["kd"], ref_position[axis],
                                              est_position[axis], est_velocity[axis], dt, ref_velocity[axis])
                            for axis in range(2)])
        if accel_feedforward:
            command = command + accel_feedforward * tj.tilt_for_acceleration(ref_acceleration)
        command = np.clip(command, -tj.MAX_TILT_DEG, tj.MAX_TILT_DEG)

        for key, value in zip(keys, [k * DT, position[0], position[1], ball_velocity[0], ball_velocity[1],
                                     ref_position[0], ref_position[1], command[0], command[1]]):
            record[key].append(value)

        if plant.bump is not None and not bumped and k * DT >= plant.bump[0]:
            ball_velocity = ball_velocity + np.array(plant.bump[1:], dtype=float)   # something knocks the ball
            bumped = True

        # The plate follows the command with a lag, and the ball rolls for one cycle
        tilt = tilt + (command - tilt) * (1 - math.exp(-DT / plant.servo_tau))
        h = DT / 5
        for _ in range(5):
            acceleration = tj.ROLLING_FACTOR * tj.G * np.sin(np.radians(tilt + bias)) - plant.friction * ball_velocity
            ball_velocity = ball_velocity + acceleration * h
            position = position + ball_velocity * h
        if np.linalg.norm(position) > PLATE_RADIUS_CM:
            fell = True
            break

    run = {key: np.array(values) for key, values in record.items()}
    run.update(start=start, target=target, fell=fell, planner=planner)
    return run


def metrics_of(run, **kwargs):
    result = compute_metrics(run["t"], run["x"], run["y"], run["target"], run["start"],
                             run["ref_x"], run["ref_y"], run["tilt_x"], run["tilt_y"], **kwargs)
    # The simulation stops the moment the ball leaves the plate, so that position is never recorded and
    # the metrics cannot see it: take the fall from the simulation itself
    result["fell"] = bool(run["fell"] or result["fell"])
    result["stable"] = bool(result["stable"] and not result["fell"])
    return result
