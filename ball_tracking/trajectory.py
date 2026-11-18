"""Trajectory planning for moving the ball to a target, following the Trajectory notes.

Three ways to move the ball from where it is to a target. They all have the same interface, so the
simulation (trajectory_study.py) and the real controller (basic_controller.py) use the same code:

    StepPlanner         the reference jumps straight to the target (one PID loop)
    TrajectoryPlanner   one smooth cubic Hermite path planned at the start; the PID follows it
                        (two loops: the PID and the position-update loop that walks along the path)
    ReplanningPlanner   the same, but the path is planned again from the *measured* ball state every
                        replan_interval seconds (three loops: PID, position update, replanning)

Each planner is called every control cycle as
    reference_position, reference_velocity = planner.reference(t, measured_position, measured_velocity, target)
and the PID then works on  e_p = reference_position - position  and  e_v = reference_velocity - velocity.
planner.acceleration(t) gives the acceleration the path is asking for, which the controller can feed forward
(as the tilt that produces it) so the plate starts braking when the plan says to, not after an error appears.

Units: cm, cm/s, seconds. Positions and velocities are arrays [x, y]; each axis is planned independently.
"""
import math

import numpy as np

G = 981.0                 # cm/s^2
ROLLING_FACTOR = 5 / 7    # a solid rolling ball accelerates at 5/7 * g * sin(tilt)
MAX_TILT_DEG = 10.0       # the platform accepts tilts up to this


def max_ball_acceleration(tilt_deg=MAX_TILT_DEG):
    """Largest acceleration the plate can give the ball (cm/s^2), at full tilt."""
    return ROLLING_FACTOR * G * math.sin(math.radians(tilt_deg))


# Plan paths that need at most this much acceleration, leaving the rest for the PID to correct errors
DEFAULT_ACCEL_LIMIT = 0.5 * max_ball_acceleration()

# Plan paths that stay this close to the plate centre (cm). The plate radius is 13.5 cm and the ball
# has a radius of its own, so keep a margin.
DEFAULT_EDGE_LIMIT = 12.0

# The most acceleration the plate can really give (full tilt). Only used when no comfortable path exists.
HARD_ACCEL_LIMIT = max_ball_acceleration()


# --------------------------------------------------------------------------- Hermite curve
def hermite(p0, v0, pf, vf, duration, t):
    """Cubic Hermite path from (p0, v0) to (pf, vf) over `duration`, evaluated at time t (from the start).

    Returns (position, velocity) at time t. After `duration` it stays at (pf, vf).
    Works on numbers or arrays (one entry per axis).
    """
    p0, v0, pf, vf = (np.asarray(a, dtype=float) for a in (p0, v0, pf, vf))
    s = min(max(t / duration, 0.0), 1.0)    # normalised time, 0..1

    # Hermite basis functions
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2

    position = h00 * p0 + h10 * duration * v0 + h01 * pf + h11 * duration * vf

    # Derivative with respect to time (the basis derivatives divided by the duration)
    velocity = ((6 * s**2 - 6 * s) / duration) * (p0 - pf) + (3 * s**2 - 4 * s + 1) * v0 + (3 * s**2 - 2 * s) * vf
    return position, velocity


def hermite_acceleration(p0, v0, pf, vf, duration, t):
    """Acceleration of the Hermite path at time t (zero once the path has ended)."""
    p0, v0, pf, vf = (np.asarray(a, dtype=float) for a in (p0, v0, pf, vf))
    if t >= duration:
        return np.zeros_like(p0)
    s = min(max(t / duration, 0.0), 1.0)
    return ((12 * s - 6) * p0 + (6 * s - 4) * duration * v0 + (-12 * s + 6) * pf
            + (6 * s - 2) * duration * vf) / duration ** 2


def tilt_for_acceleration(acceleration):
    """Plate tilt (deg) that gives the ball this acceleration (cm/s^2): a = 5/7 g sin(tilt)."""
    ratio = np.clip(np.asarray(acceleration, dtype=float) / (ROLLING_FACTOR * G), -1.0, 1.0)
    return np.degrees(np.arcsin(ratio))


def hermite_peak_acceleration(p0, v0, pf, vf, duration):
    """Largest acceleration (cm/s^2, over both axes) the Hermite path asks for.

    The acceleration of a cubic changes linearly with time, so its largest size is at the start or the end.
    """
    p0, v0, pf, vf = (np.asarray(a, dtype=float) for a in (p0, v0, pf, vf))
    T = duration
    a_start = (-6 * p0 - 4 * T * v0 + 6 * pf - 2 * T * vf) / T**2
    a_end = (6 * p0 + 2 * T * v0 - 6 * pf + 4 * T * vf) / T**2
    return float(max(np.max(np.abs(a_start)), np.max(np.abs(a_end))))


def hermite_path(p0, v0, pf, vf, duration, samples=60):
    """Positions along the Hermite path at samples + 1 evenly spaced times: an array (samples + 1, axes)."""
    p0, v0, pf, vf = (np.asarray(a, dtype=float) for a in (p0, v0, pf, vf))
    s = np.linspace(0.0, 1.0, samples + 1)[:, None]
    return ((2 * s**3 - 3 * s**2 + 1) * p0 + (s**3 - 2 * s**2 + s) * duration * v0
            + (-2 * s**3 + 3 * s**2) * pf + (s**3 - s**2) * duration * vf)


def hermite_peak_radius(p0, v0, pf, vf, duration, samples=60):
    """Furthest the Hermite path gets from the plate centre (cm)."""
    return float(np.max(np.linalg.norm(hermite_path(p0, v0, pf, vf, duration, samples), axis=1)))


def hermite_overshoot(p0, v0, pf, vf, duration, samples=60):
    """How far (cm) the path goes past the target, measured along the line from the start to the target.

    A cubic that starts out moving quickly towards the target and is given too long can swing past it
    and come back; this is the size of that swing.
    """
    p0, pf = np.asarray(p0, dtype=float), np.asarray(pf, dtype=float)
    gap = p0 - pf
    distance = np.linalg.norm(gap)
    if distance < 1e-6:
        return 0.0
    along = gap / distance    # points from the target back towards the start
    nearest = float(np.min((hermite_path(p0, v0, pf, vf, duration, samples) - pf) @ along))
    return max(0.0, -nearest)


def choose_feasible_duration(p0, v0, pf, vf, requested, accel_limit=DEFAULT_ACCEL_LIMIT,
                             edge_limit=DEFAULT_EDGE_LIMIT, overshoot_limit=0.5,
                             min_duration=0.3, max_duration=10.0, step=0.05):
    """The duration closest to `requested` whose path the plate can follow.

    A path is feasible if
      * it needs no more than accel_limit (the tilt can only give the ball so much acceleration),
      * it stays within edge_limit of the plate centre (the ball must not leave the plate), and
      * it does not go more than overshoot_limit past the target.
    A far-away or fast ball needs a longer path to stay within the acceleration limit. A ball moving
    away from the target, or towards it quickly, needs a *shorter* one: the path keeps the ball's own
    velocity at first, so a long path carries it far out, or past the target.

    If no comfortable path exists (for example a ball rolling quickly towards the edge), it is an
    emergency: take the path that goes least far past the limits among those the plate can still
    produce at full tilt.
    """
    # A ball that already starts beyond edge_limit can't be asked to stay inside it: allow a little more
    # than where it starts and where it is going
    if edge_limit is not None:
        edge_limit = max(edge_limit, np.linalg.norm(p0) + 0.5, np.linalg.norm(pf) + 0.5)

    def comfortable(duration):
        return (hermite_peak_acceleration(p0, v0, pf, vf, duration) <= accel_limit
                and (edge_limit is None or hermite_peak_radius(p0, v0, pf, vf, duration) <= edge_limit)
                and hermite_overshoot(p0, v0, pf, vf, duration) <= overshoot_limit)

    requested = max(requested, min_duration)
    if comfortable(requested):
        return requested
    candidates = np.arange(min_duration, max_duration + 1e-9, step)
    ok = [d for d in candidates if comfortable(d)]
    if ok:
        return float(min(ok, key=lambda d: abs(d - requested)))

    def badness(duration):
        radius = hermite_peak_radius(p0, v0, pf, vf, duration)
        too_far = 0.0 if edge_limit is None else max(0.0, radius - edge_limit)
        return too_far + hermite_overshoot(p0, v0, pf, vf, duration), radius

    producible = [d for d in candidates if hermite_peak_acceleration(p0, v0, pf, vf, d) <= HARD_ACCEL_LIMIT]
    return float(min(producible or candidates, key=badness))


def circular_reference(t, radius=5.0, period=8.0, center=(0.0, 0.0)):
    """Reference for a ball going round a circle: returns (position, velocity) at time t."""
    omega = 2 * math.pi / period
    center = np.asarray(center, dtype=float)
    position = center + radius * np.array([math.cos(omega * t), math.sin(omega * t)])
    velocity = radius * omega * np.array([-math.sin(omega * t), math.cos(omega * t)])
    return position, velocity


# --------------------------------------------------------------------------- the three methods
class StepPlanner:
    """Method (a): the reference is the target, always. One PID loop."""
    label = "Step"

    def reset(self):
        pass

    def reference(self, t, position, velocity, target):
        return np.asarray(target, dtype=float), np.zeros(2)

    def acceleration(self, t):
        return np.zeros(2)


class TrajectoryPlanner:
    """Method (b): plan one Hermite path from the ball's state when the move starts, then follow it.

    The move starts the first time reference() is called after reset(), and starts again if the
    target changes. After `duration` the reference holds at the target.
    """
    label = "Trajectory"

    def __init__(self, duration=3.0, accel_limit=DEFAULT_ACCEL_LIMIT, edge_limit=DEFAULT_EDGE_LIMIT):
        self.duration = duration
        self.accel_limit = accel_limit
        self.edge_limit = edge_limit
        self.reset()

    def reset(self):
        """Forget the current move; the next reference() call starts a new one."""
        self.start_time = None
        self.target = None

    def reference(self, t, position, velocity, target):
        target = np.asarray(target, dtype=float)
        if self.start_time is None or not np.allclose(target, self.target):
            self._start_move(t, position, velocity, target)
        return self._evaluate(t)

    def _start_move(self, t, position, velocity, target):
        self.start_time = t
        self.target = target
        self.start_position = np.asarray(position, dtype=float)
        self.start_velocity = np.asarray(velocity, dtype=float)
        # Keep the requested duration if the plate can follow that path, otherwise the nearest one it can
        self.planned_duration = choose_feasible_duration(self.start_position, self.start_velocity, target,
                                                         np.zeros(2), self.duration, self.accel_limit,
                                                         self.edge_limit)

    def _evaluate(self, t):
        return hermite(self.start_position, self.start_velocity, self.target, np.zeros(2),
                       self.planned_duration, t - self.start_time)

    def acceleration(self, t):
        """Acceleration of the path being followed at time t (for feedforward)."""
        if self.start_time is None:
            return np.zeros(2)
        return hermite_acceleration(self.start_position, self.start_velocity, self.target, np.zeros(2),
                                    self.planned_duration, t - self.start_time)


class ReplanningPlanner(TrajectoryPlanner):
    """Method (c): like the trajectory planner, but replanned from the measured state every so often.

    Every replan_interval seconds a new Hermite path is made from the measured position and velocity
    to the target (arriving with zero velocity). Its duration is the time left until the planned
    arrival, adjusted if needed so that the plate can follow the path.

    Two rules stop it misbehaving (a version that replans forever chatters, see trajectory_study.py):
      * no replanning in the final approach, the last `final_approach` seconds before the arrival: with
        so little time left a replan would demand a violent stop from a few millimetres of noise
      * after the arrival the target is simply held, unless the ball has been knocked more than
        `restart_distance` away, in which case a new move is planned
    """
    label = "Trajectory + replanning"

    def __init__(self, duration=3.0, replan_interval=0.5, accel_limit=DEFAULT_ACCEL_LIMIT,
                 edge_limit=DEFAULT_EDGE_LIMIT, final_approach=0.5, restart_distance=3.0, trigger=0.0):
        self.replan_interval = replan_interval
        self.final_approach = final_approach
        self.restart_distance = restart_distance
        self.trigger = trigger    # cm: if > 0, only replan when the ball is at least this far off the plan
        super().__init__(duration, accel_limit, edge_limit)

    def reset(self):
        super().reset()
        self.last_replan_time = None
        self.arrival_time = None
        self.replan_count = 0

    def _start_move(self, t, position, velocity, target):
        super()._start_move(t, position, velocity, target)
        self.arrival_time = t + self.planned_duration
        self.last_replan_time = t

    def reference(self, t, position, velocity, target):
        target = np.asarray(target, dtype=float)
        if self.start_time is not None and np.allclose(target, self.target):
            position = np.asarray(position, dtype=float)
            velocity = np.asarray(velocity, dtype=float)
            time_left = self.arrival_time - t
            off_plan = np.linalg.norm(position - self._evaluate(t)[0])
            if time_left > self.final_approach and t - self.last_replan_time >= self.replan_interval                     and off_plan >= self.trigger:
                self._replan(t, position, velocity)
            elif time_left <= 0 and np.linalg.norm(position - self.target) > self.restart_distance:
                self._start_move(t, position, velocity, target)   # knocked far off: plan a new move
        return super().reference(t, position, velocity, target)

    def _replan(self, t, position, velocity):
        self.last_replan_time = t
        duration = choose_feasible_duration(position, velocity, self.target, np.zeros(2),
                                            self.arrival_time - t, self.accel_limit, self.edge_limit)
        self.start_time = t
        self.start_position = position
        self.start_velocity = velocity
        self.planned_duration = duration
        self.arrival_time = t + duration
        self.replan_count += 1


METHODS = {
    "step": StepPlanner,
    "trajectory": TrajectoryPlanner,
    "update": ReplanningPlanner,
}


def make_planner(method, **settings):
    """Planner for "step", "trajectory" or "update".

    Settings (all optional): duration, replan_interval, accel_limit, edge_limit, and for "update" also
    final_approach, restart_distance and trigger.
    """
    if method == "step":
        return StepPlanner()
    limits = {"accel_limit": settings.get("accel_limit", DEFAULT_ACCEL_LIMIT),
              "edge_limit": settings.get("edge_limit", DEFAULT_EDGE_LIMIT)}
    if method == "trajectory":
        return TrajectoryPlanner(settings.get("duration", 3.0), **limits)
    if method == "update":
        return ReplanningPlanner(settings.get("duration", 3.0), settings.get("replan_interval", 0.5), **limits,
                                 final_approach=settings.get("final_approach", 0.5),
                                 restart_distance=settings.get("restart_distance", 3.0),
                                 trigger=settings.get("trigger", 0.0))
    raise ValueError(f"method must be one of {sorted(METHODS)}")
