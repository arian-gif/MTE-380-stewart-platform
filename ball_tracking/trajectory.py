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


# --------------------------------------------------------------------------- shapes (paths round the target)
# A shape is drawn round the target (the setpoint is its centre). size is the radius in cm: the distance
# from the centre to the corners (polygons) or the radius of the circle.
POLYGONS = {"line": 2, "triangle": 3, "square": 4}
CONTINUOUS = ("circle", "infinity")
SHAPES = ["none", "line", "triangle", "square", "circle", "infinity"]


def polygon_vertices(shape, size):
    """Corners of the shape relative to its centre (cm): an array (corners, 2).

    A triangle points towards +y (like the +Y dot of the calibration), the square
    sits with flat sides along x and y, and the line runs along x.
    """
    n = POLYGONS[shape]
    if shape == "line":
        return np.array([[size, 0.0], [-size, 0.0]])
    first = math.pi / 4 if n == 4 else math.pi / 2
    angles = first + 2 * math.pi * np.arange(n) / n
    return size * np.column_stack([np.cos(angles), np.sin(angles)])


def continuous_reference(shape, size, lap_time, tau):
    """(position, velocity, acceleration) relative to the centre, tau seconds into a lap of a circle or an infinity sign.

    The circle starts at +x and goes anticlockwise. The infinity sign starts at the centre:
    x = r sin(w t), y = (r/2) sin(2 w t).
    """
    w = 2 * math.pi / lap_time
    if shape == "circle":
        c, s = math.cos(w * tau), math.sin(w * tau)
        return (size * np.array([c, s]), size * w * np.array([-s, c]), -size * w**2 * np.array([c, s]))
    s1, c1, s2, c2 = math.sin(w * tau), math.cos(w * tau), math.sin(2 * w * tau), math.cos(2 * w * tau)
    return (np.array([size * s1, size / 2 * s2]), np.array([size * w * c1, size * w * c2]),
            np.array([-size * w**2 * s1, -2 * size * w**2 * s2]))


def shortest_lap_time(shape, size, accel_limit):
    """Quickest lap the plate can drive: the ball's acceleration is w^2 r (circle) or 2 w^2 r (infinity)."""
    factor = 1.0 if shape == "circle" else 2.0
    return 2 * math.pi * math.sqrt(factor * size / accel_limit)


def shape_outline(shape, size, points=80):
    """Points along the whole shape relative to its centre (for drawing it): an array (points, 2)."""
    if shape in POLYGONS:
        corners = polygon_vertices(shape, size)
        return np.vstack([corners, corners[:1]]) if shape != "line" else corners
    return np.array([continuous_reference(shape, size, 1.0, k / points)[0] for k in range(points + 1)])


class ShapePlanner:
    """Moves the ball round a shape drawn about the target, using one of the three methods for the moves.

    Polygons (line, triangle, square, ...): the ball goes from corner to corner. Each leg is a move with
    the chosen method (a step to the next corner, a Hermite path to it, or a replanned one), so the ball
    stops at each corner for a moment. The next leg starts when the path of the leg has finished and the
    ball is close, or after a time limit so a stuck ball cannot stall the lap.

    Circle and infinity sign: the ball first goes to the start of the shape with the chosen method, then
    follows a path that is a function of time, over and over. With the step method the reference is just
    the point where the ball should be now (no reference velocity and no feedforward); with the trajectory
    methods it also carries the velocity and the acceleration of the path.

    If the ball is knocked more than `lost_distance` off the path it goes back to the first corner / the start.
    The planner has the same interface as the others, with the target as the centre of the shape.
    """

    def __init__(self, inner, shape, size=4.0, lap_time=10.0, arrive_distance=1.2, lost_distance=4.0,
                 accel_limit=DEFAULT_ACCEL_LIMIT):
        self.inner = inner
        self.shape = shape
        self.size = size
        self.lap_time = lap_time
        self.arrive_distance = arrive_distance
        self.lost_distance = lost_distance
        self.accel_limit = accel_limit
        self.label = f"{inner.label}, {shape}"
        self.reset()

    # The controller changes duration / replan_interval on the planner it holds: pass them to the inner one
    @property
    def duration(self):
        return self.inner.duration

    @duration.setter
    def duration(self, value):
        self.inner.duration = value

    @property
    def replan_interval(self):
        return getattr(self.inner, "replan_interval", None)

    @replan_interval.setter
    def replan_interval(self, value):
        if hasattr(self.inner, "replan_interval"):
            self.inner.replan_interval = value

    def reset(self):
        self.inner.reset()
        self.corner = 0               # polygons: the corner the ball is heading for
        self.leg_start = None
        self.lap_start = None         # continuous shapes: when the lap began (None while going to the start)
        self.laps = 0
        self._acceleration = np.zeros(2)

    # ---- geometry
    def effective_lap_time(self):
        """The lap time asked for, lengthened if the plate could not drive it."""
        if self.shape in CONTINUOUS:
            return max(self.lap_time, shortest_lap_time(self.shape, self.size, self.accel_limit))
        return self.lap_time

    def outline(self, center):
        """The shape drawn about `center`, for the camera view."""
        return np.asarray(center, dtype=float) + shape_outline(self.shape, self.size)

    # ---- planner interface
    def reference(self, t, position, velocity, center):
        center = np.asarray(center, dtype=float)
        position = np.asarray(position, dtype=float)
        self._acceleration = np.zeros(2)
        if self.shape in CONTINUOUS:
            return self._continuous(t, position, velocity, center)
        return self._polygon(t, position, velocity, center)

    def _arrived(self, t, position, goal):
        """Has the leg towards `goal` finished? The path is over and the ball is close, or time ran out."""
        if self.leg_start is None:
            return False
        path_over = t - self.leg_start >= getattr(self.inner, "planned_duration", 0.0)
        close = np.linalg.norm(position - goal) <= self.arrive_distance
        if isinstance(self.inner, StepPlanner):
            path_over = True            # nothing is planned: being close is enough
        return (path_over and close) or t - self.leg_start > self.inner.duration + 2.0

    def _polygon(self, t, position, velocity, center):
        corners = polygon_vertices(self.shape, self.size)
        goal = center + corners[self.corner]
        if self.leg_start is None:
            self.leg_start = t
        reference = self.inner.reference(t, position, velocity, goal)
        if self._arrived(t, position, goal):
            self.corner = (self.corner + 1) % len(corners)
            if self.corner == 0:
                self.laps += 1
            self.leg_start = t
        return reference

    def _continuous(self, t, position, velocity, center):
        start = center + continuous_reference(self.shape, self.size, 1.0, 0.0)[0]
        if self.lap_start is None:                       # going to the start of the shape
            if self.leg_start is None:
                self.leg_start = t
            reference = self.inner.reference(t, position, velocity, start)
            if self._arrived(t, position, start):
                self.lap_start = t
            return reference
        lap = self.effective_lap_time()
        tau = t - self.lap_start
        self.laps = int(tau // lap)
        p, v, a = continuous_reference(self.shape, self.size, lap, tau)
        p = center + p
        if np.linalg.norm(position - p) > self.lost_distance:   # knocked off, or fell behind: start again
            self.inner.reset()
            self.leg_start = None
            self.lap_start = None
            return self._continuous(t, position, velocity, center)
        if isinstance(self.inner, StepPlanner):
            return p, np.zeros(2)
        self._acceleration = a
        return p, v

    def acceleration(self, t):
        if self.shape in CONTINUOUS and self.lap_start is not None:
            return self._acceleration
        return self.inner.acceleration(t)


class CenterPlanner:
    """Method (d): go straight to the target at a cruise speed, slowing down smoothly as it gets close.

    The path is a straight line from where the ball is to the target. Along it the speed asked for is
        speed = min(cruise_speed, distance_left / tau),   tau = slow_distance / cruise_speed
    so the ball cruises until it is slow_distance from the target, then its speed falls in proportion to
    the distance left (an exponential approach, like a car easing off the gas): fast far away, gentle at
    the end, and it never arrives with a bump. The actual speed follows that asked-for speed with the
    plate's acceleration limit, so the start (from rest or from a moving ball) and the braking are smooth.

    The profile is worked out when the move starts (and again if the target changes). It is replanned from
    the measured ball position and velocity only when the ball gets further than `replan_error` from where the
    plan says it should be (a knock, a slip, a bad start), never on a timer; `replan_gap` is the shortest
    time between two replans, so noise and lag cannot make it chatter. replan_error = 0 turns replanning off.
    After the plan ends the reference holds at the target (and a ball knocked away from it is replanned too).
    """
    label = "Trajectory to centre"

    def __init__(self, cruise_speed=8.0, slow_distance=4.0, accel_limit=DEFAULT_ACCEL_LIMIT, replan_error=0.0,
                 replan_gap=1.0):
        self.cruise_speed = cruise_speed        # cm/s
        self.slow_distance = slow_distance      # cm from the target where the slowing down starts
        self.replan_error = replan_error        # cm: replan when the ball is this far off the plan (0 = never)
        self.replan_gap = replan_gap            # s, at least this long between replans
        self.accel_limit = accel_limit
        self.duration = 3.0                     # unused here (the speed sets the time); kept so the controller can set it
        self.reset()

    def reset(self):
        self.start_time = None
        self.target = None
        self.planned_duration = 0.0
        self.replan_count = 0
        self.last_plan_time = None

    def reference(self, t, position, velocity, target):
        target = np.asarray(target, dtype=float)
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if self.start_time is None or not np.allclose(target, self.target):
            self.replan_count = 0
            self._plan(t, position, velocity, target)
        elif (self.replan_error > 0 and t - self.last_plan_time >= self.replan_gap
              and np.linalg.norm(position - self._evaluate(t)[0]) > self.replan_error):
            self.replan_count += 1
            self._plan(t, position, velocity, target)    # off the plan: plan again from where the ball is
        return self._evaluate(t)

    def _plan(self, t, position, velocity, target):
        self.start_time = t
        self.last_plan_time = t
        self.target = target
        gap = position - target                       # from the target back to the ball
        distance = float(np.linalg.norm(gap))
        self.start_position = position
        if distance < 0.05:                           # already there
            self.direction = np.zeros(2)
            self.times = np.array([0.0, 1.0])
            self.left = np.zeros(2)
            self.speeds = np.zeros(2)
            self.planned_duration = 0.0
            return
        self.direction = -gap / distance              # unit vector towards the target
        cruise = max(self.cruise_speed, 0.5)
        tau = max(self.slow_distance, 0.5) / cruise
        speed = float(velocity @ self.direction)      # the ball's speed towards the target now
        step = 0.01
        times, left, speeds = [0.0], [distance], [speed]
        r = distance
        while r > 0.05 and times[-1] < 20.0:
            wanted = min(cruise, r / tau)
            # follow the asked-for speed, no faster than the plate can change it
            speed += float(np.clip(wanted - speed, -self.accel_limit * step, self.accel_limit * step))
            r -= speed * step
            r = min(r, self.edge_distance(distance))
            times.append(times[-1] + step)
            left.append(max(r, 0.0))
            speeds.append(speed)
        self.times = np.array(times)
        self.left = np.array(left)
        self.speeds = np.array(speeds)
        self.planned_duration = float(self.times[-1])

    @staticmethod
    def edge_distance(distance):
        """Never let the ball be asked to go further from the target than 1.5x where it started."""
        return 1.5 * distance + 1.0

    def _evaluate(self, t):
        if not np.any(self.direction):
            return self.target.copy(), np.zeros(2)
        tau = t - self.start_time
        if tau >= self.times[-1]:
            return self.target.copy(), np.zeros(2)
        r = float(np.interp(tau, self.times, self.left))
        speed = float(np.interp(tau, self.times, self.speeds))
        return self.target - self.direction * r, self.direction * speed

    def acceleration(self, t):
        """Acceleration of the planned path at time t (for feedforward)."""
        if self.start_time is None or not np.any(self.direction):
            return np.zeros(2)
        tau = t - self.start_time
        if tau <= 0 or tau >= self.times[-1] - 0.02:
            return np.zeros(2)
        i = int(np.searchsorted(self.times, tau))
        i = min(max(i, 1), len(self.times) - 1)
        return self.direction * float((self.speeds[i] - self.speeds[i - 1]) / (self.times[i] - self.times[i - 1]))


METHODS = {
    "step": StepPlanner,
    "trajectory": TrajectoryPlanner,
    "update": ReplanningPlanner,
    "center": CenterPlanner,
}


def relevant_settings(method, shape="none"):
    """Names of the planner settings that change anything for this method and shape (for the GUIs to show only those).

    Names: duration, replan_interval, feedforward, cruise_speed, slow_distance, replan_error, size, lap_time.
    """
    names = set()
    if method in ("trajectory", "update"):
        names.add("duration")
    if method == "update":
        names.add("replan_interval")
    if method != "step":
        names.add("feedforward")          # the step method plans no acceleration to feed forward
    if method == "center":
        names |= {"cruise_speed", "slow_distance", "replan_error"}
    if shape != "none":
        names.add("size")
    if shape in CONTINUOUS:
        names.add("lap_time")
    return names


def make_planner(method, **settings):
    """Planner for "step", "trajectory", "update" or "center" (slows down on the way to the target).

    Settings (all optional): duration, replan_interval, accel_limit, edge_limit, and for "update" also
    final_approach, restart_distance and trigger; for "center" cruise_speed (cm/s), slow_distance (cm) and replan_error (cm, 0 = never replan).

    shape ("none", "line", "triangle", "square", "circle", "infinity"), size (cm) and
    lap_time (s, circle and infinity) make the ball go round a shape centred on the target instead of
    stopping at the target. "none" is the original behaviour.
    """
    shape = settings.get("shape", "none")
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    limits = {"accel_limit": settings.get("accel_limit", DEFAULT_ACCEL_LIMIT),
              "edge_limit": settings.get("edge_limit", DEFAULT_EDGE_LIMIT)}
    if method == "step":
        planner = StepPlanner()
        planner.duration = settings.get("duration", 3.0)    # only used as the time limit of a leg of a shape
    elif method == "trajectory":
        planner = TrajectoryPlanner(settings.get("duration", 3.0), **limits)
    elif method == "center":
        planner = CenterPlanner(settings.get("cruise_speed", 8.0), settings.get("slow_distance", 4.0),
                                limits["accel_limit"], settings.get("replan_error", 0.0),
                                settings.get("replan_gap", 1.0))
    elif method == "update":
        planner = ReplanningPlanner(settings.get("duration", 3.0), settings.get("replan_interval", 0.5), **limits,
                                    final_approach=settings.get("final_approach", 0.5),
                                    restart_distance=settings.get("restart_distance", 3.0),
                                    trigger=settings.get("trigger", 0.0))
    else:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    if shape == "none":
        return planner
    return ShapePlanner(planner, shape, settings.get("size", 4.0), settings.get("lap_time", 10.0),
                        accel_limit=limits["accel_limit"])
