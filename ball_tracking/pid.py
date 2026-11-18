"""PID controller for one axis of the plate: ball position error in cm -> tilt in degrees."""
import numpy as np

from trajectory import MAX_TILT_DEG


class AxisPID:
    """PID for one axis: error in cm, output is a tilt in degrees.

        error = reference_position - position
        P = Kp * error
        I = Ki * integral of error
        D = Kd * (reference_velocity - velocity)

    The D term works on the velocity error. For a fixed target the reference velocity is 0, so it is
    -Kd * velocity: no differencing of noisy positions, and no jump when the target moves. For a moving
    reference (a trajectory) it lets the controller follow the planned speed.
    """

    def __init__(self, limit_deg=MAX_TILT_DEG):
        self.limit = limit_deg
        self.integral = 0.0   # cm * s

    def reset(self):
        self.integral = 0.0

    def update(self, kp, ki, kd, reference, position, velocity, dt, reference_velocity=0.0):
        error = reference - position
        p = kp * error
        d = kd * (reference_velocity - velocity)

        # Anti-windup: stop adding to the integral while the output is pinned at the limit
        # and the error would push it further
        unsaturated = p + ki * self.integral + d
        if abs(unsaturated) < self.limit or unsaturated * error < 0:
            self.integral += error * dt
        if ki > 1e-6:
            cap = self.limit / ki   # the integral term alone can never exceed the limit
            self.integral = float(np.clip(self.integral, -cap, cap))

        output = p + ki * self.integral + d
        return float(np.clip(output, -self.limit, self.limit))
