"""Type alpha and beta in the terminal and the platform tilts to match.

    python platform_control.py                 # real hardware on the port set in motor_control.py (COM5)
    python platform_control.py --dry-run       # no hardware, just shows what would be sent

At the prompt enter two numbers, "alpha beta" in degrees, 80 to 100 (90 90 is level, e.g. "95 87"), or "q" to quit.
Each entry: alpha, beta -> motor angles theta_11, theta_21, theta_31 -> servo commands.
"""
import argparse

import numpy as np

import motor_control
import stewart_kinematics as sk

ANGLE_RANGE = (80.0, 100.0)   # deg, allowed alpha and beta (level = 90, 90)

# TODO: calibrate once the servos are mounted. Servo command = SERVO_ZERO + SERVO_DIRECTION * theta.
# SERVO_ZERO is the servo angle at which link 1 is horizontal (platform level);
# 90 is a placeholder guess (middle of the servo's travel).
SERVO_ZERO = [86.1, 76, 85.9]
SERVO_DIRECTION = [-1, -1, -1]   # use -1 for a motor mounted so that positive theta turns it backwards


def tilt_platform(link, alpha, beta):
    """Work out the three motor angles for (alpha, beta) and send them."""
    low, high = ANGLE_RANGE
    if not (low <= alpha <= high and low <= beta <= high):
        raise ValueError(f"alpha and beta must be between {low:g} and {high:g} deg")

    _points, theta_1, _theta_2 = sk.platform_to_motor_angles(alpha, beta)

    for i in range(3):
        servo_angle = SERVO_ZERO[i] + SERVO_DIRECTION[i] * theta_1[i]
        print(f"  motor {i + 1}: theta_{i + 1}1 = {theta_1[i]:6.2f} deg  ->  servo {servo_angle:6.2f} deg")
        link.move(i + 1, round(servo_angle, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=motor_control.COM_PORT)
    ap.add_argument("--dry-run", action="store_true", help="print commands instead of sending them")
    args = ap.parse_args()

    link = motor_control.MotorLink(args.port, args.dry_run)

    while True:
        text = input("alpha beta (deg) > ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        try:
            values = [float(v) for v in text.split()]
            if len(values) != 2:
                raise ValueError("enter two numbers: alpha beta")
            tilt_platform(link, *values)
        except ValueError as error:
            print("error:", error)


if __name__ == "__main__":
    main()
