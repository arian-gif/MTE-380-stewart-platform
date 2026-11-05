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

# Servo command = SERVO_ZERO + SERVO_DIRECTION * theta, where theta is the motor link's angle from the kinematics.
# SERVO_ZERO[i] is the servo angle at which motor i's link 1 is exactly horizontal (theta = 0), measured on the
# real platform. A LEVEL plate is not theta = 0: with the current dimensions it needs theta = 8.03 deg, so level
# gives servo angles 78.1, 68.0, 77.9.
SERVO_ZERO = [86.1, 76, 85.9]
SERVO_DIRECTION = [-1, -1, -1]   # use -1 for a motor mounted so that positive theta turns it backwards


def servo_angles(alpha, beta):
    """alpha, beta (deg) -> ([servo angle for motor 1, 2, 3], [theta_11, theta_21, theta_31]).

    Raises ValueError if alpha or beta is outside ANGLE_RANGE or the pose is out of reach.
    """
    low, high = ANGLE_RANGE
    if not (low <= alpha <= high and low <= beta <= high):
        raise ValueError(f"alpha and beta must be between {low:g} and {high:g} deg")

    _points, theta_1, _theta_2 = sk.platform_to_motor_angles(alpha, beta)

    servos = [round(SERVO_ZERO[i] + SERVO_DIRECTION[i] * theta_1[i], 1) for i in range(3)]
    return servos, theta_1


def tilt_platform(link, alpha, beta):
    """Work out the three motor angles for (alpha, beta) and send them."""
    servos, theta_1 = servo_angles(alpha, beta)

    for i in range(3):
        print(f"  motor {i + 1}: theta_{i + 1}1 = {theta_1[i]:6.2f} deg  ->  servo {servos[i]:6.2f} deg")
        link.move(i + 1, servos[i])


def set_safe_pose(link, alpha=90.0, beta=90.0):
    """Tell the Arduino to go to this pose (default: level) if the program stops sending commands."""
    servos, _theta_1 = servo_angles(alpha, beta)
    link.set_safe_angles({1: servos[0], 2: servos[1], 3: servos[2]})


def tilt_platform_fast(link, alpha, beta):
    """Same as tilt_platform but quiet and without waiting for replies, for control loops."""
    servos, _theta_1 = servo_angles(alpha, beta)
    link.move_many({1: servos[0], 2: servos[1], 3: servos[2]})


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
