"""Platform angle control, following the handwritten "Platform Angle Control" notes.

Steps (same order as the notes):
    1. Inputs:        platform centre x0, angles alpha and beta (80..100 deg, level = 90)
    2. Normal vector: n
    3. Vectors on the platform: d1, d2, d3 (one towards each actuator)
    4. Support points: x_i = x0 + r * d_i
    5. Inverse kinematics (functions.py) -> motor angles theta_11, theta_21, theta_31

All lengths in mm, all angles in degrees.
"""
import numpy as np

import functions as fn

# ---------------------------------------------------------------- constants (from the notes)
R_PLATFORM = 135.0                 # r: centre of platform to each support point
L_BASE = [50.0, 50.0, 50.0]        # l_1, l_2, l_3:     centre of base to each motor axis
L_LINK_1 = [80.0, 80.0, 80.0]      # l_11, l_21, l_31:  motor-driven links
L_LINK_2 = [94.0, 94.0, 94.0]      # l_12, l_22, l_32:  passive links to the platform
X0 = np.array([0.0, 0.0, 94.0])    # platform centre, kept fixed for now

# Direction of each actuator in the x-y plane: actuator 1 on +y, 2 at -30 deg, 3 at 210 deg.
ACTUATOR_ANGLES = [90.0, -30.0, 210.0]

# The inverse kinematics from functions.py, one per actuator.
INVERSE = [fn.calculate_vectors_and_angles_1,
           fn.calculate_vectors_and_angles_2,
           fn.calculate_vectors_and_angles_3]
FORWARD = [fn.calculate_position_1,
           fn.calculate_position_2,
           fn.calculate_position_3]


def normal_vector(alpha, beta):
    """Unit normal of the platform, exactly as in the notes.

    alpha: angle between the platform normal and the x axis (deg)
    beta:  angle between the platform normal and the y axis (deg)
    Level platform: alpha = beta = 90 deg -> n = [0, 0, 1]. Allowed range is 80..100 deg.
    """
    nx = np.cos(np.radians(alpha))
    ny = np.cos(np.radians(beta))
    nz = np.sqrt(1 - nx ** 2 - ny ** 2)
    return np.array([nx, ny, nz])


def platform_directions(n):
    """Unit vectors d1, d2, d3 lying in the platform, pointing at the three support points."""
    i_hat = np.array([1.0, 0.0, 0.0])
    j_hat = np.array([0.0, 1.0, 0.0])
    t = np.tan(np.radians(60))

    d = [np.cross(n, i_hat),
         np.cross(i_hat + j_hat * t, n),
         np.cross(i_hat - j_hat * t, n)]

    return [v / np.linalg.norm(v) for v in d]   # make them unit vectors


def support_points(x0, directions, r=R_PLATFORM):
    """Support points x1, x2, x3 on the platform: x_i = x0 + r * d_i."""
    return [x0 + r * d for d in directions]


def solve_motor_angles(points):
    """Run the inverse kinematics for each actuator.

    Returns two lists: motor angles theta_i1 and passive-joint angles theta_i2.
    """
    x1, x2, x3 = points
    theta_1 = []   # theta_11, theta_21, theta_31 (the motors)
    theta_2 = []   # theta_12, theta_22, theta_32
    for i in range(3):
        result = INVERSE[i](0, L_BASE[i], L_LINK_1[i], L_LINK_2[i], x1, x2, x3)
        if any(np.isnan(v) for v in result.values()):
            raise ValueError(f"actuator {i + 1}: support point {np.round(points[i], 1)} is out of reach")
        theta_1.append(result[f"theta_{i + 1}1"])
        theta_2.append(result[f"theta_{i + 1}2"])
    return theta_1, theta_2


def platform_to_motor_angles(alpha, beta, x0=X0):
    """Whole chain of the notes. Returns (support points, theta_i1 list, theta_i2 list)."""
    n = normal_vector(alpha, beta)
    directions = platform_directions(n)
    points = support_points(x0, directions)
    theta_1, theta_2 = solve_motor_angles(points)
    return points, theta_1, theta_2


# ---------------------------------------------------------------- helpers for checks/drawing
def forward_points(theta_1, theta_2):
    """Support points recomputed from the angles with the forward kinematics (a check)."""
    points = []
    for i in range(3):
        result = FORWARD[i](0, L_BASE[i], L_LINK_1[i], L_LINK_2[i], theta_1[i], theta_2[i])
        points.append(result[f"x{i + 1}"])
    return points


def _radial_to_xyz(i, radial, z):
    a = np.radians(ACTUATOR_ANGLES[i])
    return np.array([radial * np.cos(a), radial * np.sin(a), z])


def base_joint(i):
    """Where motor i sits on the base."""
    return _radial_to_xyz(i, L_BASE[i], 0.0)


def elbow(i, theta_1):
    """Joint between the motor link and the passive link of actuator i."""
    t = np.radians(theta_1)
    return _radial_to_xyz(i, L_BASE[i] + L_LINK_1[i] * np.cos(t), L_LINK_1[i] * np.sin(t))


if __name__ == "__main__":
    for alpha, beta in [(90, 90), (100, 90), (90, 100), (80, 100), (80, 80)]:
        points, theta_1, theta_2 = platform_to_motor_angles(alpha, beta)
        check = forward_points(theta_1, theta_2)
        error = max(np.linalg.norm(check[i] - points[i]) for i in range(3))
        print(f"alpha={alpha:4}, beta={beta:4}:  theta_11={theta_1[0]:6.1f}  "
              f"theta_21={theta_1[1]:6.1f}  theta_31={theta_1[2]:6.1f}   (FK check error {error:.1e} mm)")
