"""Stewart platform animations driven by the platform angles alpha and beta (see stewart_kinematics.py).

Outputs (written next to this script):
    static_configs.png      four fixed tilts
    animation_frames.png    12 frames of the tilt going round a circle
    stewart_animation.gif   smooth version of the same motion

Run:  python stewart_animation.py            (add --show for an interactive window)
"""
import argparse
import os
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import stewart_kinematics as sk

OUT = os.path.dirname(os.path.abspath(__file__))
MAX_TILT = 10.0                                   # deg away from level (alpha, beta = 90 +/- 10)
COLORS = ["tab:red", "tab:green", "tab:orange"]   # actuators 1, 2, 3


def draw(ax, alpha, beta):
    ax.clear()

    # base circle through the motor axes
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(sk.L_BASE[0] * np.cos(t), sk.L_BASE[0] * np.sin(t), 0 * t, "r-", lw=1, label="Base Circle")

    # platform triangle
    points, theta_1, theta_2 = sk.platform_to_motor_angles(alpha, beta)
    triangle = np.array(points + [points[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], "b-", lw=2, label="Moving Platform")

    # the three actuators: motor axis -> elbow -> platform support point
    for i in range(3):
        arm = np.array([sk.base_joint(i), sk.elbow(i, theta_1[i]), points[i]])
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], color=COLORS[i], lw=2, label=f"Actuator {i + 1}")

    ax.set(xlim=(-150, 150), ylim=(-150, 150), zlim=(0, 200),
           xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    ax.set_title(f"α={alpha:.0f}°, β={beta:.0f}°   motors: "
                 f"{theta_1[0]:.0f}°, {theta_1[1]:.0f}°, {theta_1[2]:.0f}°", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")


def tilt_on_circle(angle_deg):
    """alpha, beta for a tilt of MAX_TILT that goes once around as angle_deg goes 0..360."""
    a = np.radians(angle_deg)
    return 90 + MAX_TILT * np.cos(a), 90 + MAX_TILT * np.sin(a)


def static_configs():
    fig = plt.figure(figsize=(10, 9))
    tilts = [(90, 90), (100, 90), (90, 100), (80, 80)]
    for k, (alpha, beta) in enumerate(tilts):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        draw(ax, alpha, beta)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "static_configs.png"), dpi=110)
    plt.close(fig)


def frame_grid():
    fig = plt.figure(figsize=(14, 8))
    for k in range(12):
        ax = fig.add_subplot(3, 4, k + 1, projection="3d")
        draw(ax, *tilt_on_circle(30 * k))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "animation_frames.png"), dpi=100)
    plt.close(fig)


def make_gif(show=False):
    frames = 72
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    anim = FuncAnimation(fig, lambda f: draw(ax, *tilt_on_circle(360 * f / frames)), frames=frames, interval=50)
    anim.save(os.path.join(OUT, "stewart_animation.gif"), writer=PillowWriter(fps=20))
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="open an interactive window after saving")
    args = ap.parse_args()
    static_configs()
    frame_grid()
    make_gif(args.show)
    print("wrote static_configs.png, animation_frames.png, stewart_animation.gif")
