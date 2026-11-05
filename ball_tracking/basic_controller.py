"""Stewart platform ball-balancing PID controller (replaces the 1D beam version).

    python basic_controller.py             # real hardware, port set in motor_control.py (COM5)
    python basic_controller.py --dry-run   # camera and GUI run, but nothing is sent to the motors

Each camera frame:
    camera -> ball position and velocity (BallTracker) -> one PID per axis -> tilt of the plate
    -> alpha, beta -> servo angles (platform_control) -> Arduino

Units: positions in cm, velocities in cm/s, tilts in degrees.
Run simple_cal.py first: it writes config.json (ball colour, plate centre and scale).
"""
import argparse
import csv
import json
import os
import queue
import sys
import time
import tkinter as tk
from datetime import datetime
from threading import Thread
from tkinter import ttk

import cv2
import numpy as np

# motor_control.py and platform_control.py live in the parent folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import motor_control
import platform_control
from ball_detection import (BallDetector, BallTracker, CAPTURE_SIZE, PROCESS_SIZE,
                            CONFIG_PATH, HERE, fit_square)

LEVEL_ALPHA, LEVEL_BETA = 90.0, 90.0   # platform angles for a level plate
MAX_TILT_DEG = 10.0                    # the platform accepts alpha and beta within 90 +/- 10

# A positive tilt slopes the plate down towards +x (or +y), so the ball rolls that way.
# TODO: check on the real platform. If the ball runs away along an axis, flip that axis' sign.
TILT_SIGN_X = 1
TILT_SIGN_Y = 1

LOST_TIMEOUT_S = 0.3   # level the platform if the ball has not been seen for this long
RUNS_DIR = os.path.join(HERE, "tuning_runs")   # everything a run saves goes under here
CSV_DIR = os.path.join(RUNS_DIR, "csv")        # the logged values of each run, one CSV per run
PLOTS_DIR = os.path.join(RUNS_DIR, "plots")    # the matplotlib plot of each run, one PNG per run


class AxisPID:
    """PID controller for one axis: error in cm, output is a tilt in degrees.

    The D term uses the ball's measured velocity. With a fixed setpoint, d(error)/dt = -velocity,
    so there is no need to difference noisy positions, and it does not jump when you move the setpoint.
    """

    def __init__(self, limit_deg=MAX_TILT_DEG):
        self.limit = limit_deg
        self.integral = 0.0   # cm * s

    def reset(self):
        self.integral = 0.0

    def update(self, kp, ki, kd, setpoint, position, velocity, dt):
        error = setpoint - position
        p = kp * error
        d = -kd * velocity

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


class StewartBallController:
    # Columns of the run log / CSV
    COLUMNS = ["time_s", "found", "x_cm", "y_cm", "vx_cm_s", "vy_cm_s", "setpoint_x_cm", "setpoint_y_cm",
               "tilt_x_deg", "tilt_y_deg", "alpha_deg", "beta_deg", "enabled", "kp", "ki", "kd"]

    def __init__(self, link, config_file=CONFIG_PATH):
        """Load the calibration and set defaults. link is a motor_control.MotorLink."""
        self.link = link
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.detector = BallDetector(config_file)
        if self.detector.plate_center_px is None or not self.detector.meters_per_pixel:
            raise RuntimeError("config.json has no plate calibration. Run simple_cal.py first.")
        self.tracker = BallTracker(self.detector)
        self.camera_index = self.config["camera"]["index"]
        self.plate_radius_cm = self.config.get("plate_radius_m", 0.135) * 100

        # PID gains (set by the sliders). Starting point from the ball's physics: a rolling ball
        # accelerates about 12 cm/s^2 per degree of tilt; tune on the real platform from here.
        self.Kp = 0.75
        self.Ki = 0.0
        self.Kd = 0.35
        self.setpoint = [0.0, 0.0]               # cm from the plate centre
        self.pid = [AxisPID(), AxisPID()]        # x axis, y axis

        # Controller state
        self.enabled = False                     # control is off until you tick the box
        self.running = False                     # False tells all threads to stop
        self.failed = False                      # set if sending to the platform fails; control then stays off
        self.tilt = None                         # last tilt sent (x, y) in degrees; None = nothing sent yet
        self.last_seen = time.time()             # last time the ball was found
        self.last_control_time = None
        self.sample_queue = queue.Queue(maxsize=1)   # newest camera sample only

        # Shown in the GUI
        self.status = "DISABLED"
        self.position_cm = None

        # Run log. The CSV and the plot of a run share this name, so they pair up.
        self.log = []
        self.start_time = None
        self.run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"

    # ------------------------------------------------------------------ platform output
    def send_tilt(self, tilt_x, tilt_y):
        """Tilt the plate: convert the tilts (deg) to alpha, beta and send the servo commands."""
        alpha = LEVEL_ALPHA - TILT_SIGN_X * tilt_x
        beta = LEVEL_BETA - TILT_SIGN_Y * tilt_y
        platform_control.tilt_platform_fast(self.link, alpha, beta)
        self.tilt = (tilt_x, tilt_y)

    def level(self):
        """Bring the plate level and clear the PID memory. Only sends if it is not already level."""
        for pid in self.pid:
            pid.reset()
        self.last_control_time = None
        if self.tilt != (0.0, 0.0):
            self.send_tilt(0.0, 0.0)
        return (0.0, 0.0)

    # ------------------------------------------------------------------ one control cycle
    def control_step(self, t, found, position_cm, velocity_cm_s):
        """One control cycle. Returns the (tilt_x, tilt_y) in degrees now applied to the plate."""
        if self.failed:
            self.status = "ERROR - restart the program"
            return (0.0, 0.0)

        if found:
            self.last_seen = t

        if not self.enabled:
            self.status = "DISABLED"
            return self.level()

        if not found:
            if t - self.last_seen > LOST_TIMEOUT_S:
                self.status = "BALL LOST - holding level"
                return self.level()
            return self.tilt   # short dropout: keep the last command

        # Time since the previous cycle. After a pause, start fresh so the integral cannot jump.
        dt = None if self.last_control_time is None else t - self.last_control_time
        self.last_control_time = t
        if dt is None or not (0 < dt <= LOST_TIMEOUT_S):
            for pid in self.pid:
                pid.reset()
            dt = 1 / 30

        tilt = [self.pid[axis].update(self.Kp, self.Ki, self.Kd, self.setpoint[axis],
                                      position_cm[axis], velocity_cm_s[axis], dt)
                for axis in range(2)]
        self.send_tilt(*tilt)
        self.status = "CONTROLLING"
        return tuple(tilt)

    def record(self, t, found, position_cm, velocity_cm_s, tilt):
        """Add one row to the run log."""
        nan = float("nan")
        x, y = position_cm if found else (nan, nan)
        vx, vy = velocity_cm_s if found else (nan, nan)
        alpha = LEVEL_ALPHA - TILT_SIGN_X * tilt[0]
        beta = LEVEL_BETA - TILT_SIGN_Y * tilt[1]
        self.log.append([t - self.start_time, int(found), x, y, vx, vy, self.setpoint[0], self.setpoint[1],
                         tilt[0], tilt[1], alpha, beta, int(self.enabled), self.Kp, self.Ki, self.Kd])

    # ------------------------------------------------------------------ threads
    def camera_thread(self):
        """Capture frames, track the ball, show the video and pass the newest sample on."""
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            t = time.time()
            frame = cv2.resize(frame, PROCESS_SIZE)
            found, position_m, velocity_mps = self.tracker.update(frame, t)

            # Keep only the newest sample
            if self.sample_queue.full():
                try:
                    self.sample_queue.get_nowait()
                except queue.Empty:
                    pass
            self.sample_queue.put_nowait((t, found, position_m, velocity_mps))

            cv2.imshow("Ball Tracking", fit_square(self.draw_overlay(frame))[0])
            if cv2.waitKey(1) & 0xFF == 27:  # ESC exits
                self.running = False
        cap.release()
        cv2.destroyAllWindows()

    def draw_overlay(self, frame):
        """Tracker overlay plus a magenta cross at the setpoint."""
        overlay = self.tracker.draw(frame)
        center_x, center_y = self.detector.plate_center_px
        pixels_per_cm = 0.01 / self.detector.meters_per_pixel
        setpoint_px = (int(center_x + self.setpoint[0] * pixels_per_cm),
                       int(center_y - self.setpoint[1] * pixels_per_cm))   # image y points down
        cv2.drawMarker(overlay, setpoint_px, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        return overlay

    def control_thread(self):
        """Wait for camera samples and run one control cycle for each."""
        while self.running:
            try:
                t, found, position_m, velocity_mps = self.sample_queue.get(timeout=0.1)
            except queue.Empty:
                t, found, position_m, velocity_mps = time.time(), False, None, None   # camera stalled

            position_cm = (position_m[0] * 100, position_m[1] * 100) if found else None
            velocity_cm_s = (velocity_mps[0] * 100, velocity_mps[1] * 100) if found else None
            self.position_cm = position_cm

            try:
                tilt = self.control_step(t, found, position_cm, velocity_cm_s)
            except Exception as error:   # e.g. the serial link failed: stop controlling for good
                print(f"[CONTROL] Error: {error}. Control stopped - fix the cause and restart the program.")
                self.failed = True
                self.enabled = False
                tilt = (0.0, 0.0)
            self.record(t, found, position_cm, velocity_cm_s, tilt)

        # Leave the platform level (not possible if sending already failed)
        if not self.failed:
            try:
                self.level()
            except Exception as error:
                print(f"[CONTROL] Could not level the platform on exit: {error}")

    # ------------------------------------------------------------------ GUI
    def add_slider(self, name, label, value, low, high):
        ttk.Label(self.root, text=label, font=("Arial", 12)).pack()
        var = tk.DoubleVar(value=value)
        ttk.Scale(self.root, from_=low, to=high, variable=var, orient=tk.HORIZONTAL, length=500).pack(pady=3)
        value_label = ttk.Label(self.root, text="", font=("Arial", 11))
        value_label.pack()
        self.sliders[name] = (var, value_label)

    def create_gui(self):
        """Build the Tkinter window: gain and setpoint sliders, enable box, buttons and status."""
        self.root = tk.Tk()
        self.root.title("Stewart Platform PID Controller")
        self.root.geometry("560x720")
        ttk.Label(self.root, text="PID Gains (both axes)", font=("Arial", 16, "bold")).pack(pady=8)

        self.sliders = {}
        self.add_slider("Kp", "Kp (Proportional, deg per cm)", self.Kp, 0, 3)
        self.add_slider("Ki", "Ki (Integral)", self.Ki, 0, 1)
        self.add_slider("Kd", "Kd (Derivative, deg per cm/s)", self.Kd, 0, 2)
        limit = self.plate_radius_cm
        self.add_slider("Setpoint X", "Setpoint X (cm from centre)", 0.0, -limit, limit)
        self.add_slider("Setpoint Y", "Setpoint Y (cm from centre)", 0.0, -limit, limit)

        self.enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.root, text="Enable control (platform tilts)", variable=self.enable_var).pack(pady=8)

        buttons = ttk.Frame(self.root)
        buttons.pack(pady=8)
        ttk.Button(buttons, text="Level platform", command=self.level_button).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Reset integral", command=self.reset_integral).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Plot results", command=self.plot_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=4)

        self.status_label = ttk.Label(self.root, text="", font=("Arial", 13, "bold"))
        self.status_label.pack(pady=4)
        self.ball_label = ttk.Label(self.root, text="", font=("Arial", 11))
        self.ball_label.pack()
        self.tilt_label = ttk.Label(self.root, text="", font=("Arial", 11))
        self.tilt_label.pack()
        self.update_gui()

    def update_gui(self):
        """Copy the slider values into the controller and refresh the labels (every 50 ms)."""
        if not self.running:
            self.root.quit()   # camera window closed with ESC
            return
        self.Kp = self.sliders["Kp"][0].get()
        self.Ki = self.sliders["Ki"][0].get()
        self.Kd = self.sliders["Kd"][0].get()
        self.setpoint = [self.sliders["Setpoint X"][0].get(), self.sliders["Setpoint Y"][0].get()]
        self.enabled = self.enable_var.get()
        for name, (var, label) in self.sliders.items():
            label.config(text=f"{name}: {var.get():.2f}")

        self.status_label.config(text=f"State: {self.status}")
        position = self.position_cm
        self.ball_label.config(text="Ball: not seen" if position is None
                               else f"Ball: x = {position[0]:.1f} cm, y = {position[1]:.1f} cm")
        tilt = self.tilt or (0.0, 0.0)
        self.tilt_label.config(text=f"Tilt: x = {tilt[0]:.1f} deg, y = {tilt[1]:.1f} deg")
        self.root.after(50, self.update_gui)

    def level_button(self):
        """Turn control off and level the platform."""
        self.enable_var.set(False)
        self.enabled = False
        self.level()

    def reset_integral(self):
        for pid in self.pid:
            pid.reset()
        print("[RESET] Integral terms reset")

    # ------------------------------------------------------------------ results
    def draw_run(self, fig):
        """Draw this run on a matplotlib Figure: position vs setpoint for each axis, and the tilt commands.

        Returns False (and draws nothing) if nothing was logged.
        """
        if not self.log:
            return False
        data = np.array(self.log, dtype=float)
        t = data[:, 0]
        axes = fig.subplots(3, 1, sharex=True)
        for ax, column, setpoint_column, name in ((axes[0], 2, 6, "x"), (axes[1], 3, 7, "y")):
            ax.plot(t, data[:, column], label=f"ball {name}", linewidth=2)
            ax.plot(t, data[:, setpoint_column], "--", label=f"setpoint {name}", linewidth=2)
            ax.set_ylabel(f"{name} position (cm)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        axes[2].plot(t, data[:, 8], label="tilt x")
        axes[2].plot(t, data[:, 9], label="tilt y")
        axes[2].set_ylabel("Tilt command (deg)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)
        axes[0].set_title(f"{self.run_id}:  Kp={self.Kp:.2f}  Ki={self.Ki:.2f}  Kd={self.Kd:.2f}")
        fig.tight_layout()
        return True

    def plot_path(self):
        """Where this run's plot is saved: tuning_runs/plots/<run name>.png"""
        os.makedirs(PLOTS_DIR, exist_ok=True)
        return os.path.join(PLOTS_DIR, f"{self.run_id}.png")

    def plot_results(self):
        """Turn control off, level the platform, save the plot of the run so far and show it."""
        self.level_button()
        import matplotlib.pyplot as plt   # imported here so the controller starts without it

        fig = plt.figure(figsize=(10, 9))
        if not self.draw_run(fig):
            print("[PLOT] No data to plot")
            plt.close(fig)
            return
        path = self.plot_path()
        fig.savefig(path, dpi=120)
        print(f"[PLOT] Saved {path}")
        plt.show()

    def save_run(self):
        """Save this run: the logged values to tuning_runs/csv/ and the plot to tuning_runs/plots/.

        Returns (csv_path, png_path); both are None if nothing was logged.
        """
        if not self.log:
            return None, None

        os.makedirs(CSV_DIR, exist_ok=True)
        csv_path = os.path.join(CSV_DIR, f"{self.run_id}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for row in self.log:
                # Blank cells (not "nan") where the ball was not found; found/enabled as 0 or 1
                cells = [f"{v:.4f}" if v == v else "" for v in row]
                cells[1] = int(row[1])
                cells[12] = int(row[12])
                writer.writerow(cells)

        png_path = None
        try:
            from matplotlib.figure import Figure   # not pyplot, so no window opens
            fig = Figure(figsize=(10, 9))
            self.draw_run(fig)
            png_path = self.plot_path()
            fig.savefig(png_path, dpi=120)
        except ImportError:
            print("[PLOT] matplotlib is not installed - saved the CSV only")
        return csv_path, png_path

    def stop(self):
        """Stop everything and close the window."""
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        """Start the camera and control threads, run the GUI, then level the platform and save the run."""
        print("[INFO] Starting Stewart platform PID controller")
        print("Tick 'Enable control' to start balancing. Close the window, press Stop or ESC in the video to exit.")
        self.running = True
        self.start_time = time.time()
        self.run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"
        platform_control.set_safe_pose(self.link, LEVEL_ALPHA, LEVEL_BETA)   # Arduino goes level if we stop sending
        self.level()   # always start from a level platform

        camera = Thread(target=self.camera_thread, daemon=True)
        controller = Thread(target=self.control_thread, daemon=True)
        camera.start()
        controller.start()

        self.create_gui()
        self.root.mainloop()

        self.running = False
        camera.join(timeout=2)
        controller.join(timeout=2)   # the control thread levels the platform before it ends
        csv_path, png_path = self.save_run()
        if csv_path:
            print(f"[INFO] Controller stopped. Saved {csv_path}" + (f" and {png_path}" if png_path else ""))
        else:
            print("[INFO] Controller stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=motor_control.COM_PORT)
    ap.add_argument("--dry-run", action="store_true", help="run the camera and GUI without sending to the motors")
    args = ap.parse_args()

    if not os.path.exists(CONFIG_PATH):
        sys.exit(f"{CONFIG_PATH} not found. Run simple_cal.py first.")
    link = motor_control.MotorLink(args.port, args.dry_run)
    try:
        controller = StewartBallController(link)
    except RuntimeError as error:
        sys.exit(str(error))
    controller.run()


if __name__ == "__main__":
    main()
