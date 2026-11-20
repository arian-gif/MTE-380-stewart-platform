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
from pid import AxisPID, CENTRE_PD_GAINS, CONTROLLER_DEFAULT_GAINS
from trajectory import make_planner, tilt_for_acceleration

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


class StewartBallController:
    # Columns of the run log / CSV
    COLUMNS = ["time_s", "found", "x_cm", "y_cm", "vx_cm_s", "vy_cm_s", "setpoint_x_cm", "setpoint_y_cm",
               "reference_x_cm", "reference_y_cm", "reference_vx_cm_s", "reference_vy_cm_s",
               "tilt_x_deg", "tilt_y_deg", "alpha_deg", "beta_deg", "enabled", "kp", "ki", "kd",
               "method", "feedforward", "duration_s", "replan_interval_s"]
    METHOD_LABELS = {"step": "Step", "trajectory": "Trajectory", "update": "Trajectory + replanning"}

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
        self.Kp, self.Ki, self.Kd = CONTROLLER_DEFAULT_GAINS
        self.setpoint = [0.0, 0.0]               # cm from the plate centre
        self.pid = [AxisPID(), AxisPID()]        # x axis, y axis

        # How the ball is moved to the setpoint (see trajectory.py):
        #   "step"        the PID aims straight at the setpoint
        #   "trajectory"  a smooth path to the setpoint is planned when the move starts and the PID follows it
        #   "update"      the same, replanned from the measured ball state every replan_interval seconds
        self.method = "step"
        self.duration = 3.0                      # s, planned length of a move (trajectory methods)
        self.replan_interval = 0.5               # s, how often the path is replanned ("update")
        self.feedforward = 0.0                   # 0 = PID only, 1 = also tilt for the planned acceleration
        self.planner = make_planner(self.method, duration=self.duration, replan_interval=self.replan_interval)
        self.reference = None                    # (position, velocity) the PID is aiming at, None when not controlling

        # Controller state
        self.enabled = False                     # control is off until you tick the box
        self.running = False                     # False tells all threads to stop
        self.failed = False                      # set if sending to the platform fails; control then stays off
        self.tilt = None                         # last tilt sent (x, y) in degrees; None = nothing sent yet
        self.last_seen = time.time()             # last time the ball was found
        self.last_control_time = None
        self.centre_start = None
        self.centre_peak = None  # (cm, %), retained when control stops
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
        """Bring the plate level and clear the PID memory and the current move. Only sends if not already level."""
        for pid in self.pid:
            pid.reset()
        self.planner.reset()         # the next move starts from wherever the ball is
        self.reference = None
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
            self.centre_start = None
            for pid in self.pid:
                pid.reset()
            dt = 1 / 30

        # Where should the ball be now? The planner decides (for the step method it is just the setpoint)
        reference_position, reference_velocity = self.planner.reference(t, position_cm, velocity_cm_s, self.setpoint)
        self.reference = (reference_position, reference_velocity)

        tilt = [self.pid[axis].update(self.Kp, self.Ki, self.Kd, reference_position[axis],
                                      position_cm[axis], velocity_cm_s[axis], dt, reference_velocity[axis])
                for axis in range(2)]
        if self.feedforward:
            # Add the tilt that gives the ball the planned acceleration (zero for the step method)
            extra = self.feedforward * tilt_for_acceleration(self.planner.acceleration(t))
            tilt = [float(np.clip(tilt[axis] + extra[axis], -MAX_TILT_DEG, MAX_TILT_DEG)) for axis in range(2)]
        if self.method == "step" and self.setpoint == [0.0, 0.0]:
            start = self.centre_start
            if start is None:
                start = np.array(position_cm)
                self.centre_start = start
                self.centre_peak = None
            distance = np.linalg.norm(start)
            if distance > 1e-9:
                past = -np.dot(position_cm, start) / distance
                peak = max(self.centre_peak[0] if self.centre_peak else 0.0, past)
                self.centre_peak = (peak, 100 * peak / distance)
        else:
            self.centre_start = self.centre_peak = None
        self.send_tilt(*tilt)
        self.status = "CONTROLLING"
        return tuple(tilt)

    def record(self, t, found, position_cm, velocity_cm_s, tilt):
        """Add one row to the run log."""
        nan = float("nan")
        x, y = position_cm if found else (nan, nan)
        vx, vy = velocity_cm_s if found else (nan, nan)
        if self.reference is not None:
            (ref_x, ref_y), (ref_vx, ref_vy) = self.reference
        else:
            ref_x = ref_y = ref_vx = ref_vy = nan
        alpha = LEVEL_ALPHA - TILT_SIGN_X * tilt[0]
        beta = LEVEL_BETA - TILT_SIGN_Y * tilt[1]
        self.log.append([t - self.start_time, int(found), x, y, vx, vy, self.setpoint[0], self.setpoint[1],
                         ref_x, ref_y, ref_vx, ref_vy, tilt[0], tilt[1], alpha, beta, int(self.enabled),
                         self.Kp, self.Ki, self.Kd, self.method, self.feedforward, self.duration,
                         self.replan_interval])

    def set_method(self, method):
        """Switch between "step", "trajectory" and "update". The next move starts fresh."""
        self.method = method
        self.planner = make_planner(method, duration=self.duration, replan_interval=self.replan_interval)
        self.reference = None

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
        if self.method != "step" and self.reference is not None:
            reference_px = (int(center_x + self.reference[0][0] * pixels_per_cm),
                            int(center_y - self.reference[0][1] * pixels_per_cm))
            cv2.circle(overlay, reference_px, 7, (0, 255, 255), 2)   # yellow ring: where the plan says the ball should be
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
    def add_slider(self, name, label, value, low, high, parent=None):
        parent = parent or self.root
        ttk.Label(parent, text=label, font=("Arial", 12)).pack()
        var = tk.DoubleVar(value=value)
        ttk.Scale(parent, from_=low, to=high, variable=var, orient=tk.HORIZONTAL, length=500).pack(pady=3)
        value_label = ttk.Label(parent, text="", font=("Arial", 11))
        value_label.pack()
        self.sliders[name] = (var, value_label)

    def create_gui(self):
        """Build the Tkinter window: gain and setpoint sliders, enable box, buttons and status."""
        self.root = tk.Tk()
        self.root.title("Stewart Platform PID Controller")
        self.root.geometry("560x620")
        ttk.Label(self.root, text="PID Gains (both axes)", font=("Arial", 16, "bold")).pack(pady=8)

        # Two tabs: the PID gains and target as before, and the trajectory settings on a second tab.
        # This keeps the window short enough for a 720-pixel-high screen, so the Enable box and the
        # buttons below the tabs stay visible.
        tabs = ttk.Notebook(self.root)
        tabs.pack()
        pid_tab = ttk.Frame(tabs)
        method_tab = ttk.Frame(tabs)
        tabs.add(pid_tab, text="PID and target")
        tabs.add(method_tab, text="Method")

        overshoot_tab = ttk.Frame(tabs)
        tabs.add(overshoot_tab, text="Maximum overshoot")
        ttk.Label(overshoot_tab, text="Step response to plate centre", font=("Arial", 12, "bold")).pack(pady=12)
        ttk.Label(overshoot_tab, text="Choose gains while control is off. Start the ball away from centre,\n"
                  "then enable control. Record one move per session for comparison.").pack(padx=12)
        self.overshoot_buttons = []
        for label, gains in (("Baseline PID", CONTROLLER_DEFAULT_GAINS), ("Low-overshoot PD", CENTRE_PD_GAINS)):
            button = ttk.Button(overshoot_tab, text=f"{label}   Kp={gains[0]}   Ki={gains[1]}   Kd={gains[2]}",
                                command=lambda g=gains: self.prepare_centre_trial(g))
            button.pack(pady=6)
            self.overshoot_buttons.append(button)

        self.peak_label = ttk.Label(overshoot_tab, text="")
        self.peak_label.pack(pady=8)

        self.sliders = {}
        self.add_slider("Kp", "Kp (Proportional, deg per cm)", self.Kp, 0, 3, pid_tab)
        self.add_slider("Ki", "Ki (Integral)", self.Ki, 0, 1, pid_tab)
        self.add_slider("Kd", "Kd (Derivative, deg per cm/s)", self.Kd, 0, 2, pid_tab)
        limit = self.plate_radius_cm
        self.add_slider("Setpoint X", "Setpoint X (cm from centre)", 0.0, -limit, limit, pid_tab)
        self.add_slider("Setpoint Y", "Setpoint Y (cm from centre)", 0.0, -limit, limit, pid_tab)

        ttk.Label(method_tab, text="How the ball is moved to the target", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.method_var = tk.StringVar(value=self.method)
        for value, label in self.METHOD_LABELS.items():
            ttk.Radiobutton(method_tab, text=label, value=value, variable=self.method_var).pack(anchor="w", padx=40)
        self.add_slider("Duration", "Move duration (s, trajectory methods)", self.duration, 0.5, 8, method_tab)
        self.add_slider("Replan", "Replan interval (s, replanning method)", self.replan_interval, 0.1, 2, method_tab)
        self.add_slider("Feedforward", "Acceleration feedforward (0 = PID only, 1 = full)", self.feedforward, 0, 1, method_tab)

        self.enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.root, text="Enable control (platform tilts)", variable=self.enable_var).pack(pady=8)

        buttons = ttk.Frame(self.root)
        buttons.pack(pady=8)
        ttk.Button(buttons, text="Restart move", command=self.planner_reset).pack(side=tk.LEFT, padx=4)
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
        self.duration = self.sliders["Duration"][0].get()
        self.replan_interval = self.sliders["Replan"][0].get()
        self.feedforward = self.sliders["Feedforward"][0].get()
        if self.method_var.get() != self.method:
            self.set_method(self.method_var.get())
        for attribute in ("duration", "replan_interval"):   # used from the next move on
            if hasattr(self.planner, attribute):
                setattr(self.planner, attribute, getattr(self, attribute))
        self.enabled = self.enable_var.get()
        for button in self.overshoot_buttons:
            button.state(["disabled"] if self.enabled else ["!disabled"])
        for name, (var, label) in self.sliders.items():
            label.config(text=f"{name}: {var.get():.2f}")

        peak = self.centre_peak
        self.peak_label.config(text="Peak overshoot: --" if peak is None else
                               f"Peak overshoot: {peak[0]:.2f} cm ({peak[1]:.1f}%)")
        self.status_label.config(text=f"State: {self.status}")
        position = self.position_cm
        self.ball_label.config(text="Ball: not seen" if position is None
                               else f"Ball: x = {position[0]:.1f} cm, y = {position[1]:.1f} cm")
        tilt = self.tilt or (0.0, 0.0)
        self.tilt_label.config(text=f"Tilt: x = {tilt[0]:.1f} deg, y = {tilt[1]:.1f} deg")
        self.root.after(50, self.update_gui)

    def prepare_centre_trial(self, gains):
        if self.enable_var.get() or self.enabled:
            return
        for name, value in zip(("Kp", "Ki", "Kd", "Setpoint X", "Setpoint Y", "Feedforward"), (*gains, 0, 0, 0)):
            self.sliders[name][0].set(value)
        self.method_var.set("step")

    def level_button(self):
        """Turn control off and level the platform."""
        self.enable_var.set(False)
        self.enabled = False
        self.level()

    def planner_reset(self):
        """Start the move again from where the ball is now."""
        self.planner.reset()
        self.centre_start = None

    def reset_integral(self):
        for pid in self.pid:
            pid.reset()
        print("[RESET] Integral terms reset")

    # ------------------------------------------------------------------ results
    def column(self, name):
        """One column of the run log as an array."""
        index = self.COLUMNS.index(name)
        return np.array([row[index] for row in self.log], dtype=float)

    def draw_run(self, fig):
        """Draw this run on a matplotlib Figure: position, target and planned path for each axis, and the tilt commands.

        Returns False (and draws nothing) if nothing was logged.
        """
        if not self.log:
            return False
        t = self.column("time_s")
        axes = fig.subplots(3, 1, sharex=True)
        for ax, name in zip(axes[:2], ("x", "y")):
            ax.plot(t, self.column(f"{name}_cm"), label=f"ball {name}", linewidth=2)
            ax.plot(t, self.column(f"setpoint_{name}_cm"), "--", label=f"target {name}", linewidth=2)
            ax.plot(t, self.column(f"reference_{name}_cm"), ":", label=f"planned {name}", linewidth=2)
            ax.set_ylabel(f"{name} position (cm)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        axes[2].plot(t, self.column("tilt_x_deg"), label="tilt x")
        axes[2].plot(t, self.column("tilt_y_deg"), label="tilt y")
        axes[2].set_ylabel("Tilt command (deg)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)
        axes[0].set_title(f"{self.run_id}: {self.METHOD_LABELS[self.method]}, feedforward {self.feedforward:.1f}, "
                          f"Kp={self.Kp:.2f} Ki={self.Ki:.2f} Kd={self.Kd:.2f}")
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
                cells = []
                for name, value in zip(self.COLUMNS, row):
                    if name in ("found", "enabled"):
                        cells.append(int(value))                           # 0 or 1
                    elif isinstance(value, str):
                        cells.append(value)                                # the method name
                    else:
                        cells.append(f"{value:.4f}" if value == value else "")   # blank where there is no value
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
