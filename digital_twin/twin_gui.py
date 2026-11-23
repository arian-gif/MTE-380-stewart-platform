"""Digital twin with sliders: change anything and see what the ball does. No hardware needed.

    python digital_twin/twin_gui.py

Left: the settings (how the ball is moved, the controller gains, where the ball starts, and the assumptions
about the platform). Right: the ball's path, the two tracking errors and the tilt command.
  * "Auto-run" re-simulates whenever you let go of a slider (a run takes about 0.3 s).
  * "Keep previous runs" overlays the new run on the old ones, so you can see the effect of one change.
  * "Clear runs" forgets them.
The numbers under the plot are the RMS pointwise and cross-track errors of the newest run.
"""
import tkinter as tk
from dataclasses import replace
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from twin import METHODS, SHAPES, Settings, draw, relevant_settings, run, summary

# (Settings field, label, low, high, is_integer)
SLIDERS = [
    ("Move", [("duration", "Move time / time per side (s)", 0.5, 8, False),
              ("size", "Shape radius (cm)", 1, 10, False),
              ("lap_time", "Lap time, circle / infinity (s)", 3, 30, False),
              ("replan_interval", "Replan interval (s)", 0.1, 2, False),
              ("cruise_speed", "Cruise speed, to centre (cm/s)", 2, 20, False),
              ("slow_distance", "Slow-down distance, to centre (cm)", 1, 10, False),
              ("replan_error", "Replan when this far off plan, to centre (cm, 0 = never)", 0, 6, False)]),
    ("Controller", [("kp", "Kp (deg per cm)", 0, 3, False),
                    ("ki", "Ki", 0, 1, False),
                    ("kd", "Kd (deg per cm/s)", 0, 2, False),
                    ("feedforward", "Acceleration feedforward", 0, 1, False)]),
    ("Ball start", [("start_x", "Start x (cm)", -12, 12, False),
                    ("start_y", "Start y (cm)", -12, 12, False),
                    ("velocity_x", "Start velocity x (cm/s)", -30, 30, False),
                    ("velocity_y", "Start velocity y (cm/s)", -30, 30, False),
                    ("seconds", "Run length (s)", 5, 60, False)]),
    ("Platform (assumed)", [("friction", "Rolling friction (1/s)", 0, 2, False),
                            ("servo_tau", "Servo lag (s)", 0.01, 0.3, False),
                            ("camera_delay_frames", "Camera delay (frames of 33 ms)", 0, 10, True),
                            ("noise_cm", "Camera noise (cm)", 0, 0.5, False),
                            ("bias_x", "Plate bias x (deg)", -1, 1, False),
                            ("bias_y", "Plate bias y (deg)", -1, 1, False)]),
]


class TwinGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stewart platform digital twin")
        self.root.geometry("1250x760")
        defaults = Settings()
        self.values = {"start_x": defaults.start[0], "start_y": defaults.start[1],
                       "velocity_x": defaults.velocity[0], "velocity_y": defaults.velocity[1],
                       "bias_x": defaults.tilt_bias[0], "bias_y": defaults.tilt_bias[1]}
        self.vars, self.labels, self.integer = {}, {}, {}
        self.frames = {}     # tab -> [(setting name, frame)] in order, so unused sliders can be hidden
        self.runs = []
        self.counter = 0

        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # How the ball is moved: method and shape
        top = ttk.Frame(left)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Method").grid(row=0, column=0, sticky="w")
        self.method = tk.StringVar(value=defaults.method)
        box = ttk.Combobox(top, textvariable=self.method, values=METHODS, state="readonly", width=12)
        box.grid(row=0, column=1, padx=6)
        ttk.Label(top, text="Shape").grid(row=0, column=2, sticky="w")
        self.shape = tk.StringVar(value=defaults.shape)
        shape_box = ttk.Combobox(top, textvariable=self.shape, values=SHAPES, state="readonly", width=10)
        shape_box.grid(row=0, column=3, padx=6)
        for widget in (box, shape_box):
            widget.bind("<<ComboboxSelected>>", lambda event: self.maybe_run())

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.X, pady=6)
        for tab_name, sliders in SLIDERS:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=tab_name)
            for name, label, low, high, integer in sliders:
                self.add_slider(tab, name, label, low, high, integer, defaults)

        controls = ttk.Frame(left)
        controls.pack(fill=tk.X)
        self.auto = tk.BooleanVar(value=True)
        self.keep = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Auto-run", variable=self.auto).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(controls, text="Keep previous runs", variable=self.keep).grid(row=0, column=1, sticky="w")
        ttk.Button(controls, text="Run", command=self.run_once).grid(row=1, column=0, pady=6, sticky="w")
        ttk.Button(controls, text="Clear runs", command=self.clear).grid(row=1, column=1, pady=6, sticky="w")
        ttk.Button(controls, text="Reset all", command=self.reset).grid(row=1, column=2, pady=6, sticky="w")
        self.text = tk.Label(left, text="", justify="left", font=("Consolas", 10), anchor="w")
        self.text.pack(fill=tk.X, pady=4)

        self.figure = Figure(figsize=(8, 6.5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.run_once()

    # ---- widgets
    def add_slider(self, parent, name, label, low, high, integer, defaults):
        if name in self.values:
            value = self.values[name]
        else:
            value = getattr(defaults, name)
        var = tk.DoubleVar(value=value)
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)
        self.frames.setdefault(parent, []).append((name, frame))
        ttk.Label(frame, text=label).pack(anchor="w", padx=6)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=6)
        scale = ttk.Scale(row, from_=low, to=high, variable=var, orient=tk.HORIZONTAL, length=300,
                          command=lambda _v, n=name: self.update_label(n))
        scale.pack(side=tk.LEFT)
        scale.bind("<ButtonRelease-1>", lambda event: self.maybe_run())
        value_label = ttk.Label(row, width=7)
        value_label.pack(side=tk.LEFT, padx=6)
        self.vars[name], self.labels[name], self.integer[name] = var, value_label, integer
        self.update_label(name)

    def update_label(self, name):
        value = self.vars[name].get()
        self.labels[name].config(text=f"{int(round(value))}" if self.integer[name] else f"{value:.2f}")

    def get(self, name):
        value = self.vars[name].get()
        return int(round(value)) if self.integer[name] else float(value)

    def settings(self):
        g = self.get
        return Settings(method=self.method.get(), shape=self.shape.get(), size=g("size"), lap_time=g("lap_time"),
                        duration=g("duration"), replan_interval=g("replan_interval"), cruise_speed=g("cruise_speed"),
                        slow_distance=g("slow_distance"), replan_error=g("replan_error"), kp=g("kp"), ki=g("ki"), kd=g("kd"),
                        feedforward=g("feedforward"), start=(g("start_x"), g("start_y")),
                        velocity=(g("velocity_x"), g("velocity_y")), seconds=g("seconds"), friction=g("friction"),
                        servo_tau=g("servo_tau"), camera_delay_frames=g("camera_delay_frames"),
                        noise_cm=g("noise_cm"), tilt_bias=(g("bias_x"), g("bias_y")))

    # ---- running
    def refresh_sliders(self):
        """Show only the settings that change something for the chosen method and shape."""
        relevant = relevant_settings(self.method.get(), self.shape.get())
        hideable = {"duration", "replan_interval", "feedforward", "cruise_speed", "slow_distance", "replan_error",
                    "size", "lap_time"}
        for frames in self.frames.values():
            for _, frame in frames:
                frame.pack_forget()
            for name, frame in frames:
                if name not in hideable or name in relevant:
                    frame.pack(fill=tk.X)

    def maybe_run(self):
        if self.auto.get():
            self.run_once()

    def run_once(self):
        self.refresh_sliders()
        s = self.settings()
        result = run(s)
        self.counter += 1
        name = s.method + (f" {s.shape}" if s.shape != "none" else "")
        label = f"#{self.counter} {name} Kp{s.kp:.2f} Kd{s.kd:.2f} FF{s.feedforward:.2f}"
        self.runs = (self.runs if self.keep.get() else []) + [(label, result)]
        self.runs = self.runs[-6:]
        draw(self.figure, self.runs)
        self.canvas.draw_idle()
        self.text.config(text=summary(result))

    def clear(self):
        self.runs = []
        self.run_once()

    def reset(self):
        defaults = Settings()
        values = {**vars(defaults), "start_x": defaults.start[0], "start_y": defaults.start[1],
                  "velocity_x": defaults.velocity[0], "velocity_y": defaults.velocity[1],
                  "bias_x": defaults.tilt_bias[0], "bias_y": defaults.tilt_bias[1]}
        for name, var in self.vars.items():
            var.set(values[name])
            self.update_label(name)
        self.method.set(defaults.method)
        self.shape.set(defaults.shape)
        self.clear()


if __name__ == "__main__":
    gui = TwinGui()
    gui.root.mainloop()
