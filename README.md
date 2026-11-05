# Stewart platform

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

| Command | What it does |
|---|---|
| `python stewart_kinematics.py` | Prints motor angles for a few tilts, with a forward-kinematics check |
| `python stewart_animation.py` | Writes `static_configs.png`, `animation_frames.png`, `stewart_animation.gif` |
| `python platform_control.py --dry-run` | Type `alpha beta` (deg, 80..100, level = 90 90); computes and sends all 3 motor angles (use `--port COMx` on hardware) |
| `python ball_tracking/simple_cal.py` | Camera calibration for the plate: click the ball and press `c`, size the circle with `+`/`-` (`[`/`]` big steps) and move the plate to fit it, press `g`, then `s` to save `config.json` (`l` levels the platform, `r` redoes the circle) |
| `python ball_tracking/ball_detection.py` | Live test of ball tracking; shows the ball's x, y (mm from the plate centre) and velocity (mm/s) once calibrated; press `q` to save `ball_tracking/tracks/ball_track_<date>.csv` + `.png` and show position-time and velocity-time graphs |
| `python ball_tracking/basic_controller.py --dry-run` | PID ball-balancing controller: camera -> ball x, y and velocity -> PID per axis -> plate tilt -> servos. GUI sliders for Kp, Ki, Kd and the x/y setpoint; tick "Enable control" to start. Drop `--dry-run` to move the real platform. When you stop, each run is saved as `ball_tracking/tuning_runs/csv/run_<date>_<time>.csv` (all logged values) and `ball_tracking/tuning_runs/plots/run_<date>_<time>.png` (position, setpoint and tilt plots) with matching names. The **Plot results** button also shows the plot |
| `python motor_control.py 1 90 --dry-run` | Shows the command that would be sent to motor 1 (no hardware) |
| `python motor_control.py 1 90 --port COMx` | Moves motor 1 to 90 deg |

## Files

- `functions.py` — slide forward/inverse kinematics for the three actuators
- `stewart_kinematics.py` — platform angles (alpha, beta) -> motor angles, following the "Platform Angle Control" notes
- `stewart_animation.py` — plots and GIF
- `motor_control.py` + `arduino/motor_control.ino` — move one motor at a time, and the quiet batch commands used by the controller. The sketch (version "v2") has a safety timeout: once the controller has told it the level pose, it goes there if commands stop for 0.5 s. **Re-upload the sketch whenever it changes**; the controller warns if the board still has an old version.
- `ball_tracking/` — camera and control (run from anywhere; they find their own `config.json`)
  - `simple_cal.py` — calibration, writes `config.json` (ball colour, plate centre and scale)
  - `ball_detection.py` — ball detector, `BallTracker` (alpha-beta position and velocity), CSV and plots
  - `basic_controller.py` — the PID controller

## TODO

- COM port is set to `COM5` (Arduino Mega 2560, `COM_PORT` in `motor_control.py`); it can change if you use a different USB port.
- Hardware is an Adafruit 16-channel PWM/Servo shield (I2C): servo channels in the `.ino` (`MOTOR_CHANNELS`, currently 15, 14, 13) are placeholders, and the pulse range (`MIN_US`/`MAX_US`) needs calibrating. Install the "Adafruit PWM Servo Driver Library" in the Arduino IDE.
- The shield needs its own 5-6 V supply for the servos (screw terminal); the Arduino's 5 V cannot power them.
- Motors are assumed to be hobby servos.
- Calibrate motor angle zero (level platform is theta = 0, link horizontal) against the real servos.
- Controller: with the ball on the plate, check the tilt directions (`TILT_SIGN_X` / `TILT_SIGN_Y` in `ball_tracking/basic_controller.py`; flip one if the ball runs away along that axis), then tune Kp, then Kd, then a little Ki. The default gains (Kp 0.75, Ki 0, Kd 0.35) are a physics-based first guess, not tuned on hardware.
