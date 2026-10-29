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
| `python motor_control.py 1 90 --dry-run` | Shows the command that would be sent to motor 1 (no hardware) |
| `python motor_control.py 1 90 --port COMx` | Moves motor 1 to 90 deg |

## Files

- `functions.py` — slide forward/inverse kinematics for the three actuators
- `stewart_kinematics.py` — platform angles (alpha, beta) -> motor angles, following the "Platform Angle Control" notes
- `stewart_animation.py` — plots and GIF
- `motor_control.py` + `arduino/motor_control/motor_control.ino` — move one motor at a time

## TODO

- COM port is set to `COM5` (Arduino Mega 2560, `COM_PORT` in `motor_control.py`); it can change if you use a different USB port.
- Hardware is an Adafruit 16-channel PWM/Servo shield (I2C): servo channels in the `.ino` (`MOTOR_CHANNELS`, currently 15, 14, 13) are placeholders, and the pulse range (`MIN_US`/`MAX_US`) needs calibrating. Install the "Adafruit PWM Servo Driver Library" in the Arduino IDE.
- The shield needs its own 5-6 V supply for the servos (screw terminal); the Arduino's 5 V cannot power them.
- Motors are assumed to be hobby servos.
- Calibrate motor angle zero (level platform is theta = 0, link horizontal) against the real servos.
