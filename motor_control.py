"""Move one Stewart-platform motor at a time through the Arduino motor shield.

Usage:
    python motor_control.py 1 90            # motor 1 -> 90 deg
    python motor_control.py 2 45 --dry-run  # print what would be sent, no hardware
    python motor_control.py --sweep 3       # sweep motor 3 min..max..min
    python motor_control.py                 # interactive: type "<motor> <angle>", "q" to quit

Requires pyserial (pip install pyserial) unless --dry-run is used.
Matching firmware: arduino/motor_control.ino (version "v2", re-upload it if you see a warning about that)
Wire protocol (115200 baud, newline terminated):
    "M<motor> <angle>"  move a motor, reply "OK <motor> <angle>"       (used by move)
    "m<motor> <angle>"  same, no reply                                   (used by move_many, the control loop)
    "s<motor> <angle>"  store a motor's safe angle, no reply             (used by set_safe_angles)
    Once all three safe angles are stored, the Arduino goes to them if no m/M command arrives for 0.5 s.

NOTE: the angle sent is the motor angle theta_i1 from stewart_kinematics.py.
"""
import argparse
import sys
import time

# COM port of the Arduino with the motor shield (Windows: Device Manager > Ports (COM & LPT)).
# Can be overridden with --port COMx.
COM_PORT = "COM5"
BAUD = 115200
FIRMWARE_VERSION = "v2"   # what the sketch prints after "READY" when the board starts

# TODO: confirm per motor once wired/assembled. Limits protect the linkage.
MOTORS = {
    1: {"min": 0, "max": 180},
    2: {"min": 0, "max": 180},
    3: {"min": 0, "max": 180},
}


class MotorLink:
    def __init__(self, port, dry_run=False):
        self.dry_run = dry_run
        self.ser = None
        if dry_run:
            return
        if not port:
            sys.exit("COM port not set: edit COM_PORT in motor_control.py or pass --port COMx "
                     "(or use --dry-run). Port is unknown until the wiring is done.")
        import serial  # imported lazily so --dry-run works without pyserial
        import serial.tools.list_ports
        try:
            self.ser = serial.Serial(port, BAUD, timeout=2)
        except serial.SerialException as error:
            ports = "\n".join(f"  {p.device}: {p.description}" for p in serial.tools.list_ports.comports())
            sys.exit(f"Could not open {port}: {error}\n"
                     f"Is the Arduino plugged in (USB) and the Serial Monitor closed? Ports on this PC:\n{ports}\n"
                     "Use the one that says Arduino / USB Serial, via --port COMx or COM_PORT in motor_control.py.")
        time.sleep(2)  # Arduino resets when the port opens
        banner = self.ser.read_all().decode(errors="ignore")   # "READY v2" if the board is running the current sketch
        if "READY" in banner and FIRMWARE_VERSION not in banner:
            print("WARNING: the sketch on the Arduino is out of date. Re-upload arduino/motor_control.ino "
                  "from the Arduino IDE, otherwise the control loop's commands will be rejected.")

    def _command(self, letter, motor, angle):
        """Check the motor number and angle, and return the serial command for them."""
        if motor not in MOTORS:
            raise ValueError(f"motor must be one of {sorted(MOTORS)}")
        lim = MOTORS[motor]
        if not lim["min"] <= angle <= lim["max"]:
            raise ValueError(f"motor {motor}: angle {angle} outside [{lim['min']}, {lim['max']}]")
        return f"{letter}{motor} {angle:g}\n"

    def move(self, motor, angle):
        cmd = self._command("M", motor, angle)
        if self.dry_run:
            print("would send:", cmd.strip())
            return
        self.ser.write(cmd.encode())
        print(self.ser.readline().decode().strip() or "(no reply)")

    def move_many(self, angles, verbose=False):
        """Send several motors at once, e.g. {1: 90, 2: 85, 3: 95}, without waiting for replies.

        For control loops that send many updates per second. All angles are checked before
        anything is sent, so a bad value moves no motor.
        """
        self._send_quiet("m", angles, verbose)

    def set_safe_angles(self, angles, verbose=False):
        """Tell the Arduino where to put the servos if this program stops sending, e.g. {1: 78, 2: 68, 3: 78}.

        Does not move anything. Once all three are set, the Arduino goes to these angles if no
        move_many/move command arrives for 0.5 s. Opening the serial port resets the Arduino, which
        clears them, so a later program always starts without a safe pose.
        """
        self._send_quiet("s", angles, verbose)

    def _send_quiet(self, letter, angles, verbose):
        commands = "".join(self._command(letter, motor, angle) for motor, angle in angles.items())
        if self.dry_run:
            if verbose:
                print("would send:", commands.strip().replace("\n", " | "))
            return
        self.ser.reset_input_buffer()  # discard any replies the Arduino sent meanwhile
        self.ser.write(commands.encode())

    def sweep(self, motor, step=5, delay=0.05):
        lo, hi = MOTORS[motor]["min"], MOTORS[motor]["max"]
        path = list(range(lo, hi + 1, step)) + list(range(hi, lo - 1, -step))
        for a in path:
            self.move(motor, a)
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motor", nargs="?", type=int)
    ap.add_argument("angle", nargs="?", type=float)
    ap.add_argument("--port", default=COM_PORT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sweep", type=int, metavar="MOTOR")
    args = ap.parse_args()

    link = MotorLink(args.port, args.dry_run)
    if args.sweep:
        link.sweep(args.sweep)
    elif args.motor is not None and args.angle is not None:
        try:
            link.move(args.motor, args.angle)
        except ValueError as error:
            sys.exit(f"error: {error}")
    else:
        while True:
            line = input("motor angle > ").strip()
            if line in ("q", "quit", ""):
                break
            try:
                m, a = line.split()
                link.move(int(m), float(a))
            except ValueError as e:
                print("error:", e)


if __name__ == "__main__":
    main()
