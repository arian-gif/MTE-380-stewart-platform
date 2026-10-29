"""Move one Stewart-platform motor at a time through the Arduino motor shield.

Usage:
    python motor_control.py 1 90            # motor 1 -> 90 deg
    python motor_control.py 2 45 --dry-run  # print what would be sent, no hardware
    python motor_control.py --sweep 3       # sweep motor 3 min..max..min
    python motor_control.py                 # interactive: type "<motor> <angle>", "q" to quit

Requires pyserial (pip install pyserial) unless --dry-run is used.
Matching firmware: arduino/motor_control/motor_control.ino
Wire protocol (115200 baud, newline terminated):  "M<motor> <angle>\\n"  ->  "OK <motor> <angle>"

NOTE: the angle sent is the motor angle theta_i1 from stewart_kinematics.py.
"""
import argparse
import sys
import time

# COM port of the Arduino with the motor shield (Windows: Device Manager > Ports (COM & LPT)).
# Can be overridden with --port COMx.
COM_PORT = "COM5"
BAUD = 115200

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

    def move(self, motor, angle):
        if motor not in MOTORS:
            raise ValueError(f"motor must be one of {sorted(MOTORS)}")
        lim = MOTORS[motor]
        if not lim["min"] <= angle <= lim["max"]:
            raise ValueError(f"motor {motor}: angle {angle} outside [{lim['min']}, {lim['max']}]")
        cmd = f"M{motor} {angle:g}\n"
        if self.dry_run:
            print("would send:", cmd.strip())
            return
        self.ser.write(cmd.encode())
        print(self.ser.readline().decode().strip() or "(no reply)")

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
