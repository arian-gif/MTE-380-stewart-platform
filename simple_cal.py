# Simple Calibration for the Stewart Platform (ball on a plate, camera looking straight down)
# Interactive tool for ball colour and plate geometry. Writes config.json for ball_detection.py.
#
# Phase 1: click the ball to sample its colour, press 'c'
# Phase 2: a circle is drawn at the centre of the view. Size it with '+' / '-' ('[' / ']' for big steps),
#          move the real platform until its edge fits the circle, then press 'g'
# Then press 's' to save. Press 'l' any time to send the platform to level (alpha = beta = 90) and
# check the servos, 'r' to redo the circle, 'q' to quit.

import cv2
import numpy as np
import json
import math
from datetime import datetime

import motor_control
import platform_control
from ball_detection import fit_square, CAPTURE_SIZE, PROCESS_SIZE

# Radius of the real plate, the size the phase 2 circle stands for (notes: r = 135 mm).
# TODO: measure on the real platform.
PLATE_RADIUS_M = 0.135

LEVEL_ALPHA, LEVEL_BETA = 90.0, 90.0  # platform angles for a level plate

class StewartCalibrator:
    """Interactive calibration for the ball-on-plate camera setup."""

    def __init__(self):
        """Initialize calibration parameters and default values."""
        # Camera configuration
        self.CAM_INDEX = 1  # Default camera index
        self.FRAME_W, self.FRAME_H = PROCESS_SIZE  # Frames are shrunk to this size before use

        # Calibration state tracking
        self.current_frame = None  # Current video frame
        self.phase = "color"  # Current phase: "color", "geometry", "complete"

        # Color calibration data
        self.hsv_samples = []  # Collected HSV color samples
        self.lower_hsv = None  # Lower HSV bound for ball detection
        self.upper_hsv = None  # Upper HSV bound for ball detection
        self.ball_radius_px = None  # Ball radius seen when the colour phase ended

        # Geometry calibration data: a circle at the centre of the view that the plate is fitted to
        self.circle_center_px = (self.FRAME_W // 2, self.FRAME_H // 2)
        self.circle_radius_px = int(min(self.FRAME_W, self.FRAME_H) * 0.4)  # starting size
        self.plate_center_px = None  # Plate centre in the frame (pixels), set when you press 'g'
        self.pixel_to_meter_ratio = None  # Meters per pixel at the plate

        # Platform connection (Arduino with the servo shield), optional
        self.link = None

        # How the frame is placed in the square window (set each frame in run())
        self.display_scale, self.display_pad_x, self.display_pad_y = 1.0, 0, 0

    def read_frame(self):
        """Read a camera frame and shrink it to the processing size (FRAME_W x FRAME_H)."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.FRAME_W, self.FRAME_H))
        return ret, frame

    def connect_platform(self):
        """Open the serial link to the Arduino (port set in motor_control.py)."""
        try:
            self.link = motor_control.MotorLink(motor_control.COM_PORT)
            print("[PLATFORM] Connected")
        except SystemExit as error:
            self.link = None
            print(f"[PLATFORM] Not connected - 'l' (level) is disabled.\n{error.code}")

    def level_platform(self):
        """Send the platform to level so you can check the servos and the camera view."""
        if self.link is None:
            print("[PLATFORM] Not connected")
            return
        print("[PLATFORM] Moving to level")
        platform_control.tilt_platform(self.link, LEVEL_ALPHA, LEVEL_BETA)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for interactive calibration.

        Args:
            event: OpenCV mouse event type
            x, y: Mouse click coordinates in the window
            flags: Additional event flags
            param: User data (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # The window shows a scaled, padded copy of the frame: convert the click back to frame pixels
            x = int(round((x - self.display_pad_x) / self.display_scale))
            y = int(round((y - self.display_pad_y) / self.display_scale))
            if self.current_frame is None:
                return
            height, width = self.current_frame.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                return  # click landed on the black padding
            if self.phase == "color":
                # Color sampling phase - collect HSV samples at click point
                self.sample_color(x, y)

    def sample_color(self, x, y):
        """Sample HSV color values in a 5x5 region around click point.

        Args:
            x, y: Center coordinates for color sampling
        """
        if self.current_frame is None:
            return

        # Convert frame to HSV color space
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)

        # Sample 5x5 region around click point
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = x + dx, y + dy
                # Check bounds and collect valid samples
                if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                    self.hsv_samples.append(hsv[py, px])

        # Update HSV bounds based on collected samples
        if self.hsv_samples:
            samples = np.array(self.hsv_samples, dtype=int)  # int, not uint8: adding a margin to 255 would wrap around

            # Calculate adaptive margins for each HSV channel
            h_margin = max(5, (np.max(samples[:, 0]) - np.min(samples[:, 0])) * 0.1)
            s_margin = max(10, (np.max(samples[:, 1]) - np.min(samples[:, 1])) * 0.15)
            v_margin = max(10, (np.max(samples[:, 2]) - np.min(samples[:, 2])) * 0.15)

            # Set lower bounds with margin
            self.lower_hsv = [
                max(0, np.min(samples[:, 0]) - h_margin),
                max(0, np.min(samples[:, 1]) - s_margin),
                max(0, np.min(samples[:, 2]) - v_margin)
            ]

            # Set upper bounds with margin
            self.upper_hsv = [
                min(179, np.max(samples[:, 0]) + h_margin),
                min(255, np.max(samples[:, 1]) + s_margin),
                min(255, np.max(samples[:, 2]) + v_margin)
            ]

            print(f"[COLOR] Samples: {len(self.hsv_samples)}")

    def resize_circle(self, delta):
        """Grow or shrink the geometry circle by delta pixels (kept inside the frame height)."""
        limit = self.FRAME_H // 2
        self.circle_radius_px = int(np.clip(self.circle_radius_px + delta, 10, limit))

    def confirm_geometry(self):
        """The plate now fits the circle: take the circle as the plate and work out meters-per-pixel."""
        self.plate_center_px = self.circle_center_px
        self.pixel_to_meter_ratio = PLATE_RADIUS_M / self.circle_radius_px
        print(f"[GEO] Plate centre: {self.plate_center_px}, radius {self.circle_radius_px} px, "
              f"meters per pixel: {self.pixel_to_meter_ratio:.6f}")
        self.phase = "complete"

    def find_ball(self, frame):
        """Find the ball by colour.

        Returns:
            (x, y, radius) in pixels, or None if no ball is found
        """
        if not self.lower_hsv:
            return None

        # Convert to HSV and create color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.lower_hsv, dtype=np.uint8)
        upper = np.array(self.upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Clean up mask with morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Largest contour is assumed to be the ball
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)

        # Filter out very small detections
        if radius < 5:
            return None
        return x, y, radius

    def ball_position_m(self, x, y):
        """Ball pixel position -> (x, y) in meters from the plate centre (+x right, +y up)."""
        return ((x - self.plate_center_px[0]) * self.pixel_to_meter_ratio,
                (self.plate_center_px[1] - y) * self.pixel_to_meter_ratio)

    def save_config(self):
        """Save all calibration results to config.json file."""
        # Accept balls between half and one and a half times the radius seen when calibrating
        radius = self.ball_radius_px
        config = {
            "timestamp": datetime.now().isoformat(),
            "plate_radius_m": float(PLATE_RADIUS_M),
            "camera": {
                "index": int(self.CAM_INDEX),
                "frame_width": int(self.FRAME_W),
                "frame_height": int(self.FRAME_H)
            },
            "ball_detection": {
                "lower_hsv": [float(x) for x in self.lower_hsv] if self.lower_hsv else None,
                "upper_hsv": [float(x) for x in self.upper_hsv] if self.upper_hsv else None,
                "min_radius_px": float(radius * 0.5) if radius else 5.0,
                "max_radius_px": float(radius * 1.5) if radius else 100.0
            },
            "calibration": {
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None,
                "plate_center_px": [int(v) for v in self.plate_center_px] if self.plate_center_px else None
            },
            "platform": {
                "port": str(motor_control.COM_PORT),
                "level_alpha": LEVEL_ALPHA,
                "level_beta": LEVEL_BETA
            }
        }

        # Write configuration to JSON file
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json")

    def draw_overlay(self, frame):
        """Draw calibration status and instructions overlay on frame.

        Args:
            frame: Input BGR image frame

        Returns:
            numpy.ndarray: Frame with overlay graphics and text
        """
        overlay = frame.copy()

        # Phase-specific instruction text
        phase_text = {
            "color": "Click on the ball to sample its colour. Press 'c' when done.",
            "geometry": "Move the plate to fit the circle. +/- size ([ ] big steps), 'g' when it fits",
            "complete": "Calibration complete! Press 's' to save"
        }

        # Draw current phase and instructions
        cv2.putText(overlay, f"Phase: {self.phase}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, phase_text[self.phase], (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show color calibration progress
        if self.hsv_samples:
            cv2.putText(overlay, f"Color samples: {len(self.hsv_samples)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Geometry phase: the circle the plate is fitted to
        if self.phase == "geometry":
            cv2.circle(overlay, self.circle_center_px, self.circle_radius_px, (0, 255, 0), 2)
            cv2.drawMarker(overlay, self.circle_center_px, (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.putText(overlay, f"Circle radius: {self.circle_radius_px} px = {PLATE_RADIUS_M * 1000:.0f} mm",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw the plate outline once the geometry is known
        if self.plate_center_px:
            radius_px = int(PLATE_RADIUS_M / self.pixel_to_meter_ratio)
            cv2.circle(overlay, self.plate_center_px, radius_px, (255, 0, 0), 2)

        # Show real-time ball detection once the colour is calibrated
        ball = self.find_ball(frame)
        if ball:
            x, y, radius = ball
            cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 255), -1)

            # Show position (mm from the plate centre) once the geometry is known
            if self.plate_center_px:
                px, py = self.ball_position_m(x, y)
                cv2.putText(overlay, f"x: {px * 1000:.1f}  y: {py * 1000:.1f} mm",
                           (int(x) + 20, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Platform status
        status = "Platform: connected ('l' = level)" if self.link else "Platform: not connected"
        cv2.putText(overlay, status, (10, overlay.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return overlay

    def run(self):
        """Main calibration loop with interactive GUI."""
        # Initialize camera capture
        self.cap = cv2.VideoCapture(self.CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Setup OpenCV window and mouse callback
        cv2.namedWindow("Stewart Calibration")
        cv2.setMouseCallback("Stewart Calibration", self.mouse_callback)

        # Attempt platform connection
        self.connect_platform()

        # Display instructions
        print("[INFO] Stewart Platform Calibration")
        print("Phase 1: Click on the ball to sample its colour, press 'c' when done")
        print("Phase 2: Size the circle with + / - ([ ] for big steps), move the platform to fit it, press 'g'")
        print("Press 'l' to level the platform, 'r' to redo the circle, 's' to save, 'q' to quit")

        # Main calibration loop
        while True:
            ret, frame = self.read_frame()
            if not ret:
                continue

            self.current_frame = frame

            # Draw overlay and display frame
            display = self.draw_overlay(frame)
            display, self.display_scale, self.display_pad_x, self.display_pad_y = fit_square(display)
            cv2.imshow("Stewart Calibration", display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit calibration
                break
            elif key == ord('c') and self.phase == "color":
                # Complete color calibration phase
                if self.hsv_samples:
                    ball = self.find_ball(frame)
                    if ball:
                        self.ball_radius_px = ball[2]
                    self.phase = "geometry"
                    print("[INFO] Color calibration complete. Fit the circle to the plate and press 'g'.")
            elif key == ord('r') and self.phase in ("geometry", "complete"):
                # Redo the circle
                self.plate_center_px = None
                self.pixel_to_meter_ratio = None
                self.phase = "geometry"
                print("[INFO] Fit the circle to the plate and press 'g'.")
            elif self.phase == "geometry" and key in (ord('+'), ord('=')):
                self.resize_circle(2)
            elif self.phase == "geometry" and key in (ord('-'), ord('_')):
                self.resize_circle(-2)
            elif self.phase == "geometry" and key == ord(']'):
                self.resize_circle(10)
            elif self.phase == "geometry" and key == ord('['):
                self.resize_circle(-10)
            elif key == ord('g') and self.phase == "geometry":
                # The plate fits the circle
                self.confirm_geometry()
            elif key == ord('l'):
                # Send the platform to level
                self.level_platform()
            elif key == ord('s') and self.phase == "complete":
                # Save configuration and exit
                self.save_config()
                break

        # Clean up resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    """Run calibration when script is executed directly."""
    calibrator = StewartCalibrator()
    calibrator.run()
