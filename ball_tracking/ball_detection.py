# Ball Detection Module for Computer Vision Ball Tracking System
# Detects colored balls in video frames using HSV color space filtering
# Provides both class-based and legacy function interfaces

import cv2
import numpy as np
import csv
import json
import os
import time
from datetime import datetime

# Files live next to this script, so everything works no matter which folder you run it from
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")  # written by simple_cal.py
TRACKS_DIR = os.path.join(HERE, "tracks")  # CSVs and plots saved when you quit ball_detection.py

DISPLAY_SIZE = 480  # side of the square preview window in pixels (smaller number = zoomed out more)

# Capture 16:9 at 640x360. Measured on this webcam: 30 fps at 640x360 and 640x480, but only 20 fps at
# 960x540 and 10 fps at 1280x720 (too slow to balance a ball), so do not raise this without re-measuring.
# Every frame is resized to PROCESS_SIZE (a no-op while the two match) so pixel_to_meter_ratio and the
# radius filter keep their scale.
CAPTURE_SIZE = (640, 360)
PROCESS_SIZE = (640, 360)

def fit_square(frame, size=DISPLAY_SIZE):
    """Scale the whole frame to fit a size x size square, padding with black (nothing is cropped).

    Only for display; detection still runs on the original frame.
    Returns (square_image, scale, pad_x, pad_y): a frame point (x, y) shows up at
    (x * scale + pad_x, y * scale + pad_y) in the square.
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    pad_x = (size - resized.shape[1]) // 2
    pad_y = (size - resized.shape[0]) // 2
    square = np.zeros((size, size, 3), dtype=frame.dtype)
    square[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
    return square, scale, pad_x, pad_y

class BallDetector:
    """Computer vision ball detector using HSV color space filtering."""
    
    def __init__(self, config_file=CONFIG_PATH):
        """Initialize ball detector with HSV bounds from config file.
        
        Args:
            config_file (str): Path to JSON config file with HSV bounds and calibration
        """
        # Default HSV bounds for orange ball detection
        self.lower_hsv = np.array([5, 150, 150], dtype=np.uint8)  # Orange lower bound
        self.upper_hsv = np.array([20, 255, 255], dtype=np.uint8)  # Orange upper bound
        self.scale_factor = 1.0  # Conversion factor from normalized coords to meters
        self.min_radius_px, self.max_radius_px = 25, 100  # Accepted ball radius in pixels
        self.plate_center_px = None  # Stewart platform: plate centre in the frame (pixels)
        self.meters_per_pixel = None  # Stewart platform: scale of the frame at the plate

        # Load configuration from file if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Extract HSV color bounds from config
                if 'ball_detection' in config:
                    if config['ball_detection']['lower_hsv']:
                        self.lower_hsv = np.array(config['ball_detection']['lower_hsv'], dtype=np.uint8)
                    if config['ball_detection']['upper_hsv']:
                        self.upper_hsv = np.array(config['ball_detection']['upper_hsv'], dtype=np.uint8)
                    self.min_radius_px = config['ball_detection'].get('min_radius_px', self.min_radius_px)
                    self.max_radius_px = config['ball_detection'].get('max_radius_px', self.max_radius_px)

                # Stewart platform calibration: plate centre (pixels) and metres per pixel
                calibration = config.get('calibration', {})
                self.meters_per_pixel = calibration.get('pixel_to_meter_ratio')
                self.plate_center_px = calibration.get('plate_center_px')

                # Extract scale factor for position conversion from pixels to meters
                if 'calibration' in config and 'pixel_to_meter_ratio' in config['calibration']:
                    if config['calibration']['pixel_to_meter_ratio']:
                        frame_width = config.get('camera', {}).get('frame_width', 640)
                        self.scale_factor = config['calibration']['pixel_to_meter_ratio'] * (frame_width / 2)
                
                print(f"[BALL_DETECT] Loaded HSV bounds: {self.lower_hsv} to {self.upper_hsv}")
                print(f"[BALL_DETECT] Scale factor: {self.scale_factor:.6f} m/normalized_unit")
                
            except Exception as e:
                print(f"[BALL_DETECT] Config load error: {e}, using defaults")
        else:
            print("[BALL_DETECT] No config file found, using default HSV bounds")

    def detect_ball(self, frame):
        """Detect ball in frame and return detection results.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            found (bool): True if ball detected
            center (tuple): (x, y) pixel coordinates of ball center
            radius (float): Ball radius in pixels
            position_m (float): Ball position in meters from center
        """
        # Convert frame from BGR to HSV color space for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create binary mask using HSV color bounds
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Clean up mask using morphological operations
        mask = cv2.erode(mask, None, iterations=2)  # Remove noise
        mask = cv2.dilate(mask, None, iterations=2)  # Fill gaps
        
        # Find all contours in the cleaned mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, None, 0.0
        
        # Select the largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum enclosing circle around the contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Filter out detections that are too small or too large
        if radius < self.min_radius_px or radius > self.max_radius_px:
            return False, None, None, 0.0
        
        # Convert pixel position to meters from center
        center_x = frame.shape[1] // 2  # Frame center x-coordinate
        normalized_x = (x - center_x) / center_x  # Normalize to -1 to +1 range
        position_m = normalized_x * self.scale_factor  # Convert to meters
        
        self.last_center_subpixel = (x, y)  # unrounded centre, used for the Stewart platform position
        return True, (int(x), int(y)), radius, position_m

    def draw_detection(self, frame, show_info=True):
        """Detect ball and draw detection overlay on frame.
        
        Args:
            frame: Input BGR image frame
            show_info (bool): Whether to display position information text
            
        Returns:
            frame_with_overlay: Frame with detection drawn
            found: True if ball detected
            position_m: Ball position in meters
        """
        # Perform ball detection
        found, center, radius, position_m = self.detect_ball(frame)
        
        # Create overlay copy for drawing
        overlay = frame.copy()
        
        # Draw vertical center reference line
        height, width = frame.shape[:2]
        center_x = width // 2
        cv2.line(overlay, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.putText(overlay, "Center", (center_x + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if found:
            # Draw circle around detected ball
            cv2.circle(overlay, center, int(radius), (0, 255, 0), 2)  # Green circle
            cv2.circle(overlay, center, 3, (0, 255, 0), -1)  # Green center dot
            
            if show_info:
                # Display ball position information
                cv2.putText(overlay, f"x: {center[0]}", (center[0] - 30, center[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(overlay, f"pos: {position_m:.4f}m", (center[0] - 40, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return overlay, found, position_m

    def detect_ball_xy(self, frame):
        """Stewart platform: ball position on the plate in meters.

        The origin is the plate centre from the calibration, +x is to the right in the image and
        +y is up in the image.

        Returns:
            found (bool): True if the ball was detected (and the plate is calibrated)
            position_m (tuple): (x, y) in meters from the plate centre, or None
            center (tuple): (x, y) pixel coordinates of the ball center, or None
            radius (float): Ball radius in pixels, or None
        """
        found, center, radius, _ = self.detect_ball(frame)
        if not found or self.plate_center_px is None or not self.meters_per_pixel:
            return False, None, center, radius
        sub_x, sub_y = self.last_center_subpixel
        x_m = (sub_x - self.plate_center_px[0]) * self.meters_per_pixel
        y_m = (self.plate_center_px[1] - sub_y) * self.meters_per_pixel
        return True, (x_m, y_m), center, radius

    def draw_detection_xy(self, frame):
        """Detect the ball with detect_ball_xy and draw the plate centre, ball and position.

        Returns:
            frame_with_overlay, found, position_m ((x, y) in meters or None)
        """
        found, position_m, center, radius = self.detect_ball_xy(frame)
        overlay = frame.copy()

        # Plate centre crosshair
        if self.plate_center_px is not None:
            cx, cy = int(self.plate_center_px[0]), int(self.plate_center_px[1])
            cv2.line(overlay, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
            cv2.line(overlay, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

        if found:
            cv2.circle(overlay, center, int(radius), (0, 255, 0), 2)
            cv2.circle(overlay, center, 3, (0, 255, 0), -1)
            cv2.putText(overlay, f"x: {position_m[0] * 1000:.1f}  y: {position_m[1] * 1000:.1f} mm",
                       (center[0] - 60, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return overlay, found, position_m

class BallTracker:
    """Stewart platform: ball position (m from the plate centre) and velocity (m/s) over time.

    Uses an alpha-beta filter. Each frame it
        1. predicts where the ball should be:   predicted = position + velocity * dt
        2. compares that with the camera:       error = measured - predicted
        3. corrects:                            position += alpha * error
                                                velocity += (beta / dt) * error
    so position and velocity are smoothed together and a missed frame is covered by the prediction.
    +x is right and +y is up in the image, same as BallDetector.detect_ball_xy.
    """

    def __init__(self, detector=None, alpha=0.5, beta=None, max_gap_s=0.3):
        """
        Args:
            detector: BallDetector to use (default: one built from config.json)
            alpha (float): 0..1, how much of the position error is corrected each frame.
                Larger trusts the camera more (less lag, more noise).
            beta (float): how much of the error goes into the velocity.
                Default alpha^2 / (2 - alpha), a standard well-behaved pairing for alpha.
            max_gap_s (float): if the ball was lost for longer than this, tracking restarts
        """
        self.detector = detector or BallDetector()
        self.alpha = alpha
        self.beta = beta if beta is not None else alpha ** 2 / (2 - alpha)
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self):
        """Forget the previous position and velocity."""
        self.position = None  # filtered (x, y) in meters, None until the ball is first seen
        self.velocity = np.zeros(2)  # filtered (vx, vy) in m/s
        self.raw_position_m = None  # unfiltered camera measurement of the last found frame
        self.last_time = None
        self.found = False
        self.center_px = None
        self.radius_px = None

    @property
    def position_m(self):
        return None if self.position is None else (float(self.position[0]), float(self.position[1]))

    @property
    def velocity_mps(self):
        return (float(self.velocity[0]), float(self.velocity[1]))

    def update(self, frame, timestamp=None):
        """Detect the ball in this frame and update the filter.

        Args:
            frame: Input BGR image frame
            timestamp (float): time the frame was captured in seconds (default: now)

        Returns:
            found (bool): True if the ball was detected
            position_m (tuple): filtered (x, y) in meters from the plate centre, or None
            velocity_mps (tuple): filtered (vx, vy) in m/s ((0, 0) if the ball is not found)
        """
        now = time.time() if timestamp is None else timestamp
        self.found, measured, self.center_px, self.radius_px = self.detector.detect_ball_xy(frame)

        if not self.found:
            return False, None, (0.0, 0.0)

        measured = np.array(measured)
        self.raw_position_m = tuple(measured)

        dt = None if self.last_time is None else now - self.last_time
        if self.position is None or dt is None or not (0 < dt <= self.max_gap_s):
            # First sighting (or lost for too long): start from the measurement with zero velocity
            self.position = measured
            self.velocity = np.zeros(2)
        else:
            predicted = self.position + self.velocity * dt  # 1. predict
            error = measured - predicted                    # 2. compare with the camera
            self.position = predicted + self.alpha * error  # 3. correct position...
            self.velocity = self.velocity + (self.beta / dt) * error  # ...and velocity

        self.last_time = now
        return True, self.position_m, self.velocity_mps

    def draw(self, frame, arrow_seconds=0.3):
        """Draw the last update on the frame: plate centre, ball, position, velocity and an arrow.

        The arrow shows where the ball would be after arrow_seconds at its current velocity.
        """
        overlay = frame.copy()

        # Plate centre crosshair
        if self.detector.plate_center_px is not None:
            cx, cy = int(self.detector.plate_center_px[0]), int(self.detector.plate_center_px[1])
            cv2.line(overlay, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
            cv2.line(overlay, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

        if self.found:
            center = self.center_px
            vx, vy = self.velocity_mps
            x, y = self.position_m
            cv2.circle(overlay, center, int(self.radius_px), (0, 255, 0), 2)
            cv2.circle(overlay, center, 3, (0, 255, 0), -1)

            # Velocity arrow (image y points down, so flip vy)
            pixels_per_meter = 1.0 / self.detector.meters_per_pixel
            tip = (int(center[0] + vx * arrow_seconds * pixels_per_meter),
                   int(center[1] - vy * arrow_seconds * pixels_per_meter))
            cv2.arrowedLine(overlay, center, tip, (0, 0, 255), 2, tipLength=0.3)

            cv2.putText(overlay, f"x: {x * 1000:.1f}  y: {y * 1000:.1f} mm",
                       (center[0] - 60, center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(overlay, f"vx: {vx * 1000:.0f}  vy: {vy * 1000:.0f} mm/s",
                       (center[0] - 60, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return overlay

# Legacy function for backward compatibility with existing code
def detect_ball_x(frame):
    """Legacy function that matches the old ball_detection.py interface.
    
    This function maintains compatibility with existing code that expects
    the original function signature and return format.
    
    Args:
        frame: Input BGR image frame
        
    Returns:
        found (bool): True if ball detected
        x_normalized (float): Normalized x position (-1 to +1)
        vis_frame (array): Frame with detection overlay
    """
    # Create detector instance using default config
    detector = BallDetector()
    
    # Get detection results with visual overlay
    vis_frame, found, position_m = detector.draw_detection(frame)
    
    if found:
        # Convert back to normalized coordinates for legacy compatibility
        x_normalized = position_m / detector.scale_factor if detector.scale_factor != 0 else 0.0
        x_normalized = np.clip(x_normalized, -1.0, 1.0)  # Ensure within bounds
    else:
        x_normalized = 0.0
    
    return found, x_normalized, vis_frame

class TrackLog:
    """Records the ball's position and velocity over time, then saves a CSV and plots it.

    Units in the file and on the plots are mm and mm/s, measured from the plate centre
    (+x right, +y up in the image). Frames where the ball was not found are logged as blanks,
    so the plot shows a gap there.
    """

    COLUMNS = ["time_s", "found", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s", "raw_x_mm", "raw_y_mm"]

    def __init__(self):
        self.start_time = None
        self.rows = []

    def add(self, timestamp, found, position_m, velocity_mps, raw_position_m):
        """Record one frame. position_m / velocity_mps are the filtered values from BallTracker."""
        if self.start_time is None:
            self.start_time = timestamp
        t = timestamp - self.start_time
        if found:
            self.rows.append([t, 1,
                              position_m[0] * 1000, position_m[1] * 1000,
                              velocity_mps[0] * 1000, velocity_mps[1] * 1000,
                              raw_position_m[0] * 1000, raw_position_m[1] * 1000])
        else:
            self.rows.append([t, 0] + [float("nan")] * 6)

    def save_csv(self, path=None):
        """Write the log to a CSV file (default name has the date and time). Returns the path."""
        if path is None:
            os.makedirs(TRACKS_DIR, exist_ok=True)
            path = os.path.join(TRACKS_DIR, f"ball_track_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for row in self.rows:
                # Blank cells (not "nan") for frames where the ball was not found
                writer.writerow([f"{row[0]:.4f}", int(row[1])] + [f"{v:.4f}" if v == v else "" for v in row[2:]])
        return path

    def plot(self, csv_path=None, show=True):
        """Draw position-time and velocity-time graphs. Saved as a PNG next to the CSV if one is given."""
        import matplotlib.pyplot as plt  # imported here so the detector works without matplotlib

        data = np.array(self.rows, dtype=float)
        t = data[:, 0]
        fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Position vs time: filtered (solid) and the raw camera measurement (faint)
        ax_pos.plot(t, data[:, 6], color="tab:blue", alpha=0.3, label="x raw")
        ax_pos.plot(t, data[:, 7], color="tab:orange", alpha=0.3, label="y raw")
        ax_pos.plot(t, data[:, 2], color="tab:blue", label="x")
        ax_pos.plot(t, data[:, 3], color="tab:orange", label="y")
        ax_pos.set_ylabel("Position (mm)")
        ax_pos.set_title("Ball position vs time (from plate centre)")

        # Velocity vs time
        ax_vel.plot(t, data[:, 4], color="tab:blue", label="vx")
        ax_vel.plot(t, data[:, 5], color="tab:orange", label="vy")
        ax_vel.set_ylabel("Velocity (mm/s)")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_title("Ball velocity vs time")

        for ax in (ax_pos, ax_vel):
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        fig.tight_layout()

        if csv_path:
            png_path = os.path.splitext(csv_path)[0] + ".png"
            fig.savefig(png_path, dpi=120)
            print(f"Saved plot to {png_path}")
        if show:
            plt.show()
        plt.close(fig)

# For testing/calibration when run directly
def main():
    """Test ball detection with current config."""
    detector = BallDetector()
    tracker = BallTracker(detector)
    log = TrackLog()  # position and velocity over time, saved when you quit
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
    
    print("Ball Detection Test")
    print("Press 'q' to quit (saves a CSV and shows position/velocity graphs)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, PROCESS_SIZE)
        
        # Get detection results with overlay
        if detector.plate_center_px is not None:
            # Stewart platform calibration found: 2D position and velocity on the plate
            now = time.time()
            found, position_m, velocity_mps = tracker.update(frame, now)
            log.add(now, found, position_m, velocity_mps, tracker.raw_position_m)
            vis_frame = tracker.draw(frame)
            if found:
                print(f"Ball at x={position_m[0] * 1000:.1f} mm, y={position_m[1] * 1000:.1f} mm  |  "
                      f"vx={velocity_mps[0] * 1000:.0f} mm/s, vy={velocity_mps[1] * 1000:.0f} mm/s")
        else:
            vis_frame, found, position_m = detector.draw_detection(frame)
            if found:
                print(f"Ball detected at {position_m:.4f}m from center")
        
        # Display frame with detection overlay
        cv2.imshow("Ball Detection Test", fit_square(vis_frame)[0])
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Save the recorded track and plot it
    if log.rows:
        csv_path = log.save_csv()
        print(f"Saved {len(log.rows)} samples to {csv_path}")
        log.plot(csv_path)
    else:
        print("Nothing recorded: no plate calibration found. Run simple_cal.py first.")

if __name__ == "__main__":
    main()