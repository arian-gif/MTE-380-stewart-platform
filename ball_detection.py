# Ball Detection Module for Computer Vision Ball Tracking System
# Detects colored balls in video frames using HSV color space filtering
# Provides both class-based and legacy function interfaces

import cv2
import numpy as np
import json
import os

DISPLAY_SIZE = 480  # side of the square preview window in pixels (smaller number = zoomed out more)

# Ask the camera for its widest 16:9 mode (4:3 modes are often cropped from it), then shrink every
# frame to PROCESS_SIZE. Keep the width at 640 so pixel_to_meter_ratio and the radius filter keep their scale.
CAPTURE_SIZE = (1280, 720)
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
    
    def __init__(self, config_file="config.json"):
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
        x_m = (center[0] - self.plate_center_px[0]) * self.meters_per_pixel
        y_m = (self.plate_center_px[1] - center[1]) * self.meters_per_pixel
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

# For testing/calibration when run directly
def main():
    """Test ball detection with current config."""
    detector = BallDetector()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
    
    print("Ball Detection Test")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, PROCESS_SIZE)
        
        # Get detection results with overlay
        if detector.plate_center_px is not None:
            # Stewart platform calibration found: 2D position on the plate
            vis_frame, found, position_m = detector.draw_detection_xy(frame)
            if found:
                print(f"Ball at x={position_m[0] * 1000:.1f} mm, y={position_m[1] * 1000:.1f} mm from plate centre")
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

if __name__ == "__main__":
    main()