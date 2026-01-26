import cv2
import numpy as np
import collections
import threading
import time
import pyrealsense2 as rs
from math import atan2, cos, sin, sqrt, pi, radians
from pathlib import Path

# --- Configuration ---
WINDOW_NAME = "Pipeline View"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
HALF_WIDTH = WINDOW_WIDTH // 2
HALF_HEIGHT = WINDOW_HEIGHT // 2

# Colors
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_PURPLE = (128, 0, 128)
COLOR_GRAY = (50, 50, 50)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

# Terminal Log
log_buffer = collections.deque(maxlen=15)

T_TCP_CAM_PATH = Path("T_tcp_cam.npy")
STREAM_W, STREAM_H, FPS = 640, 480, 30

def drawAxis(img, p_, q_, color, scale):
    """
    Draw axis arrows for visualization.
    Based on the OpenCV example code.
    """
    p = list(p_)
    q = list(q_)
    
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]
    
def getOrientation(pts, img):
    """
    Get orientation using PCA analysis on contour points.
    This finds the actual principal axes of the object for accurate orientation.
    Based on the OpenCV example code.
    """
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]
    
    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
    
    # Use minor axis (thin edge)
    angle = atan2(eigenvectors[1,1], eigenvectors[1,0])
    # orientation in radians
    ## [visualization]
    
    # Label with the rotation angle in radians (no background rectangle)
    label = f"Rotation: {angle:.3f} rad ({np.degrees(angle):.1f} deg)"
    
    # Get text size to position it better
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Position text above the object center to avoid overlap
    text_x = cntr[0] - text_width // 2  # Center the text horizontally
    text_y = cntr[1] - 40  # Position above the center
    
    # Draw text with black outline for better visibility
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)  # Black outline
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)  # White text
    
    return angle, cntr

def add_log(message):
    log_buffer.append(f"[{time.strftime('%H:%M:%S')}] {message}")

class CameraStream:
    """Class to handle a camera feed in a separate thread."""
    def __init__(self, src=0, name="Camera"):
        self.stream = cv2.VideoCapture(src)
        self.name = name
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.ret, self.frame = self.stream.read()

    def get_frame(self, width, height):
        if self.ret and self.frame is not None:
            return cv2.resize(self.frame, (width, height))
        return None

    def stop(self):
        self.stopped = True
        self.stream.release()

def draw_text_centered(img, text, center_x, center_y, color=(255, 255, 255), scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    cv2.putText(img, text, (center_x - text_size[0] // 2, center_y + text_size[1] // 2), 
                font, scale, color, thickness, cv2.LINE_AA)

def create_ui_frame(realsense_img, dobot_img):
    canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # --- LEFT SIDE: Camera Feeds ---
    # Realsense Slot (Top Left)
    if realsense_img is not None:
        canvas[0:HALF_HEIGHT, 0:HALF_WIDTH] = realsense_img
    cv2.putText(canvas, "REALSENSE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    cv2.rectangle(canvas, (0,0), (HALF_WIDTH, HALF_HEIGHT), COLOR_GRAY, 1)

    # DOBOT Slot (Bottom Left)
    if dobot_img is not None:
        canvas[HALF_HEIGHT:WINDOW_HEIGHT, 0:HALF_WIDTH] = dobot_img
    cv2.putText(canvas, "DOBOT (USB)", (10, HALF_HEIGHT + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    cv2.rectangle(canvas, (0, HALF_HEIGHT), (HALF_WIDTH, WINDOW_HEIGHT), COLOR_GRAY, 1)

    # --- RIGHT SIDE: Terminal (Top Right) ---
    term_x = HALF_WIDTH
    canvas[0:HALF_HEIGHT, term_x:WINDOW_WIDTH] = COLOR_WHITE
    cv2.putText(canvas, "SYSTEM LOG", (term_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PURPLE, 2)
    
    for i, line in enumerate(log_buffer):
        cv2.putText(canvas, f"> {line}", (term_x + 10, 70 + (i * 25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PURPLE, 1)

    # --- RIGHT SIDE: Buttons (Bottom Right) ---
    bx, by = term_x, HALF_HEIGHT
    # Button Draw Logic
    cv2.rectangle(canvas, (bx + 50, by + 50), (bx + 250, by + 120), COLOR_GREEN, -1)
    draw_text_centered(canvas, "START", bx + 150, by + 85, COLOR_BLACK)
    
    cv2.rectangle(canvas, (bx + 300, by + 50), (bx + 500, by + 120), COLOR_RED, -1)
    draw_text_centered(canvas, "STOP", bx + 400, by + 85, COLOR_WHITE)
    
    cv2.rectangle(canvas, (bx + 50, by + 150), (bx + 500, by + 220), COLOR_BLUE, -1)
    draw_text_centered(canvas, "CALIBRATE", bx + 275, by + 185, COLOR_WHITE)

    return canvas

def get_realsense_annotated_frame(pipeline, align, K, D, T_tcp_cam):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color = aligned_frames.get_color_frame()
    if not color:
        return None
    img = np.asanyarray(color.get_data())
    ui = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    detected_objects = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 5000 or 100000 < area:
            continue
        cv2.drawContours(ui, contours, i, (0, 0, 255), 2)
        try:
            angle, center_point = getOrientation(c, ui)
            detected_objects.append((c, area, angle, center_point))
        except Exception as e:
            add_log(f"[WARN] Orientation fail for contour {i}: {e}")
            continue
        # Optionally: annotate with orientation/position here if you have robot code
    return cv2.resize(ui, (HALF_WIDTH, HALF_HEIGHT))

def main():
    
   # Load camera calibration
    if not T_TCP_CAM_PATH.exists():
        print(f"[ERROR] {T_TCP_CAM_PATH} not found. Run calibration first.")
        return
    
    T_tcp_cam = np.load(T_TCP_CAM_PATH)
    print(f"[OK] Loaded {T_TCP_CAM_PATH}")
    
    # Setup RealSense
    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)

    
    realsense_available = True
    try:
        start_exception = []
        def start_pipeline():
            try:
                pipeline.start(config)
            except Exception as e:
                start_exception.append(e)
        start_thread = threading.Thread(target=start_pipeline)
        start_thread.start()
        start_thread.join(timeout=5.0)
        if start_thread.is_alive():
            realsense_available = False
            add_log("RealSense stream failed to start within 5 seconds.")
        elif start_exception:
            realsense_available = False
            add_log(f"Failed to start RealSense pipeline: {start_exception[0]}")
    except Exception as e:
        realsense_available = False
        add_log(f"RealSense error: {e}")

    if realsense_available:
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        cx = float(intrinsics.ppx)
        cy = float(intrinsics.ppy)
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=float)
        D = np.array(intrinsics.coeffs[:5], dtype=float)
    else:
        K = D = None

    dobot_stream = CameraStream(src=0, name="DOBOT").start()
    add_log("System Started")
    add_log("Threads Initialized")

    while True:
        d_frame = dobot_stream.get_frame(HALF_WIDTH, HALF_HEIGHT)
        if realsense_available:
            r_frame = get_realsense_annotated_frame(pipeline, align, K, D, T_tcp_cam)
        else:
            r_frame = None

        ui_frame = create_ui_frame(r_frame, d_frame)
        cv2.imshow(WINDOW_NAME, ui_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dobot_stream.stop()
    cv2.destroyAllWindows()
    
main()