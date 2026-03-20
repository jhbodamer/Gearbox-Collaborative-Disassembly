# Demo showcasing different areas of this semester
# Mostly skeleton code for now 
from loop_pickup_with_adaptive_gripper import dbg, get_rtde_iface, get_rtde_control_iface, trigger_scanner, get_tcp_pose6, ur_pose6_to_T, pose2d_to_T, drawAxis, getOrientation, intersect_Z0
from gui_ai import drawAxis, getOrientation, add_log, CameraStream, draw_text_centered, create_ui_frame, get_realsense_annotated_frame, T_TCP_CAM_PATH, np, rs, time, cv2, collections, threading,atan2,cos,sin,sqrt,pi,radians, Path


def main():
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
    try:
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
                raise Exception("q key was hit")

    except Exception as e:
        dobot_stream.stop()
        cv2.destroyAllWindows()
        print(f"An error occurred: {type(e).__name__} – {e}")

main()
