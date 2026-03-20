import pyrealsense2 as rs
import numpy as np
import cv2

# --- Configure depth stream ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Colorize depth for visualization (optional but helpful)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.5),
            cv2.COLORMAP_JET
        )

        # Display
        cv2.imshow("Depth Feed", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
