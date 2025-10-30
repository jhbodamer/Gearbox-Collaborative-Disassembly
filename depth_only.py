import pyrealsense2 as rs
import numpy as np
import cv2

# --------------------------------------------------
# Configure RealSense streams
# --------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.color)

print("Press 'q' to quit.")

try:
    while True:
        # Wait for frames and align
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get depth scale and center pixel
        depth_scale = depth_frame.get_units()  # meters per depth unit
        h, w = depth_image.shape
        center_x, center_y = w // 2, h // 2

        # Read depth value at the center
        depth_value = depth_image[center_y, center_x] * depth_scale

        # Print depth value to console
        print(f"Depth at center: {depth_value*1000:.3f} mm")

        # --------------------------------------------------
        # Display live feed with depth overlay
        # --------------------------------------------------
        display_image = color_image.copy()

        # Draw center marker
        cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Put text with depth info
        cv2.putText(display_image, f"Depth: {depth_value*1000:.3f} mm",
                    (center_x - 100, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show color feed
        cv2.imshow("RealSense Depth Feed", display_image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
