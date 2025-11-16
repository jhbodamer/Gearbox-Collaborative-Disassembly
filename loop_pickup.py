#!/usr/bin/env python3
import numpy as np
import cv2
import time
from math import atan2, cos, sin, sqrt, pi, radians
from pathlib import Path
import pyrealsense2 as rs
import GoSdk_MsgHandler
import os
import traceback
import ctypes
from ctypes import byref
from scipy.spatial.transform import Rotation as R
from Gocator import (
    GoSdk, kApi, RecieveData, get_measurement_decision,
    kObject_Destroy, kIpAddress, GoDataSet, GoDataMsg, kNULL
)
from epick_gripper2 import (start_suction, stop_suction)
# from activate_gripper import pulse_gripper
# ---------- USER CONFIG ----------
UR_IP = "192.168.1.5"
T_TCP_CAM_PATH = Path("T_tcp_cam.npy")
STREAM_W, STREAM_H, FPS = 640, 480, 30
# ----------------------------------

# --- Scanner CONFIG ---
SCANNER_IP = b"192.168.1.10"
RECEIVE_TIMEOUT = 10000

VERBOSE = True         # toggle at runtime with 'd'
AUTO_FLIP_RAY = True   # allow one-time auto fix if ray points the wrong way
FLIP_RAY = False       # will be set True automatically if needed

# Old Observing pose (From original pixel_to_pose)
#OBSERVE_POSE = [0.155, -0.474, 0.201, 
#-1.841, 1.873, -0.675]

# New Observing Pose (For better vantage point)
OBSERVE_POSE = [0.200230894137112, -0.509647957735381, 0.51933226290406, 3.14159, 0.0, 0.0]

def dbg(*args):
    if VERBOSE:
        print(*args)

# ---------- UR helpers ----------
def get_rtde_iface(ip):
    try:
        from rtde_receive import RTDEReceiveInterface
        r = RTDEReceiveInterface(ip)
        _ = r.getActualTCPPose()
        print(f"[OK] RTDE connected to {ip}")
        return r
    except Exception as e:
        print(f"[WARN] RTDE not available: {e}")
        return None

def get_rtde_control_iface(ip):
    try:
        from rtde_control import RTDEControlInterface
        c = RTDEControlInterface(ip)
        print(f"[OK] RTDE Control connected to {ip}")
        return c
    except Exception as e:
        print(f"[WARN] RTDE Control not available: {e}")
        return None
    
def trigger_scanner():
    api = ctypes.c_void_p()
    system = ctypes.c_void_p()
    sensor = ctypes.c_void_p()
    dataset = GoDataSet()
    dataObj = GoDataMsg()

    try:
        GoSdk.GoSdk_Construct(byref(api))
        GoSdk.GoSystem_Construct(byref(system), None)

        ip_addr = kIpAddress()
        kApi.kIpAddress_Parse(byref(ip_addr), SCANNER_IP)
        GoSdk.GoSystem_FindSensorByIpAddress(system, byref(ip_addr), byref(sensor))
        GoSdk.GoSensor_Connect(sensor)
        GoSdk.GoSystem_EnableData(system, True)

        mgr = GoSdk_MsgHandler.MsgManager(GoSdk, system, dataset)
        mgr.SetDataHandler(RECEIVE_TIMEOUT, RecieveData)

        print("[*] Scanning object with Gocator...")
        GoSdk.GoSensor_Stop(sensor)
        GoSdk.GoSensor_Snapshot(sensor)

        print("[*] Waiting for scan data...")
        time.sleep(2.5)
        mgr.SetDataHandler(RECEIVE_TIMEOUT, kNULL)
        mgr.stop()

        decision = get_measurement_decision()

    except Exception as e:
        print(f"[ERROR] Scanner error: {e}")
        traceback.print_exc()
        decision = -1
    return decision

def get_tcp_pose6(rtde_iface):
    if rtde_iface is None:
        # fallback for testing
        return [0.0, 0.0, 0.6, 0.0, 0.0, 0.0]
    return rtde_iface.getActualTCPPose()

def ur_pose6_to_T(p6):
    x, y, z, rx, ry, rz = p6
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=float))
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = [x, y, z]
    return T

def pose2d_to_T(x, y, theta):
    """
    Convert 2D pose (x, y, theta) to 4x4 transformation matrix.
    Returns T where T[:2, 3] = [x, y] and T[:2, :2] is the 2D rotation matrix.
    """
    T = np.eye(4, dtype=float)
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    T[0, 3] = x
    T[1, 3] = y
    return T


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
    
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
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

# ---------- Core: pixel -> Z=0 ----------
def intersect_Z0(u, v, K, D, T_base_tcp, T_tcp_cam, img=None, orientation_debug=False, use_enhanced_method=True):
    """
    Intersect the camera ray (through pixel u,v) with BASE-plane Z=0.
    Returns (X,Y,theta) in meters and radians, plus a diagnostics dict.
    """
    global FLIP_RAY

    # base->cam
    T_base_cam = T_base_tcp @ T_tcp_cam
    R_bc = T_base_cam[:3,:3]
    cam_o = T_base_cam[:3, 3]           # camera origin in BASE

    # normalize pixel to a camera-frame direction
    pts  = np.array([[[float(u), float(v)]]], dtype=np.float64)
    norm = cv2.undistortPoints(pts, K, D)    # -> (x,y)
    x_n, y_n = norm[0,0]
    ray_cam = np.array([x_n, y_n, 1.0], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)

    # allow one-time flip if convention is reversed
    tried_flip = False
    for _ in range(2):
        ray_b = R_bc @ ( -ray_cam if FLIP_RAY else ray_cam )
        denom = ray_b[2]                 # Z component in BASE
        s = None
        if abs(denom) > 1e-12:
            s = (0.0 - cam_o[2]) / denom

        if s is not None and s > 0:
            P = cam_o + s*ray_b
            
            # For now, return 0 orientation since we're using PCA method
            theta = 0.0
            
            info = {
                "camZ": float(cam_o[2]),
                "rayZ": float(ray_b[2]),
                "s": float(s),
                "flipped": bool(FLIP_RAY),
                "orientation_rad": float(theta),
                "orientation_deg": float(np.degrees(theta)),
            }
            return float(P[0]), float(P[1]), float(theta), info

        # if we failed and auto-flip allowed, try flipping once
        if AUTO_FLIP_RAY and not tried_flip and not FLIP_RAY:
            tried_flip = True
            FLIP_RAY = True
            print("[INFO] Auto-corrected camera forward axis (flipped ray). Retrying once...")
            continue
        break

    info = {
        "camZ": float(cam_o[2]),
        "rayZ": float((R_bc @ ( -ray_cam if FLIP_RAY else ray_cam ))[2]),
        "s": None,
        "flipped": bool(FLIP_RAY),
        "orientation_rad": 0.0,
        "orientation_deg": 0.0,
    }
    return None, None, None, info

def main():
    #pulse_gripper() # make sure gripper is active
    
    """
    Main function that automatically detects objects and finds their orientation using minAreaRect.
    Based on the OpenCV example code provided by the user.
    """
    global FLIP_RAY
    
    # Load camera calibration
    if not T_TCP_CAM_PATH.exists():
        print(f"[ERROR] {T_TCP_CAM_PATH} not found. Run calibration first.")
        return
    
    T_tcp_cam = np.load(T_TCP_CAM_PATH)
    print(f"[OK] Loaded {T_TCP_CAM_PATH}")
    
    # Connect to robot
    rtde = get_rtde_iface(UR_IP)
    rtde_control = get_rtde_control_iface(UR_IP)
    
    # Move to observing position
    if rtde_control is not None:
            try:
                print("[INFO] Moving to OBSERVE_POSE before starting loop...")
                rtde_control.moveL(OBSERVE_POSE, 0.25, 0.25)
            except Exception as e:
                print(f"[WARN] Could not move to OBSERVE_POSE: {e}")

    # Setup RealSense
    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)

    
    print("[INFO] Starting RealSense stream...")
    pipeline.start(config)
    
    # Get camera intrinsics
    profile = pipeline.get_active_profile()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    
    # Extract intrinsics parameters
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)
    
    
    
    # Build camera matrix K
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=float)

    # Get distortion coefficients
    D = np.array(intrinsics.coeffs[:5], dtype=float)
    
    print(f"[INFO] Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"[INFO] Distortion: {D} {'[OK]' if np.allclose(D, 0) else '[WARN]'}")
    
    print("\n=== AUTOMATIC OBJECT DETECTION WITH PCA ORIENTATION ===")
    print("The system will automatically detect objects and find their orientation")
    print("using Principal Component Analysis (PCA) on object contours.")
    print("Keys: q=quit, r=reload T_tcp_cam.npy, p=print TCP pose, d=toggle debug")
    print("=====================================================\n")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()

            
            color = frames.get_color_frame()
            
            if not color:
                continue
                
            
    
            img = np.asanyarray(color.get_data())
            ui = img.copy()
            
            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Convert image to binary using Otsu's method (Sensitive to shadows)
            #_, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Convert image to binary but set our own threshold of 70 (Less sensitive to shadows)
            _, bw = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
            
            # Find all the contours in the thresholded image
            contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            detected_objects = []
            
            for i, c in enumerate(contours):
                # Calculate the area of each contour
                area = cv2.contourArea(c)
                #print(f"Area: {area}")
                # # Ignore contours that are too small or too large
                if area < 5000 or 100000 < area:
                    continue
            
                # Draw each contour only for visualisation purposes
                cv2.drawContours(ui, contours, i, (0, 0, 255), 2)
                
                # Find the orientation of each shape using PCA
                try:
                    angle, center_point = getOrientation(c, ui)
                    detected_objects.append((c, area, angle, center_point))
                except Exception as e:
                    print(f"[WARN] Failed to get orientation for contour {i}: {e}")
                    continue
            
            # Process detected objects to get their poses (similar to pixel_t_pose.py)
            if detected_objects:
                #print(f"\n=== PROCESSING {len(detected_objects)} DETECTED OBJECTS ===")
                for i, (contour, area, angle, center_point) in enumerate(detected_objects):
                    # Get robot pose and compute intersection using center point
                    pose6 = get_tcp_pose6(rtde)
                    T_base_tcp = ur_pose6_to_T(pose6)
                    
                    # Use center point from PCA as the "clicked point"
                    center_x, center_y = center_point
                    
                    # Get position from center point using existing logic
                    Xb, Yb, _, info = intersect_Z0(center_x, center_y, K, D, T_base_tcp, T_tcp_cam, ui, True, True)
                    
                    if Xb is not None:
                        # print(f"Object {i+1}: Center pixel ({center_x},{center_y}) -> Base XY: ({Xb*1000:.1f} mm, {Yb*1000:.1f} mm)")
                        # print(f"  Orientation: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
                        # print(f"  [camZ={info['camZ']:.3f}, rayZ={info['rayZ']:.3f}, flipped={info['flipped']}]")
                        
                        # Calculate aligned TCP pose for top-down camera
                        current_pose = get_tcp_pose6(rtde)
                        current_x, current_y, current_z = current_pose[0:3]
                        current_rx, current_ry, current_rz = current_pose[3:6]
                        
                        # For top-down camera: PCA angle directly maps to Rz (Yaw)
                        # Keep current Rx, Ry unchanged, only adjust Rz
                        aligned_rz = angle  # PCA angle in radians
                        
                        # Create aligned pose: [X, Y, Z, Rx, Ry, Rz]
                        aligned_pose = [Xb, Yb, current_z, current_rx, current_ry, aligned_rz]
                        
                        # Get current joint angles
                        current_joints = rtde.getActualQ()
                        # print(f"  Current Joints: [{np.degrees(current_joints[0]):.1f}, {np.degrees(current_joints[1]):.1f}, {np.degrees(current_joints[2]):.1f}, {np.degrees(current_joints[3]):.1f}, {np.degrees(current_joints[4]):.1f}, {np.degrees(current_joints[5]):.1f}] deg")
                        # print(f"  Current TCP: [{current_x:.3f}, {current_y:.3f}, {current_z:.3f}, {current_rx:.3f}, {current_ry:.3f}, {current_rz:.3f}]")
                        # print(f"  Aligned TCP: [{Xb:.3f}, {Yb:.3f}, {current_z:.3f}, {current_rx:.3f}, {current_ry:.3f}, {aligned_rz:.3f}]")
                        # print(f"  TCP will align with object orientation: {np.degrees(aligned_rz):.1f}°")
                        
                        # Draw coordinate info on image
                        coord_text = f"({Xb*1000:.0f},{Yb*1000:.1f})mm"
                        cv2.putText(ui, coord_text, (center_x + 10, center_y + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Draw orientation indicator on UI
                        orientation_text = f"TCP Rz: {np.degrees(aligned_rz):.1f}°"
                        cv2.putText(ui, orientation_text, (center_x + 10, center_y + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    else:
                        print(f"Object {i+1}: Center pixel ({center_x},{center_y}) -> No intersection with Z=0 plane")
                #print("================================\n")

            # Automated movement sequence logic
            if detected_objects and rtde_control is not None:
                # Start automated sequence after 2 seconds
                current_time = time.time()
                
                if not hasattr(main, 'sequence_start_time'):
                    main.sequence_start_time = current_time
                    print(f"\n=== AUTOMATED MOVEMENT SEQUENCE STARTING ===")
                    print(f"[INFO] Waiting 2 seconds for observation...")
                
                # Only proceed if sequence_start_time is not None (sequence hasn't completed)
                if main.sequence_start_time is not None:
                    time_elapsed = current_time - main.sequence_start_time
                    
                    # Only proceed if we haven't completed the sequence yet
                    if not hasattr(main, 'step8_complete') or main.step8_complete is None:
                        if time_elapsed >= 2.0:
                            # STEP 1: Execute moveL to target position with iterative refinement
                            if not hasattr(main, 'step1_complete'):
                                # Initialize step 1 iteration tracking
                                if not hasattr(main, 'step1_iteration'):
                                    main.step1_iteration = 1
                                    main.target_Xb = None
                                    main.target_Yb = None
                                    main.step1_moving = False
                                    main.step1_move_start_time = None
                                    print(f"\n=== STEP 1: STARTING ITERATIVE POSITIONING ===")
                                
                                # Get current robot pose
                                current_pose = get_tcp_pose6(rtde)
                                current_X = current_pose[0]
                                current_Y = current_pose[1]
                                
                                # Get latest object detection
                                if detected_objects:
                                    contour, area, angle, center_point = detected_objects[0]
                                    center_x, center_y = center_point
                                    
                                    # Calculate intersection point
                                    T_base_tcp = ur_pose6_to_T(current_pose)
                                    Xb, Yb, _, info = intersect_Z0(center_x, center_y, K, D, T_base_tcp, T_tcp_cam, ui, True, True)
                                    
                                    if Xb is not None:
                                        # Store target coordinates on first iteration
                                        if main.step1_iteration == 1:
                                            main.target_Xb = Xb
                                            main.target_Yb = Yb
                                            main.object_angle = angle
                                            print(f"Target object: Center pixel ({center_x},{center_y})")
                                            print(f"Target Base XY: ({Xb*1000:.1f} mm, {Yb*1000:.1f} mm)")
                                            print(f"Object angle: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
                                        
                                        # Calculate current distance from target
                                        distance_X = abs(current_X - main.target_Xb)
                                        distance_Y = abs(current_Y - main.target_Yb)
                                        total_distance = ((distance_X**2 + distance_Y**2)**0.5) * 1000  # Convert to mm
                                        
                                        # Check if we're close enough (within 5mm)
                                        if total_distance <= 1.0:
                                            print(f"Position accuracy achieved! Total distance: {total_distance:.1f}mm ≤ 5mm")
                                            print(f"Step 1 completed after {main.step1_iteration} iterations")
                                            main.step1_complete = True
                                            print("[INFO] Waiting 2 seconds after positioning...")
                                        else:
                                            # Handle movement state machine
                                            if not main.step1_moving:
                                                # Start new movement
                                                target_pose_step1 = [main.target_Xb, main.target_Yb, 0.330, 3.1415, 0.0, 0.0]
                                                print(f"\n--- Iteration {main.step1_iteration} ---")
                                                print(f"Current TCP: X={current_X*1000:.1f}mm, Y={current_Y*1000:.1f}mm")
                                                print(f"Target: X={main.target_Xb*1000:.1f}mm, Y={main.target_Yb*1000:.1f}mm")
                                                print(f"Distance: X={distance_X*1000:.1f}mm, Y={distance_Y*1000:.1f}mm, Total={total_distance:.1f}mm")
                                                print(f"Moving to: [{main.target_Xb:.3f}, {main.target_Yb:.3f}, 0.300, 3.1415, 0.0, 0.0]")
                                                
                                                try:
                                                    rtde_control.moveL(target_pose_step1, 0.3, 0.3)
                                                    print(f"[OK] MoveL iteration {main.step1_iteration} started")
                                                    main.step1_moving = True
                                                    main.step1_move_start_time = time.time()
                                                except Exception as e:
                                                    print(f"[ERROR] MoveL iteration {main.step1_iteration} failed: {e}")
                                                    main.sequence_start_time = None
                                            else:
                                                # Check if movement has settled (wait at least 1 second, then check if position is stable)
                                                time_since_move = time.time() - main.step1_move_start_time
                                                if time_since_move >= 1.0:
                                                    # Get position again to check if it's stable
                                                    new_pose = get_tcp_pose6(rtde)
                                                    new_X = new_pose[0]
                                                    new_Y = new_pose[1]
                                                    
                                                    # Check if position has changed significantly (movement completed)
                                                    position_change = abs(new_X - current_X) + abs(new_Y - current_Y)
                                                    if position_change < 0.001:  # Less than 1mm change
                                                        print(f"[OK] MoveL iteration {main.step1_iteration} settled")
                                                        main.step1_iteration += 1
                                                        main.step1_moving = False
                                                        main.step1_move_start_time = None
                                                        print(f"Ready for iteration {main.step1_iteration}...")
                                    else:
                                        print(f"[ERROR] Cannot calculate intersection for object center ({center_x}, {center_y})")
                                        main.sequence_start_time = None
                                else:
                                    print(f"[ERROR] No objects detected for iteration {main.step1_iteration}")
                                    main.sequence_start_time = None
                                    
                        # STEP 2: Execute offset to center joint 5 above object
                        if hasattr(main, 'step1_complete') and not hasattr(main, 'step2_complete'):
                            print(f"\n=== STEP 2: CENTERING JOINT 5 ABOVE OBJECT ===")
                            try:
                                # Store angle to fall back on in case it cant be taken on next step
                                contour, area, angle, center_point = detected_objects[0]
                                main.object_angle = angle
                                
                                # Before doing offset get depth reading for later
                                frames = pipeline.wait_for_frames()
                                aligned_frames = align.process(frames)
                                depth_frame = aligned_frames.get_depth_frame()
                                
                                # Convert depth frame to numpy array
                                depth_image = np.asanyarray(depth_frame.get_data())
                                
                                # Get depth scale and center pixel
                                depth_scale = depth_frame.get_units()  # meters per depth unit
                                h, w = depth_image.shape
                                center_x, center_y = w // 2, h // 2 
                                 # Read depth value at the center
                                main.depth_value = depth_image[center_y, center_x] * depth_scale
                                print("Depth value in meters: ", main.depth_value)
                                
                                # Convert to pitch roll yaw
                                axis_angle = np.array(current_pose[3:])
                                # Convert axis-angle to rotation matrix
                                r = R.from_rotvec(axis_angle)
                                rotation_matrix = r.as_matrix()
                                euler = r.as_euler('xyz', degrees=True)
                                
                                # Calculate offset for gripper based on tool's current orientation
                                magnitude_of_offset = 0.05
                                angle_of_offset = 3*pi/4
                                offset_x = magnitude_of_offset*cos(radians(euler[2])+angle_of_offset)
                                offset_y = magnitude_of_offset*sin(radians(euler[2])+angle_of_offset)
                                current_pose = get_tcp_pose6(rtde)
                                target_pose_step2 = [current_pose[0]+offset_x, current_pose[1]+offset_y, 0.330, current_pose[3], current_pose[4], current_pose[5]]  # Offset added
                                print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                print(f"Target pose (With offset)(z=330mm): [{target_pose_step2[0]:.3f}, {target_pose_step2[1]:.3f}, {target_pose_step2[2]:.3f}, {target_pose_step2[3]:.3f}, {target_pose_step2[4]:.3f}, {target_pose_step2[5]:.3f}]")
                                rtde_control.moveL(target_pose_step2, 0.6, 0.6)
                                print("[OK] Offset move command sent successfully")
                                main.step2_complete = True
                                main.step3_delay_start = time.time()
                                
                                
                            except Exception as e:
                                print(f"[ERROR] Offset Move Failed: {e}")
                                main.sequence_start_time = None         
                    
                        

                        # STEP 3: Wait 2 seconds after moveL completes, then execute moveJ
                        if hasattr(main, 'step2_complete') and not hasattr(main, 'step3_complete'):
                            if not hasattr(main, 'step3_delay_start'):
                                main.step3_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step3_delay_start
                            
                            if time_since_delay >= 1.0:
                                print(f"\n=== STEP 3: EXECUTING MOVEJ ===")
                                
                                # Use the current frame data for accurate angle calculation
                                if detected_objects:
                                    # Get the first (largest) detected object
                                    contour, area, angle, center_point = detected_objects[0]
                                    fresh_angle = angle
                                    print(f"[INFO] Current object detection: angle = {fresh_angle:.3f} rad ({np.degrees(fresh_angle):.1f}°)")
                                else:
                                    # Fallback to stored angle if no objects currently detected
                                    fresh_angle = main.object_angle
                                    print(f"[WARN] No objects currently detected, using stored angle: {np.degrees(fresh_angle):.1f}°")
                                
                                try:
                                    # Get current joint angles
                                    current_joints = rtde.getActualQ()
                                    print(f"Current joints: [{np.degrees(current_joints[0]):.1f}, {np.degrees(current_joints[1]):.1f}, {np.degrees(current_joints[2]):.1f}, {np.degrees(current_joints[3]):.1f}, {np.degrees(current_joints[4]):.1f}, {np.degrees(current_joints[5]):.1f}] deg")
                                    
                                    # Create target joint configuration with fresh angle data
                                    target_joints = [
                                        current_joints[0],  # current theta1
                                        current_joints[1],  # current theta2
                                        current_joints[2],  # current theta3
                                        current_joints[3],  # current theta4
                                        -np.pi/2,          # theta5 = 90 degrees
                                        current_joints[5] + fresh_angle  # current theta6 + fresh angle
                                    ]
                                    
                                    print(f"Target joints: [{np.degrees(target_joints[0]):.1f}, {np.degrees(target_joints[1]):.1f}, {np.degrees(target_joints[2]):.1f}, {np.degrees(target_joints[3]):.1f}, {np.degrees(target_joints[4]):.1f}, {np.degrees(target_joints[5]):.1f}] deg")
                                    print(f"Joint 5 set to -90°, Joint 6 adjusted by {np.degrees(fresh_angle):.1f}°")
                                    
                                    # Execute moveJ command
                                    rtde_control.moveJ(target_joints, 0.5, 0.5)
                                    print("[OK] MoveJ command sent successfully")
                                    # iterate moveL again
                                    main.step3_complete = True
                                    print("[INFO] MoveJ complete. Waiting 2 seconds to observe new position...")
                                except Exception as e:
                                    print(f"[ERROR] MoveJ failed: {e}")
                                    main.sequence_start_time = None
                        
               
                        
                        # STEP 4: Execute moveL to lower height immediately after moveJ completes
                        if hasattr(main, 'step3_complete') and not hasattr(main, 'step4_complete'):
                            print(f"\n=== STEP 4: EXECUTING FINAL MOVEL (LOWER HEIGHT) ===")
                            
                            try:
                                # Get current TCP position and only change Z height to 260mm
                                offset_x = 0.0
                                offset_y = 0.0
                                current_pose = get_tcp_pose6(rtde)
                                target_pose_step4 = [current_pose[0]+offset_x, current_pose[1]+offset_y, 0.260, current_pose[3], current_pose[4], current_pose[5]]  # Only Z changes to 260mm
                                
                                print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                print(f"Target pose (Z=260mm): [{target_pose_step4[0]:.3f}, {target_pose_step4[1]:.3f}, {target_pose_step4[2]:.3f}, {target_pose_step4[3]:.3f}, {target_pose_step4[4]:.3f}, {target_pose_step4[5]:.3f}]")
                                
                                # Execute moveL command to lower height
                                rtde_control.moveL(target_pose_step4, 0.6, 0.6)
                                print("[OK] Final MoveL command sent successfully")
                                time.sleep(1)
                                current_pose_post_scan = get_tcp_pose6(rtde)
                                # target_pose_step5 = [current_pose_post_scan[0], current_pose_post_scan[1], 0.300, current_pose_post_scan[3], current_pose_post_scan[4], current_pose_post_scan[5]]  # Only Z changes to 260mm
                                # rtde_control.moveL(target_pose_step5, 0.6, 0.6)
                                main.step4_complete = True
                            
                               
                                
                            except Exception as e:
                                print(f"[ERROR] Final MoveL failed: {e}")
                                main.sequence_start_time = None
                               
                                
                        # STEP 5:  Execute offset to account for gripper
                        if hasattr(main, 'step4_complete') and not hasattr(main, 'step5_complete'):
                            print(f"\n=== STEP 5: EXECUTING OFFSET FOR GRIPPER ===")
                            
                            
                            try:
                                
                                
                                # Convert to pitch roll yaw
                                axis_angle = np.array(current_pose[3:])
                                # Convert axis-angle to rotation matrix
                                r = R.from_rotvec(axis_angle)
                                rotation_matrix = r.as_matrix()
                                euler = r.as_euler('xyz', degrees=True)
                                
                                # Calculate offset for gripper based on tool's current orientation
                                magnitude_of_offset = -0.076
                                angle_of_offset = pi/2
                                offset_x = magnitude_of_offset*cos(radians(euler[2])+angle_of_offset)
                                offset_y = magnitude_of_offset*sin(radians(euler[2])+angle_of_offset)
                                current_pose = get_tcp_pose6(rtde)
                                target_pose_step5 = [current_pose[0]+offset_x, current_pose[1]+offset_y, 0.260, current_pose[3], current_pose[4], current_pose[5]]  # Offset added
                                print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                print(f"Target pose (With offset)(z=260mm): [{target_pose_step5[0]:.3f}, {target_pose_step5[1]:.3f}, {target_pose_step5[2]:.3f}, {target_pose_step5[3]:.3f}, {target_pose_step5[4]:.3f}, {target_pose_step5[5]:.3f}]")
                                rtde_control.moveL(target_pose_step5, 0.6, 0.6)
                                print("[OK] Offset move command sent successfully")
                                time.sleep(1)
                                
                                main.step5_complete = True
                            except Exception as e:
                                print(f"[ERROR] Offset Move Failed: {e}")
                                main.sequence_start_time = None     
                                
                                
                                
                        # STEP 6:  Move down into pick pose
                        if hasattr(main, 'step5_complete') and not hasattr(main, 'step6_complete'):
                            print(f"\n=== STEP 6: MOVING TO PICK POSE ===")
                            
                            
                            try:
                                
                                # Move to depth offset (gripper pressed against box surface)
                                current_pose = get_tcp_pose6(rtde)
                                z_move = -main.depth_value + 0.055 + 0.07 # account for difference between camera and suction height as well as difference in height since depth was measured
                                target_pose_step6 = [current_pose[0], current_pose[1], current_pose[2]+z_move, current_pose[3], current_pose[4], current_pose[5]]  # Only Z changes to 112mm
                                print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                print(f"Target pose (z=112mm): [{target_pose_step6[0]:.3f}, {target_pose_step6[1]:.3f}, {target_pose_step6[2]:.3f}, {target_pose_step6[3]:.3f}, {target_pose_step6[4]:.3f}, {target_pose_step6[5]:.3f}]")
                                rtde_control.moveL(target_pose_step6, 0.6, 0.6)
                                print("[OK] Pick move command sent successfully")
                                time.sleep(1)
                                
                                main.step6_complete = True
                                
                            except Exception as e:
                                print(f"[ERROR] Offset Move Failed: {e}")
                                main.sequence_start_time = None    
                                
                                
                        # STEP 7:  Activate Suction
                        if hasattr(main, 'step6_complete') and not hasattr(main, 'step7_complete'):
                            print(f"\n=== STEP 7: ACTIVATING GRIPPER ===")   
                            
                            try:
                                # Activate Gripper
                                start_suction()
                                time.sleep(1)
                                main.step7_complete = True
                                
                            except Exception as e:
                                print(f"[ERROR] Suction Start Failed: {e}")
                                main.sequence_start_time = None
                                
                        # STEP 8:  Move to human disassembly pose
                        if hasattr(main, 'step7_complete') and not hasattr(main, 'step8_complete'):
                            print(f"\n=== STEP 8: MOVING TO HUMAN DISASSEMBLY POSE ===")   
                            
                            try:
                                # Move to up to Z=260mm
                                current_pose = get_tcp_pose6(rtde)
                                target_pose_step8 = [current_pose[0], current_pose[1], 0.260, current_pose[3], current_pose[4], current_pose[5]]  # move directly up
                                print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                print(f"Target pose: [{target_pose_step8[0]:.3f}, {target_pose_step8[1]:.3f}, {target_pose_step8[2]:.3f}, {target_pose_step8[3]:.3f}, {target_pose_step8[4]:.3f}, {target_pose_step8[5]:.3f}]")
                                rtde_control.moveL(target_pose_step8, 0.6, 0.6)
                                time.sleep(2)
                                main.step8_complete = True
                                
                            except Exception as e:
                                print(f"[ERROR] Human Pose Move Failed: {e}")
                                main.sequence_start_time = None
                                
                                
                                
                                
                        # STEP 9:  Put object down
                        if hasattr(main, 'step8_complete') and not hasattr(main, 'step9_complete'):
                            print(f"\n=== STEP 9: PUTTING OBJECT DOWN ===")   
                            
                            try:
                                # Move to put object down
                                current_pose = get_tcp_pose6(rtde)
                                z_move = -main.depth_value + 0.065 + 0.07 # drop 1 cm higher than where picked up
                                target_pose_step9 = [current_pose[0], current_pose[1], current_pose[2]+z_move, current_pose[3], current_pose[4], current_pose[5]]  # Only Z changes to 112mm
                                print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                print(f"Target pose (z=112mm): [{target_pose_step9[0]:.3f}, {target_pose_step9[1]:.3f}, {target_pose_step9[2]:.3f}, {target_pose_step9[3]:.3f}, {target_pose_step9[4]:.3f}, {target_pose_step9[5]:.3f}]")
                                rtde_control.moveL(target_pose_step9, 0.6, 0.6)
                                time.sleep(1)
                                stop_suction()
                                main.step9_complete = True
                                
                
                            except Exception as e:
                                print(f"[ERROR] PLACE MANEUVER FAILED: {e}")
                                main.sequence_start_time = None
                        
                                
                        # STEP 10:  Reset
                        if hasattr(main, 'step9_complete') and not hasattr(main, 'step10_complete'):
                            print(f"\n=== STEP 10: RETURN TO OBSERVATION POSE ===")   


                            try:   
                                # Move to observing position
                                print("[INFO] Moving to OBSERVE_POSE before starting loop...")
                                rtde_control.moveL(OBSERVE_POSE, 0.5, 0.5)
                                       
                                        
                                # Reset sequence for next run
                                del main.sequence_start_time 
                                del main.step1_complete
                                del main.step2_complete
                                del main.step3_complete 
                                del main.step4_complete
                                del main.step5_complete
                                del main.step6_complete
                                del main.step7_complete
                                del main.step8_complete
                                del main.step9_complete
                                del main.step3_delay_start
                                del main.step1_iteration
                                del main.target_Xb
                                del main.target_Yb
                                del main.step1_moving
                                del main.step1_move_start_time
                                del main.depth_value
                            except Exception as e:
                                    print(f"[ERROR] RESET FAILED {e}")
                                    main.sequence_start_time = None
                                
                
                # Display current sequence status
                if hasattr(main, 'sequence_start_time') and main.sequence_start_time is not None:
                    current_time = time.time()
                    time_elapsed = current_time - main.sequence_start_time
                    
                    if not hasattr(main, 'step1_complete'):
                        if hasattr(main, 'step1_iteration'):
                            if hasattr(main, 'step1_moving') and main.step1_moving:
                                # Show movement status
                                time_since_move = time.time() - main.step1_move_start_time if main.step1_move_start_time else 0
                                status_text = f"AUTO: Iteration {main.step1_iteration} - Moving... ({time_since_move:.1f}s)"
                            else:
                                # Show positioning status
                                status_text = f"AUTO: Iteration {main.step1_iteration} - Positioning..."
                        else:
                            status_text = f"AUTO: Observing object... ({2.0 - time_elapsed:.1f}s left)"
                    elif not hasattr(main, 'step3_complete'):
                        if hasattr(main, 'step3_delay_start'):
                            time_since_delay = current_time - main.step3_delay_start
                            if time_since_delay < 2.0:
                                status_text = f"AUTO: MoveL complete, observing new position... ({2.0 - time_since_delay:.1f}s left)"
                            else:
                                status_text = "AUTO: MoveL complete, executing MoveJ..."
                        else:
                            status_text = "AUTO: MoveL complete, preparing for MoveJ..."
                    elif not hasattr(main, 'step4_complete'):
                        if hasattr(main, 'step4_delay_start'):
                            time_since_delay = current_time - main.step4_delay_start
                            if time_since_delay < 2.0:
                                status_text = f"AUTO: MoveJ complete, observing new position... ({2.0 - time_since_delay:.1f}s left)"
                            else:
                                status_text = "AUTO: MoveJ complete, executing final MoveL..."
                        else:
                            status_text = "AUTO: MoveJ complete, preparing for final MoveL..."
                    else:
                        status_text = "AUTO: Sequence completed!"
                    
                    cv2.putText(ui, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Add detailed iteration status if in step 1
                    if hasattr(main, 'step1_iteration') and not hasattr(main, 'step1_complete'):
                        if hasattr(main, 'target_Xb') and main.target_Xb is not None:
                            # Get current position for distance calculation
                            current_pose = get_tcp_pose6(rtde)
                            current_X = current_pose[0]
                            current_Y = current_pose[1]
                            
                            distance_X = abs(current_X - main.target_Xb)
                            distance_Y = abs(current_Y - main.target_Yb)
                            total_distance = ((distance_X**2 + distance_Y**2)**0.5) * 1000
                            
                            # Display target and current positions
                            target_text = f"Target: ({main.target_Xb*1000:.1f}, {main.target_Yb*1000:.1f}) mm"
                            current_text = f"Current: ({current_X*1000:.1f}, {current_Y*1000:.1f}) mm"
                            distance_text = f"Distance: {total_distance:.1f} mm"
                            
                            cv2.putText(ui, target_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(ui, current_text, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            cv2.putText(ui, distance_text, (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
                            # Show progress bar
                            progress_width = 200
                            progress_height = 15
                            progress_x = 10
                            progress_y = 180
                            
                            # Calculate progress (0 to 1, where 1 = 5mm or less)
                            max_distance = 50.0  # Assume max 50mm initial distance
                            progress = max(0, min(1, (max_distance - total_distance) / (max_distance - 5.0)))
                            
                            # Draw background
                            cv2.rectangle(ui, (progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height), (100, 100, 100), -1)
                            # Draw progress
                            progress_fill_width = int(progress_width * progress)
                            if progress_fill_width > 0:
                                cv2.rectangle(ui, (progress_x, progress_y), (progress_x + progress_fill_width, progress_y + progress_height), (0, 255, 0), -1)
                            
                            # Draw border
                            cv2.rectangle(ui, (progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height), (255, 255, 255), 2)
                            
                            # Progress text
                            progress_text = f"Progress: {progress*100:.0f}% ({total_distance:.1f}mm → 5mm)"
                            cv2.putText(ui, progress_text, (10, progress_y + progress_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display status
            status_text = f"Detected Objects: {len(detected_objects)}   FLIP_RAY={FLIP_RAY}"
            cv2.putText(ui, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            

            # Show the processed image
            cv2.imshow('Object Detection with PCA Orientation', ui)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                if T_TCP_CAM_PATH.exists():
                    T_tcp_cam = np.load(T_TCP_CAM_PATH)
                    print("[OK] Reloaded T_tcp_cam.npy")
            elif k == ord('p'):
                pose6 = get_tcp_pose6(rtde)
                T_base_tcp = ur_pose6_to_T(pose6)
                T_base_cam = T_base_tcp @ T_tcp_cam
                print("[INFO] TCP pose6 [x y z rx ry rz]:", np.array(pose6))
                print("[INFO] T_base_cam:\n", T_base_cam)
                z_axis = T_base_cam[:3,:3] @ np.array([0,0,1.0])
                print(f"[INFO] Camera +Z in BASE = {z_axis}  (dot with -Z: {float(np.dot(z_axis,[0,0,-1])):.3f})")
                
                # Show detected objects info
                if detected_objects:
                    print(f"\n[INFO] Detected {len(detected_objects)} objects:")
                    for i, (contour, area, angle, center_point) in enumerate(detected_objects):
                        print(f"  Object {i+1}: Area={area:.0f}, Angle={angle:.3f} rad ({np.degrees(angle):.1f}°)")
                        print(f"    Center pixel: {center_point}")
                else:
                    print("[INFO] No objects detected in current frame")
            elif k == ord('d'):
                VERBOSE = not VERBOSE
                print(f"[INFO] Debug = {VERBOSE}")
            elif k == ord('f'):
                FLIP_RAY = not FLIP_RAY
                print(f"[INFO] FLIP_RAY = {FLIP_RAY}")
            elif k == ord('h'):
                print("\n=== PCA ORIENTATION DETECTION HELP ===")
                print("The system automatically detects objects and finds their orientation")
                print("using Principal Component Analysis (PCA) on object contours.")
                print("\n=== AUTOMATIC POSE DETECTION ===")
                print("Center points from PCA analysis are automatically used as 'clicked points'")
                print("to compute Base (X,Y) coordinates where the ray hits Z=0 plane.")
                print("\n=== AUTOMATED MOVEMENT SEQUENCE ===")
                print("The system will automatically execute a movement sequence:")
                print("  1. Wait 2 seconds for observation")
                print("  2. MoveL to target position (Z=280mm)")
                print("  3. Wait 2 seconds")
                print("  4. MoveJ for rotation alignment")
                print("  5. Wait 2 seconds")
                print("  6. MoveL to lower height (Z=260mm)")
                print("================================\n")

    
    finally:
        pipeline.stop()
        stop_suction()
        cv2.destroyAllWindows()
        print("Stopped RealSense.")

if __name__ == "__main__":
    main()
