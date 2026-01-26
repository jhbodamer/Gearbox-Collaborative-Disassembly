#!/usr/bin/env python3
import numpy as np
import cv2
import time
from math import atan2, cos, sin, sqrt, pi, radians
from pathlib import Path
import pyrealsense2 as rs
import GoSdk_MsgHandler
import os
import random
import traceback
import ctypes
from ctypes import byref
from scipy.spatial.transform import Rotation as R
from Gocator import (
    GoSdk, kApi, RecieveData, get_measurement_decision,
    kObject_Destroy, kIpAddress, GoDataSet, GoDataMsg, kNULL
)
from epick_gripper2 import (start_suction, stop_suction)
from work_detector_vision import WorkDetector

# ---------- USER CONFIG ----------
UR_IP = "192.168.1.5"
T_TCP_CAM_PATH = Path("T_tcp_cam.npy")
STREAM_W, STREAM_H, FPS = 640, 480, 30
WORK_DETECTION_CAMERA_ID = 6  # Camera ID for work detection
# ----------------------------------

# --- Scanner CONFIG ---
SCANNER_IP = b"192.168.1.10"
RECEIVE_TIMEOUT = 10000

VERBOSE = True
AUTO_FLIP_RAY = True
FLIP_RAY = False

# Observing Pose
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
    
    angle = atan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

def getOrientation(pts, img):
    """
    Get orientation using PCA analysis on contour points.
    This finds the actual principal axes of the object for accurate orientation.
    Based on the OpenCV example code.
    """
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
    
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    
    label = f"Rotation: {angle:.3f} rad ({np.degrees(angle):.1f} deg)"
    
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    text_x = cntr[0] - text_width // 2
    text_y = cntr[1] - 40
    
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    return angle, cntr

def intersect_Z0(u, v, K, D, T_base_tcp, T_tcp_cam, img=None, orientation_debug=False, use_enhanced_method=True):
    """
    Intersect the camera ray (through pixel u,v) with BASE-plane Z=0.
    Returns (X,Y,theta) in meters and radians, plus a diagnostics dict.
    """
    global FLIP_RAY

    T_base_cam = T_base_tcp @ T_tcp_cam
    R_bc = T_base_cam[:3,:3]
    cam_o = T_base_cam[:3, 3]

    pts  = np.array([[[float(u), float(v)]]], dtype=np.float64)
    norm = cv2.undistortPoints(pts, K, D)
    x_n, y_n = norm[0,0]
    ray_cam = np.array([x_n, y_n, 1.0], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)

    tried_flip = False
    for _ in range(2):
        ray_b = R_bc @ ( -ray_cam if FLIP_RAY else ray_cam )
        denom = ray_b[2]
        s = None
        if abs(denom) > 1e-12:
            s = (0.0 - cam_o[2]) / denom

        if s is not None and s > 0:
            P = cam_o + s*ray_b
            
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
    """
    Main function that automatically detects objects and finds their orientation using minAreaRect.
    Integrated with work detection for safety.
    """
    
    global FLIP_RAY
    
    object_detected = False
    
    # Load camera calibration
    if not T_TCP_CAM_PATH.exists():
        print(f"[ERROR] {T_TCP_CAM_PATH} not found. Run calibration first.")
        return
    
    T_tcp_cam = np.load(T_TCP_CAM_PATH)
    print(f"[OK] Loaded {T_TCP_CAM_PATH}")
    
    # Connect to robot
    rtde = get_rtde_iface(UR_IP)
    rtde_control = get_rtde_control_iface(UR_IP)
    
    # Initialize Work Detector
    print("[INFO] Initializing Work Detector...")
    work_detector = WorkDetector(idle_threshold=2.0)
    
    # Open work detection camera
    work_cap = cv2.VideoCapture(WORK_DETECTION_CAMERA_ID)
    if not work_cap.isOpened():
        print(f"[ERROR] Could not open work detection camera (ID: {WORK_DETECTION_CAMERA_ID})")
        return
    print(f"[OK] Work detection camera opened (ID: {WORK_DETECTION_CAMERA_ID})")
    
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
    
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)
    
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=float)

    D = np.array(intrinsics.coeffs[:5], dtype=float)
    
    print(f"[INFO] Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"[INFO] Distortion: {D} {'[OK]' if np.allclose(D, 0) else '[WARN]'}")
    
    print("\n=== INTEGRATED LOOP PICKUP WITH WORK DETECTION ===")
    print("The system will:")
    print("  1. Monitor human work activity")
    print("  2. Detect objects when human is IDLE")
    print("  3. Execute pickup sequence only when safe")
    print("Keys: q=quit, r=reload T_tcp_cam.npy, p=print TCP pose, d=toggle debug")
    print("=====================================================\n")
    
    try:
        while True:
            # Read work detection camera
            work_ret, work_frame = work_cap.read()
            if work_ret:
                work_frame = cv2.flip(work_frame, 1)  # Mirror view
                
                # Check if person is working
                is_working, confidence, debug_info = work_detector.is_person_working(work_frame)
                
                # Draw status on work detection frame
                work_frame = work_detector.draw_status(work_frame, is_working, confidence, debug_info)
                
                # Display work detection window
                cv2.imshow('Work Detection', work_frame)
            
            # Get current work state
            current_state = work_detector.get_state()
            
            # Read object detection frames
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            
            if not color:
                continue
                
            img = np.asanyarray(color.get_data())
            ui = img.copy()
            
            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Convert image to binary
            _, bw = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
            
            # Find all the contours in the thresholded image
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
                    print(f"[WARN] Failed to get orientation for contour {i}: {e}")
                    continue
            
            # Process detected objects to get their poses
            if detected_objects:
                object_detected = True
                for i, (contour, area, angle, center_point) in enumerate(detected_objects):
                    pose6 = get_tcp_pose6(rtde)
                    T_base_tcp = ur_pose6_to_T(pose6)
                    
                    center_x, center_y = center_point
                    
                    Xb, Yb, _, info = intersect_Z0(center_x, center_y, K, D, T_base_tcp, T_tcp_cam, ui, True, True)
                    
                    if Xb is not None:
                        current_pose = get_tcp_pose6(rtde)
                        current_x, current_y, current_z = current_pose[0:3]
                        current_rx, current_ry, current_rz = current_pose[3:6]
                        
                        aligned_rz = angle
                        aligned_pose = [Xb, Yb, current_z, current_rx, current_ry, aligned_rz]
                        
                        coord_text = f"({Xb*1000:.0f},{Yb*1000:.1f})mm"
                        cv2.putText(ui, coord_text, (center_x + 10, center_y + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        orientation_text = f"TCP Rz: {np.degrees(aligned_rz):.1f}°"
                        cv2.putText(ui, orientation_text, (center_x + 10, center_y + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Automated movement sequence logic - ONLY when IDLE
            if object_detected and rtde_control is not None and current_state == "IDLE":
                current_time = time.time()
                
                if not hasattr(main, 'sequence_start_time'):
                    main.sequence_start_time = current_time
                    print(f"\n{'='*60}")
                    print(f"🤖 ROBOT SEQUENCE STARTING - Human is IDLE")
                    print(f"{'='*60}")
                    print(f"[INFO] Waiting 2 seconds for observation...")
                
                if main.sequence_start_time is not None:
                    time_elapsed = current_time - main.sequence_start_time
                    
                    if not hasattr(main, 'step10_complete') or main.step9_complete is None:
                        if time_elapsed >= 2.0:
                            # STEP 1: Execute moveL to target position with iterative refinement
                            if not hasattr(main, 'step1_complete'):
                                if not hasattr(main, 'step1_iteration'):
                                    main.step1_iteration = 1
                                    main.target_Xb = None
                                    main.target_Yb = None
                                    main.step1_moving = False
                                    main.step1_move_start_time = None
                                    print(f"\n=== STEP 1: STARTING ITERATIVE POSITIONING ===")
                                
                                current_pose = get_tcp_pose6(rtde)
                                current_X = current_pose[0]
                                current_Y = current_pose[1]
                                
                                if detected_objects:
                                    contour, area, angle, center_point = detected_objects[0]
                                    center_x, center_y = center_point
                                    
                                    T_base_tcp = ur_pose6_to_T(current_pose)
                                    Xb, Yb, _, info = intersect_Z0(center_x, center_y, K, D, T_base_tcp, T_tcp_cam, ui, True, True)
                                    
                                    if Xb is not None:
                                        if main.step1_iteration == 1:
                                            main.target_Xb = Xb
                                            main.target_Yb = Yb
                                            main.object_angle = angle
                                            print(f"Target object: Center pixel ({center_x},{center_y})")
                                            print(f"Target Base XY: ({Xb*1000:.1f} mm, {Yb*1000:.1f} mm)")
                                            print(f"Object angle: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
                                        
                                        distance_X = abs(current_X - main.target_Xb)
                                        distance_Y = abs(current_Y - main.target_Yb)
                                        total_distance = ((distance_X**2 + distance_Y**2)**0.5) * 1000
                                        
                                        if total_distance <= 1.0:
                                            print(f"Position accuracy achieved! Total distance: {total_distance:.1f}mm ≤ 5mm")
                                            print(f"Step 1 completed after {main.step1_iteration} iterations")
                                            main.step1_complete = True
                                            print("[INFO] Waiting 2 seconds after positioning...")
                                        else:
                                            if not main.step1_moving:
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
                                                time_since_move = time.time() - main.step1_move_start_time
                                                if time_since_move >= 1.0:
                                                    new_pose = get_tcp_pose6(rtde)
                                                    new_X = new_pose[0]
                                                    new_Y = new_pose[1]
                                                    
                                                    position_change = abs(new_X - current_X) + abs(new_Y - current_Y)
                                                    if position_change < 0.001:
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
                            if not hasattr(main, 'step2_delay_start'):
                                main.step2_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step2_delay_start
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 2: CENTERING JOINT 5 ABOVE OBJECT ===")
                                try:
                                    contour, area, angle, center_point = detected_objects[0]
                                    main.object_angle = angle
                                    
                                    frames = pipeline.wait_for_frames()
                                    aligned_frames = align.process(frames)
                                    depth_frame = aligned_frames.get_depth_frame()
                                    
                                    depth_image = np.asanyarray(depth_frame.get_data())
                                    
                                    depth_scale = depth_frame.get_units()
                                    h, w = depth_image.shape
                                    center_x, center_y = w // 2, h // 2 
                                    main.depth_value = depth_image[center_y, center_x] * depth_scale
                                    print("Depth value in meters: ", main.depth_value)
                                    
                                    axis_angle = np.array(current_pose[3:])
                                    r = R.from_rotvec(axis_angle)
                                    rotation_matrix = r.as_matrix()
                                    euler = r.as_euler('xyz', degrees=True)
                                    
                                    magnitude_of_offset = 0.05
                                    angle_of_offset = 3*pi/4
                                    offset_x = magnitude_of_offset*cos(radians(euler[2])+angle_of_offset)
                                    offset_y = magnitude_of_offset*sin(radians(euler[2])+angle_of_offset)
                                    current_pose = get_tcp_pose6(rtde)
                                    target_pose_step2 = [current_pose[0]+offset_x, current_pose[1]+offset_y, 0.330, current_pose[3], current_pose[4], current_pose[5]]
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
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 3: EXECUTING MOVEJ ===")
                                
                                if detected_objects:
                                    contour, area, angle, center_point = detected_objects[0]
                                    fresh_angle = angle
                                    print(f"[INFO] Current object detection: angle = {fresh_angle:.3f} rad ({np.degrees(fresh_angle):.1f}°)")
                                else:
                                    fresh_angle = main.object_angle
                                    print(f"[WARN] No objects currently detected, using stored angle: {np.degrees(fresh_angle):.1f}°")
                                
                                try:
                                    current_joints = rtde.getActualQ()
                                    print(f"Current joints: [{np.degrees(current_joints[0]):.1f}, {np.degrees(current_joints[1]):.1f}, {np.degrees(current_joints[2]):.1f}, {np.degrees(current_joints[3]):.1f}, {np.degrees(current_joints[4]):.1f}, {np.degrees(current_joints[5]):.1f}] deg")
                                    
                                    target_joints = [
                                        current_joints[0],
                                        current_joints[1],
                                        current_joints[2],
                                        current_joints[3],
                                        -np.pi/2,
                                        current_joints[5] + fresh_angle
                                    ]
                                    
                                    print(f"Target joints: [{np.degrees(target_joints[0]):.1f}, {np.degrees(target_joints[1]):.1f}, {np.degrees(target_joints[2]):.1f}, {np.degrees(target_joints[3]):.1f}, {np.degrees(target_joints[4]):.1f}, {np.degrees(target_joints[5]):.1f}] deg")
                                    print(f"Joint 5 set to -90°, Joint 6 adjusted by {np.degrees(fresh_angle):.1f}°")
                                    
                                    rtde_control.moveJ(target_joints, 1, 1)
                                    print("[OK] MoveJ command sent successfully")
                                    main.step3_complete = True
                                    print("[INFO] MoveJ complete. Waiting 2 seconds to observe new position...")
                                except Exception as e:
                                    print(f"[ERROR] MoveJ failed: {e}")
                                    main.sequence_start_time = None
                        
                        # STEP 4: Execute moveL to lower height immediately after moveJ completes
                        if hasattr(main, 'step3_complete') and not hasattr(main, 'step4_complete'):
                            if not hasattr(main, 'step4_delay_start'):
                                main.step4_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step4_delay_start
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 4: EXECUTING FINAL MOVEL (LOWER HEIGHT) ===")
                                
                                try:
                                    offset_x = 0.0
                                    offset_y = 0.0
                                    current_pose = get_tcp_pose6(rtde)
                                    target_pose_step4 = [current_pose[0]+offset_x, current_pose[1]+offset_y, 0.260, current_pose[3], current_pose[4], current_pose[5]]
                                    
                                    print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                    print(f"Target pose (Z=260mm): [{target_pose_step4[0]:.3f}, {target_pose_step4[1]:.3f}, {target_pose_step4[2]:.3f}, {target_pose_step4[3]:.3f}, {target_pose_step4[4]:.3f}, {target_pose_step4[5]:.3f}]")
                                    
                                    rtde_control.moveL(target_pose_step4, 0.6, 0.6)
                                    print("[OK] Final MoveL command sent successfully")
                                    time.sleep(1)
                                    current_pose_post_scan = get_tcp_pose6(rtde)
                                    main.step4_complete = True
                                    
                                except Exception as e:
                                    print(f"[ERROR] Final MoveL failed: {e}")
                                    main.sequence_start_time = None
                                
                        # STEP 5: Execute offset to account for gripper
                        if hasattr(main, 'step4_complete') and not hasattr(main, 'step5_complete'):
                            if not hasattr(main, 'step5_delay_start'):
                                main.step5_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step5_delay_start
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 5: EXECUTING OFFSET FOR GRIPPER ===")
                                
                                try:
                                    axis_angle = np.array(current_pose[3:])
                                    r = R.from_rotvec(axis_angle)
                                    rotation_matrix = r.as_matrix()
                                    euler = r.as_euler('xyz', degrees=True)
                                    
                                    magnitude_of_offset = -0.076
                                    angle_of_offset = pi/2
                                    offset_x = magnitude_of_offset*cos(radians(euler[2])+angle_of_offset)
                                    offset_y = magnitude_of_offset*sin(radians(euler[2])+angle_of_offset)
                                    current_pose = get_tcp_pose6(rtde)
                                    target_pose_step5 = [current_pose[0]+offset_x, current_pose[1]+offset_y, 0.260, current_pose[3], current_pose[4], current_pose[5]]
                                    print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                    print(f"Target pose (With offset)(z=260mm): [{target_pose_step5[0]:.3f}, {target_pose_step5[1]:.3f}, {target_pose_step5[2]:.3f}, {target_pose_step5[3]:.3f}, {target_pose_step5[4]:.3f}, {target_pose_step5[5]:.3f}]")
                                    rtde_control.moveL(target_pose_step5, 0.6, 0.6)
                                    print("[OK] Offset move command sent successfully")
                                    time.sleep(1)
                                    
                                    main.step5_complete = True
                                except Exception as e:
                                    print(f"[ERROR] Offset Move Failed: {e}")
                                    main.sequence_start_time = None     
                                
                        # STEP 6: Move down into pick pose
                        if hasattr(main, 'step5_complete') and not hasattr(main, 'step6_complete'):
                            if not hasattr(main, 'step6_delay_start'):
                                main.step6_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step6_delay_start
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 6: MOVING TO PICK POSE ===")
                                
                                try:
                                    current_pose = get_tcp_pose6(rtde)
                                    z_move = -main.depth_value + 0.055 + 0.07
                                    target_pose_step6 = [current_pose[0], current_pose[1], current_pose[2]+z_move, current_pose[3], current_pose[4], current_pose[5]]
                                    print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                    print(f"Target pose (z=112mm): [{target_pose_step6[0]:.3f}, {target_pose_step6[1]:.3f}, {target_pose_step6[2]:.3f}, {target_pose_step6[3]:.3f}, {target_pose_step6[4]:.3f}, {target_pose_step6[5]:.3f}]")
                                    rtde_control.moveL(target_pose_step6, 0.6, 0.6)
                                    print("[OK] Pick move command sent successfully")
                                    time.sleep(1)
                                    
                                    main.step6_complete = True
                                    
                                except Exception as e:
                                    print(f"[ERROR] Offset Move Failed: {e}")
                                    main.sequence_start_time = None    
                                
                        # STEP 7: Activate Suction
                        if hasattr(main, 'step6_complete') and not hasattr(main, 'step7_complete'):
                            if not hasattr(main, 'step7_delay_start'):
                                main.step7_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step7_delay_start
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 7: ACTIVATING GRIPPER ===")   
                                
                                try:
                                    start_suction()
                                    time.sleep(1)
                                    main.step7_complete = True
                                    
                                except Exception as e:
                                    print(f"[ERROR] Suction Start Failed: {e}")
                                    main.sequence_start_time = None
                                
                        # STEP 8: Move to human disassembly pose
                        if hasattr(main, 'step7_complete') and not hasattr(main, 'step8_complete'):
                            if not hasattr(main, 'step8_delay_start'):
                                main.step8_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step8_delay_start
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 8: MOVING TO HUMAN DISASSEMBLY POSE ===")   
                                
                                try:
                                    current_pose = get_tcp_pose6(rtde)
                                    target_pose_step8 = [current_pose[0], current_pose[1], 0.260, current_pose[3], current_pose[4], current_pose[5]]
                                    print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                    print(f"Target pose: [{target_pose_step8[0]:.3f}, {target_pose_step8[1]:.3f}, {target_pose_step8[2]:.3f}, {target_pose_step8[3]:.3f}, {target_pose_step8[4]:.3f}, {target_pose_step8[5]:.3f}]")
                                    rtde_control.moveL(target_pose_step8, 0.6, 0.6)
                                    time.sleep(2)
                                    main.step8_complete = True
                                    
                                except Exception as e:
                                    print(f"[ERROR] Human Pose Move Failed: {e}")
                                    main.sequence_start_time = None
                                
                        # STEP 9: Put object down
                        if hasattr(main, 'step8_complete') and not hasattr(main, 'step9_complete'):
                            if not hasattr(main, 'step9_delay_start'):
                                main.step9_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step9_delay_start
                            
                            if time_since_delay >= 0.5:
                                print(f"\n=== STEP 9: PUTTING OBJECT DOWN ===")   
                                
                                try:
                                    randomx = random.uniform(0.32, 0.05)
                                    randomy = random.uniform(-0.61, -0.40)
                                    randomj = random.uniform(-pi/2, pi/2)
                                    
                                    current_pose = get_tcp_pose6(rtde)
                                    z_move = -main.depth_value + 0.065 + 0.07
                                    target_pose_step9 = [randomx, randomy, current_pose[2], current_pose[3], current_pose[4], current_pose[5]]
                                    print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                    print(f"Target pose (z=112mm): [{target_pose_step9[0]:.3f}, {target_pose_step9[1]:.3f}, {target_pose_step9[2]:.3f}, {target_pose_step9[3]:.3f}, {target_pose_step9[4]:.3f}, {target_pose_step9[5]:.3f}]")
                                    rtde_control.moveL(target_pose_step9, 0.6, 0.6)
                                    
                                    current_joints = rtde.getActualQ()
                                    print(f"Current joints: [{np.degrees(current_joints[0]):.1f}, {np.degrees(current_joints[1]):.1f}, {np.degrees(current_joints[2]):.1f}, {np.degrees(current_joints[3]):.1f}, {np.degrees(current_joints[4]):.1f}, {np.degrees(current_joints[5]):.1f}] deg")
                                    
                                    target_joints = [
                                        current_joints[0],
                                        current_joints[1],
                                        current_joints[2],
                                        current_joints[3],
                                        current_joints[4],
                                        current_joints[5] + randomj
                                    ]
                                    print(f"Target joints: [{np.degrees(target_joints[0]):.1f}, {np.degrees(target_joints[1]):.1f}, {np.degrees(target_joints[2]):.1f}, {np.degrees(target_joints[3]):.1f}, {np.degrees(target_joints[4]):.1f}, {np.degrees(target_joints[5]):.1f}] deg")
                                    print(f"Joint 6 set to {np.degrees(target_joints[5]):.1f}°")
                                    
                                    rtde_control.moveJ(target_joints, 1, 1)
                                    print("[OK] MoveJ command sent successfully")
                                    
                                    current_pose = get_tcp_pose6(rtde)
                                    z_move = -main.depth_value + 0.065 + 0.07
                                    target_pose_step9 = [current_pose[0], current_pose[1], current_pose[2]+z_move, current_pose[3], current_pose[4], current_pose[5]] 
                                    print(f"Current TCP: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}]")
                                    print(f"Target pose {target_pose_step9[2]:.3f}: [{target_pose_step9[0]:.3f}, {target_pose_step9[1]:.3f}, {target_pose_step9[2]:.3f}, {target_pose_step9[3]:.3f}, {target_pose_step9[4]:.3f}, {target_pose_step9[5]:.3f}]")
                                    rtde_control.moveL(target_pose_step9, 0.3, 0.3)
                                    time.sleep(1)
                                    stop_suction()
                                    main.step9_complete = True
                                    
                                except Exception as e:
                                    print(f"[ERROR] PLACE MANEUVER FAILED: {e}")
                                    main.sequence_start_time = None
                            
                        # STEP 10: Reset
                        if hasattr(main, 'step9_complete') and not hasattr(main, 'step10_complete'):
                            if not hasattr(main, 'step10_delay_start'):
                                main.step10_delay_start = time.time()
                            
                            time_since_delay = time.time() - main.step10_delay_start
                            
                            if time_since_delay >= 1.0:
                                print(f"\n=== STEP 10: RETURN TO OBSERVATION POSE ===")   

                                try:   
                                    print("[INFO] Moving to OBSERVE_POSE before starting loop...")
                                    rtde_control.moveL(OBSERVE_POSE, 0.5, 0.5)
                                    
                                    # Reset sequence for next run
                                    object_detected = False
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
                                    del main.step2_delay_start
                                    del main.step3_delay_start
                                    del main.step4_delay_start
                                    del main.step5_delay_start
                                    del main.step6_delay_start
                                    del main.step7_delay_start
                                    del main.step8_delay_start
                                    del main.step9_delay_start
                                    del main.step10_delay_start
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
                                        time_since_move = time.time() - main.step1_move_start_time if main.step1_move_start_time else 0
                                        status_text = f"AUTO: Iteration {main.step1_iteration} - Moving... ({time_since_move:.1f}s)"
                                    else:
                                        status_text = f"AUTO: Iteration {main.step1_iteration} - Positioning..."
                                else:
                                    status_text = f"AUTO: Observing object... ({time_elapsed:.1f}s)"
                            elif not hasattr(main, 'step2_complete'):
                                if hasattr(main, 'step2_delay_start'):
                                    time_since_delay = current_time - main.step2_delay_start
                                    status_text = f"AUTO: Preparing for MoveJ with Offset ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step3_complete'):
                                if hasattr(main, 'step3_delay_start'):
                                    time_since_delay = current_time - main.step3_delay_start
                                    status_text = f"Moving Joint 6 to align rotation ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step4_complete'):
                                if hasattr(main, 'step4_delay_start'):
                                    time_since_delay = current_time - main.step4_delay_start
                                    status_text = f"Moving down to object ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step5_complete'):
                                if hasattr(main, 'step5_delay_start'):
                                    time_since_delay = current_time - main.step5_delay_start
                                    status_text = f"Executing offset for gripper ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step6_complete'):
                                if hasattr(main, 'step6_delay_start'):
                                    time_since_delay = current_time - main.step6_delay_start
                                    status_text = f"Moving down to object surface ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step7_complete'):
                                if hasattr(main, 'step7_delay_start'):
                                    time_since_delay = current_time - main.step7_delay_start
                                    status_text = f"Starting Suction ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step8_complete'):
                                if hasattr(main, 'step8_delay_start'):
                                    time_since_delay = current_time - main.step8_delay_start
                                    status_text = f"Picking up object ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step9_complete'):
                                if hasattr(main, 'step9_delay_start'):
                                    time_since_delay = current_time - main.step9_delay_start
                                    status_text = f"Placing object back down ({time_since_delay:.1f}s)"
                            elif not hasattr(main, 'step10_complete'):
                                if hasattr(main, 'step10_delay_start'):
                                    time_since_delay = current_time - main.step10_delay_start
                                    status_text = f"Resetting ({time_since_delay:.1f}s)"        
                            else:
                                status_text = "AUTO: Sequence completed!"
                            
                            cv2.putText(ui, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            elif current_state == "WORKING":
                # Pause message if work detected during sequence
                if hasattr(main, 'sequence_start_time'):
                    status_text = "PAUSED: Human Working - Waiting..."
                    cv2.putText(ui, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display work status on object detection window
            work_status = f"Human: {current_state}"
            status_color = (0, 255, 0) if current_state == "IDLE" else (0, 0, 255)
            cv2.putText(ui, work_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            status_text = f"Objects: {len(detected_objects)}   FLIP_RAY={FLIP_RAY}"
            cv2.putText(ui, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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

    finally:
        pipeline.stop()
        work_cap.release()
        stop_suction()
        cv2.destroyAllWindows()
        work_detector.hands.close()
        work_detector.pose.close()
        print("Stopped all systems.")

if __name__ == "__main__":
    main()
