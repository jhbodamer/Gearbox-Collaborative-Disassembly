#v3pipeline
#!/usr/bin/env python3
"""
scanning_pipeline.py

Combined end-to-end pipeline for:
 1) YOLO + depth live detection with conveyor & PIR trigger
 2) UR5e alignment (arrive → sample 10 depths → offset to 10 in / 254 mm) & Gocator scan
 3) Post-scan review UI (Rescan, Realign, Accept → PDF report)
 4) After Accept → UR5e [0.6171946377970486, -0.5426800992715954, 0.3305964741811336, 2.346006884060262, 1.9780108594367762, 0.14325826103810804]ePick pick & place → exit

Notes:
- Uses a SINGLE RTDE session with a movement lock.
- Waits for robot ARRIVAL at ALIGNMENT/SCAN poses before sampling/scan.
- Saves scan PNG/PLY to ~/Code/test/Object-Detection/results (absolute paths).
- UI loads the exact saved PNG (waits briefly if needed).
"""

#import faulthandler
#faulthandler.enable()

import sys
import os
import time
import threading
import ctypes
import signal
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pyrealsense2 as rs
import degirum as dg
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, Image as PILImage
from ultralytics import YOLO

# Headless matplotlib for savDETECTIONing images during scan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── LOCAL MODULE PATHS ─────────────────────────────────────────────
HOME = Path.home()
sys.path.extend([
    str(HOME / "Code/test/pydobotplus"),
    str(HOME / "Code/test/Dobot"),
    str(HOME / "Code/test/ur5e/Python"),
])
# RTDE extensions if present
build_root = HOME / "Code/test/ur5e/ur_rtde" / "build-setuptools"
temp_dirs = list(build_root.glob("temp.*"))
if temp_dirs:
    sys.path.insert(0, str(temp_dirs[0]))
sys.path.insert(0, str(HOME / "Code/test/ur5e/ur_rtde" / "src"))
# ────────────────────────────────────────────────────────────────────
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

from DoBotArm import DoBotArm
import DobotDllType as dType

# ePick gripper (UR5e suction)
from epick_gripper2 import start_suction, stop_suction # provides connect()/start_suction()/stop_suction()/status() (assumed)
import gripper

# ─── PATH CONFIGURATION ───────────────────────────────────────────────
INSPECTION_DIR   = Path(__file__).resolve().parent
BASE             = INSPECTION_DIR.parent  # ~/Code/test/Object-Detection
MODEL_NAME       = "yolov11s"
MODEL_BASE       = BASE / "models"
ZOO_URL          = str(MODEL_BASE)     # must be string for Degirum
LABEL_FILE       = MODEL_BASE / MODEL_NAME / "labels_yolov11s.json"
STANDARD_DIM_FILE= INSPECTION_DIR / "standard_dimensions.json"
RESULTS_DIR      = BASE / "results"
DOCUMENTS_DIR    = INSPECTION_DIR / "documents"

# ─── ROBOT & SENSOR CONFIG ────────────────────────────────────────────
UR5E_IP         = "192.168.1.5"
DETECTION_POSE  = [0.6171946377970486, -0.5426800992715954, 0.3305964741811336, 2.346006884060262, 1.9780108594367762, 0.14325826103810804]
ALIGNMENT_POSE  = [0.6314107462699788, -0.38074619176710145, 0.2535261746053752, -2.294871608335058, -2.098870236207419, 0.011794935441387941]
SCAN_POSE       = [0.6086226435395277, -0.38097815812075364, 0.2959422865791701, 2.278351038309512, 2.138517359773826, 0.03197524434968253]
HOME_X, HOME_Y, HOME_Z = 200, 0, 50
PIR_PIN        = 15
CONVEYOR_SPEED = 60
UR_SPEED       = 0.15   # per user request: speed/accel 0.3
UR_ACCEL       = 0.1
SCANNER_IP     = b"192.168.1.10"
RECEIVE_TIMEOUT= 10000

# ─── UR5e ePick PnP COORDS (provided) ─────────────────────────────────
UR_PICK_POSE = [0.6985281499256294, -0.39591478900692456, 0.11397398181224824, 2.388209001361736, 2.013067155064894, 0.04936957744248028]
UR_DROP_POSE = [0.35408779470312507, -0.6277985314083585, 0.09163414487131588,-2.11324724252493, -2.2819505729846736, -0.02765755486127989]
UR_PNP_LIFT = 0.06  # 60 mm hover for approach/depart

# ─── DOBOT PICK & PLACE COORDINATES (X, Y, Z, R) ──────────────────────
DOBOT_PICK = (-128.94435119628906, 313.7967834472656, -48.55160140991211, 112.33857727050781)
DOBOT_DROP = (342.01251220703125, 140.06790161132812, 33.94829177856445, 22.271074295043945)
DOBOT_LIFT_MM = 60.0  # lift amount between pick and place

# ─── YOLO MODEL CONFIG ───────────────────────────────────────────────
MODEL_PATH     = BASE / "best.pt"  # Use best.pt from Object-Detection directory
CONF_THRESH    = 0.5  # easy to tweak
MODEL_IN_SIZE  = 640
# Optional tiny relax if it takes a while to see the object:
CONF_MIN = 0.35
CONF_RELAX_AFTER_S = 6.0

# ─── GLOBAL SYNCHRONIZATION & STATE ───────────────────────────────────
finish_event       = threading.Event()
detected_box       = [0, 0]
latest_depth_frame = None
depth_lock         = threading.Lock()
scan_basic         = None
scan_measurements  = None
scan_inspection    = None
scan_image_path    = None
standards          = None
scan_lock          = threading.Lock()  # prevent double scans
live_view_ready    = threading.Event()

# UR single-session + lock
URC = None
URR = None
URC_lock = threading.Lock()

# ─── THREADED YOLO INFERENCE ──────────────────────────────────────────
inference_lock     = threading.Lock()
latest_frame       = None
latest_results     = None
inference_ready    = threading.Event()

# ─── NEW: ePick setup/helpers (ONLY change area) ──────────────────────
GRIPPER_HOST = "192.168.1.5"   # existing constants now used
GRIPPER_PORT = 63352
GRIPPER_ID   = 9
GRIPPER_READY = False

# def epick_init():
    # """Connect to the ePick controller. Uses epick_gripper module."""
    # global GRIPPER_READY
    # try:
        # # If your module needs only host/port or a device id, adapt below:
        # #epick_gripper.connect(GRIPPER_HOST, GRIPPER_PORT, GRIPPER_ID)
GRIPPER_READY = True
        # print("[✓] ePick connected.")
    # except Exception as e:
        # print(f"[!] ePick connect failed: {e}")
        # GRIPPER_READY = False

# def epick_start(wait_seal=True, timeout=2.0, min_dwell=0.6):
    # """Turn vacuum ON; optionally wait until a seal is reported. Adds a safe dwell."""
    # try:
        # start_suction()
    # except Exception as e:
        # print(f"[!] ePick start_suction error: {e}")
        # return False
    # # If SDK has no status(), dwell for a reasonable time and continue
    # if not wait_seal:
        # time.sleep(min_dwell)
        # return True
    # sealed = False
    # t0 = time.time()
    # # First, minimal dwell to allow vacuum to build
    # time.sleep(min_dwell)
    # # Then poll the status() if available
    # while time.time() - t0 < timeout:
        # try:
            # st = epick_gripper2.status()  # expected to return dict-like with 'sealed'
            # sealed = bool(st.get('sealed', False))
        # except Exception:
            # # If no status API, assume OK after dwell
            # sealed = True
        # if sealed:
            # break
        # tLime.sleep(0.05)
    # print(f"[?] ePick sealed={sealed}")
    # return sealed

# def epick_stop():
    # """Turn vacuum OFF (ignore errors)."""
    # try:
        # stop_suction()
        # return True
    # except Exception as e:
        # print(f"[!] ePick stop_suction error: {e}")
        # return False

def tool_io(on: bool):
    """Optional: toggle UR tool digital output in case your ePick is wired to UR tool IO."""
    try:
        ur_init()
        with URC_lock:
            URC.setToolDigitalOut(0, bool(on))  # adjust channel if needed
        print(f"[?] UR tool DO0 = {on}")
    except Exception as e:
        print(f"[!] setToolDigitalOut failed: {e}")



# ─── UR INIT/MOTION ───────────────────────────────────────────────────
def ur_init():
    """Ensure a single RTDE control/receive session is available."""
    global URC, URR
    if URC is None:
        URC = RTDEControl(UR5E_IP)
    if URR is None:
        URR = RTDEReceive(UR5E_IP)

def ur_moveL(pose, speed=UR_SPEED, accel=UR_ACCEL, label=""):
    """Thread-safe moveL with single RTDE session."""
    try:
        ur_init()
        with URC_lock:
            if label:
                print(f"[*] UR moveL → {label}")
            URC.moveL(pose, speed, accel)
            return True
    except Exception as e:
        print(f"[!] UR moveL failed ({label}): {e}")
        return False

def ur_wait_until_reached(target_pose, pos_tol=0.002, rot_tol=0.05, timeout=12.0, label=""):
    """
    Wait until UR actual TCP pose is within tolerance of target_pose.
    pos_tol [m], rot_tol [rad].
    """
    try:
        ur_init()
    except Exception as e:
        print(f"[!] UR init failed while waiting ({label}): {e}")
        return False
    t0 = time.time()
    tgt_p = np.array(target_pose[:3], dtype=float)
    tgt_r = np.array(target_pose[3:], dtype=float)
    while time.time() - t0 < timeout:
        try:
            pose = URR.getActualTCPPose()  # [x,y,z,rx,ry,rz]
        except Exception as e:
            print(f"[!] getActualTCPPose error ({label}): {e}")
            time.sleep(0.1)
            continue
        p = np.array(pose[:3], dtype=float)
        r = np.array(pose[3:], dtype=float)
        dp = np.linalg.norm(p - tgt_p)
        dr = np.linalg.norm(r - tgt_r)
        if dp <= pos_tol and dr <= rot_tol:
            return True
        time.sleep(0.05)
    print(f"[!] UR did not reach target within tolerance ({label}). dp≈{dp:.4f}m dr≈{dr:.3f}rad")
    return False

# ─── SIGNAL HANDLER FOR CLEAN EXIT ────────────────────────────────────
def _signal_handler(sig, frame):
    print("\n[!] Interrupt—exiting.")
    finish_event.set()
    cv2.destroyAllWindows()
    sys.exit(0)
signal.signal(signal.SIGINT, _signal_handler)

# ─── PRE-INITIALIZE VIDEO STREAMS ────────────────────────────────────
color_cap = cv2.VideoCapture('/dev/video4', cv2.CAP_V4L2)
color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
color_cap.set(cv2.CAP_PROP_FPS, 30)

depth_pipe = rs.pipeline()
depth_cfg  = rs.config()
depth_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
depth_pipe.start(depth_cfg)

cv2.namedWindow("Live View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live View", 1200, 620)
0.6262366117816692, -0.35979032052551885, 0.2895795145877851, 2.276711217203636, 2.138919321529761, -0.05867999642291609
# ─── LOAD YOLO MODEL ─────────────────────────────────────────────────
yolo_model = YOLO(str(MODEL_PATH))
yolo_model.conf = CONF_THRESH
print(f"[?] Loaded YOLO model '{MODEL_PATH}' (thr={CONF_THRESH})")

# ─── THREADED INFERENCE WORKER ───────────────────────────────────────
def inference_worker():
    """Background thread that continuously runs YOLO inference on latest frame."""
    global latest_results
    print("[*] Inference worker started")
    
    while not finish_event.is_set():
        with inference_lock:
            frame = latest_frame
        
        if frame is not None:
            try:
                results = yolo_model.predict(frame, conf=CONF_THRESH, verbose=False)
                with inference_lock:
                    latest_results = results
                if not inference_ready.is_set():
                    inference_ready.set()
            except Exception as e:
                print(f"[!] Inference worker error: {e}")
        
        time.sleep(0.001)  # Tiny sleep to prevent CPU spinning
    
    print("[*] Inference worker stopped")

# ─── UTILITIES: STANDARDS LOADING & NEXT SCAN NUM ────────────────────
def load_standard_dimensions():
    global standards
    try:
        with open(STANDARD_DIM_FILE) as f:
            standards = json.load(f)
        return standards
    except Exception as e:
        print(f"[!] Error loading standard dimensions: {e}")
        return None

def get_next_scan_number(obj):
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    files = list(DOCUMENTS_DIR.glob(f"{obj}*Scan.pdf"))
    if not files:
        return 1
    nums = []
    for f in files:
        stem = f.stem.replace(obj, "").replace("Scan", "")
        if stem.isdigit():
            nums.append(int(stem))
    return max(nums)+1 if nums else 1

def wait_for_file(path, timeout=3.0):
    """Wait briefly for a file to appear (UI robustness)."""
    t0 = time.time()
    p = Path(path) if path else None
    while p and time.time() - t0 < timeout:
        if p.exists() and p.is_file():
            return True
        time.sleep(0.05)
    return bool(p and p.exists())

# ——— Background cleanup for scan PNGs (same idea as report.py) ———
_bg_cleaned_paths = set()

def remove_uniform_bg_inplace(path: str):
    """Whiten all pixels matching the top-left pixel. Modifies the file in place."""
    try:
        img = PILImage.open(path).convert("RGBA")
        bg  = img.getpixel((0, 0))[:3]  # (r,g,b)
        arr = np.array(img)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        mask = (r == bg[0]) & (g == bg[1]) & (b == bg[2])
        # White background (opaque), just like the report logic
        arr[mask] = [255, 255, 255, 255]
        PILImage.fromarray(arr, "RGBA").save(path)
    except Exception as e:
        print(f"[!] BG remove failed: {e}")

def ensure_clean_bg(path: str):
    """Run background cleanup once per path."""
    if not path or path in _bg_cleaned_paths:
        return
    remove_uniform_bg_inplace(path)
    _bg_cleaned_paths.add(path)

# ─── DOBOT HELPERS: suction + pick-and-place ──────────────────────────
def _dobot_set_suction(db: DoBotArm, on: bool):
    """
    Explicitly set suction without relying on status polling.
    Mirrors the class' toggle param pairs:
      ON  -> (enable=True,  on=True)
      OFF -> (enable=False, on=False)
    """
    if on:
        last_idx = dType.SetEndEffectorSuctionCup(db.api, True, True, isQueued=0)[0]
    else:
        last_idx = dType.SetEndEffectorSuctionCup(db.api, False, False, isQueued=0)[0]
    db.commandDelay(last_idx)
0.6262366117816692, -0.35979032052551885, 0.2895795145877851, 2.276711217203636, 2.138919321529761, -0.05867999642291609
def dobot_movl(db: DoBotArm, x: float, y: float, z: float, r: float = 0.0):
    """Linear move via DLL (includes rHead)."""
    print('DOBOT MOVING')
    last_idx = dType.SetPTPCmd(db.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, r)[0]
    db.commandDelay(last_idx)

def dobot_pick_and_place_db(db: DoBotArm):
    """Run PnP using an existing Dobot session."""
    try:
        px, py, pz, pr = DOBOT_PICK
        dx, dy, dz, dr = DOBOT_DROP
        pz_hover = pz + DOBOT_LIFT_MM
        dz_hover = dz + DOBOT_LIFT_MM

        # Approach pick (hover) → descend → suction ON → lift
        dobot_movl(db, px, py, pz_hover, pr)
        dobot_movl(db, px, py, pz,       pr)
        _dobot_set_suction(db, True)
        time.sleep(0.15)
        dobot_movl(db, px, py, pz_hover, pr)

        # Transit to drop (hover) → descend → suction OFF → lift
        dobot_movl(db, dx, dy, dz_hover, dr)
        dobot_movl(db, dx, dy, dz,       dr)
        _dobot_set_suction(db, False)
        time.sleep(0.15)
        dobot_movl(db, dx, dy, dz_hover, dr)

        # Return home
        db.moveHome()
        time.sleep(0.2)
    except Exception as e:
        print(f"[!] DoBot PnP error: {e}")
        raise

def conveyor_start_db(db: DoBotArm, speed_mm_s: float):
    """Start external motor on the SAME Dobot session (queued, wait until executed)."""
    api = db.api

    # Convert mm/s -> pulses/s
    STEP_PER_CIRCLE = 360.0 / 1.8 * 5.0 * 16.0
    MM_PER_CIRCLE   = 3.1415926535898 * 32.0
    vel = int(float(speed_mm_s) * STEP_PER_CIRCLE / MM_PER_CIRCLE)

    # Make sure the queue executor is running (idempotent to call)
    dType.SetQueuedCmdStartExec(api)

    # Enqueue motor start and wait until it executes
    idx_run = dType.SetEMotor(api, 0, 1, vel, 1)[0]
    for _ in range(300):  # ~3s
        cur = dType.GetQueuedCmdCurrentIndex(api)[0]
        if cur >= idx_run:
            break
        dType.dSleep(10)

    print(f"[?] Conveyor RUNNING ~{speed_mm_s} mm/s (pulses={vel}), queue idx={idx_run}")

def conveyor_stop_db(db: DoBotArm):
    """Stop motor on the SAME session, stop executor, disconnect."""
    api = db.api
    try:
        idx_stop = dType.SetEMotor(api, 0, 0, 0, 1)[0]
        for _ in range(300):
            cur = dType.GetQueuedCmdCurrentIndex(api)[0]
            if cur >= idx_stop:
                break
            dType.dSleep(10)
        dType.SetQueuedCmdStopExec(api)
    except Exception:
        pass

# ─── GOCATOR SCAN & DATA HANDLER ─────────────────────────────────────
def run_gocator_scan(object_type):
    """
    Returns (measurements_dict or None, image_path or None)
    """
    if not scan_lock.acquire(blocking=False):
        print("[!] Scan already in progress; skipping duplicate call.")
        return None, None
    print(f"[*] Running Gocator scan for {object_type}...")
    try:
        import GoSdk_MsgHandler
        from Gocator import (
            GoSdk, kApi, kIpAddress,
            GoDataSet, GoDataMsg, kNULL,
            GO_DATA_MESSAGE_TYPE_UNIFORM_SURFACE,
            GO_DATA_MESSAGE_TYPE_MEASUREMENT
        )

        out = RESULTS_DIR
        out.mkdir(exist_ok=True)

        scan_data = {'measurements': {}, 'scan_image_path': None}
        crank_radius  = None
        direct_height = None
        use_direct = {"bearing","bolt","crank","flat plate","gear","gearbox","nut"}

        def handler(dataset):
            nonlocal crank_radius, direct_height
            # 1) Measurement messages
            for i in range(GoSdk.GoDataSet_Count(dataset)):
                addr = GoSdk.GoDataSet_At(dataset, i)
                msg  = GoDataMsg(addr)
                if GoSdk.GoDataMsg_Type(msg) == GO_DATA_MESSAGE_TYPE_MEASUREMENT:
                    mid = GoSdk.GoMeasurementMsg_Id(msg)
                    val = GoSdk.GoMeasurementMsg_At(msg, 0).contents.numericVal
                    if object_type == 'crank' and mid == 5:
                        crank_radius = float(val)
                    if object_type in use_direct and mid == 1:
                        direct_height = float(val)

            # 2) Uniform surface → PLY + PNG
            for i in range(GoSdk.GoDataSet_Count(dataset)):
                addr = GoSdk.GoDataSet_At(dataset, i)
                msg  = GoDataMsg(addr)
                if GoSdk.GoDataMsg_Type(msg) == GO_DATA_MESSAGE_TYPE_UNIFORM_SURFACE:
                    XR = GoSdk.GoUniformSurfaceMsg_XResolution(msg)/1e6
                    YR = GoSdk.GoUniformSurfaceMsg_YResolution(msg)/1e6
                    ZR = GoSdk.GoUniformSurfaceMsg_ZResolution(msg)/1e6
                    XO = GoSdk.GoUniformSurfaceMsg_XOffset(msg)/1e3
                    YO = GoSdk.GoUniformSurfaceMsg_YOffset(msg)/1e3
                    ZO = GoSdk.GoUniformSurfaceMsg_ZOffset(msg)/1e3
                    w  = GoSdk.GoUniformSurfaceMsg_Width(msg)
                    l  = GoSdk.GoUniformSurfaceMsg_Length(msg)

                    size = w*l
                    ptr = GoSdk.GoUniformSurfaceMsg_RowAt(msg,0)
                    Z   = np.ctypeslib.as_array(ptr, shape=(size,)).astype(np.double)
                    Z[Z == -32768] = np.nan
                    Z = Z*ZR + ZO

                    X = np.tile((np.arange(w)*XR + XO), l)
                    Y = np.repeat((np.arange(l)*YR + YO), w)
                    pts = np.stack((X, Y, Z), axis=1)
                    valid = pts[~np.isnan(pts).any(axis=1)]

                    if valid.size:
                        xv, yv, zv = valid[:,0], valid[:,1], valid[:,2]
                        length_mm = float(xv.max()-xv.min())
                        width_mm  = float(yv.max()-yv.min())
                        comp_h    = float(zv.max()-zv.min())
                        height_mm = direct_height if (object_type in use_direct and direct_height is not None) else comp_h

                        scan_data['measurements'].update({
                            'length_mm': length_mm,
                            'width_mm':  width_mm,
                            'height_mm': height_mm,
                            'point_count': len(valid)
                        })
                        if object_type == 'crank':
                            scan_data['measurements']['circle_radius_mm'] = crank_radius or 0.0

                        # Save PLY
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        ply_dir = out / "UNIFORM_SURFACE_PLY"
                        ply_dir.mkdir(exist_ok=True)
                        ply_path = ply_dir / f"UNIFORM_SURFACE_{ts}.ply"
                        with open(ply_path, 'w') as f:
                            f.write('ply\nformat ascii 1.0\n')
                            f.write(f'element vertex {valid.shape[0]}\n')
                            f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
                            for x,y,z in valid:
                                f.write(f"{x} {y} {z}\n")

                        # Save height-map PNG (absolute path)
                        arr = Z.reshape((l, w))
                        m   = np.nanmean(arr)
                        arr = np.nan_to_num(arr, nan=m)
                        img_name = f"scan_{object_type}_{ts}.png"
                        img_path = out / img_name
                        plt.imsave(str(img_path), arr, cmap='viridis')
                        scan_data['scan_image_path'] = str(img_path)
                    break

        # Run snapshot
        import GoSdk_MsgHandler
        from Gocator import GoSdk, kApi, kIpAddress, GoDataSet, GoDataMsg, kNULL
        api, sys_, sensor = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        ds, dm = GoDataSet(), GoDataMsg()
        GoSdk.GoSdk_Construct(ctypes.byref(api))
        GoSdk.GoSystem_Construct(ctypes.byref(sys_), None)
        ipa = kIpAddress()
        kApi.kIpAddress_Parse(ctypes.byref(ipa), SCANNER_IP)
        GoSdk.GoSystem_FindSensorByIpAddress(sys_, ctypes.byref(ipa), ctypes.byref(sensor))
        GoSdk.GoSensor_Connect(sensor)
        GoSdk.GoSystem_EnableData(sys_, True)
        mgr = GoSdk_MsgHandler.MsgManager(GoSdk, sys_, ds)
        mgr.SetDataHandler(RECEIVE_TIMEOUT, handler)
        print("[*] Taking Gocator snapshot...")
        GoSdk.GoSensor_Stop(sensor)
        GoSdk.GoSensor_Snapshot(sensor)
        time.sleep(3.0)
        mgr.SetDataHandler(RECEIVE_TIMEOUT, kNULL)
        mgr.stop()
        print("[?] Gocator scan completed")

        return scan_data['measurements'] if scan_data['measurements'] else None, scan_data['scan_image_path']
    except Exception as e:
        print(f"[!] Gocator error: {e}")
        return None, None
    finally:
        try:
            scan_lock.release()
        except Exception:
            pass

# ─── MEASUREMENT CALCULATION & INSPECTION ────────────────────────────
def calculate_object_measurements(basic, obj):
    if not basic:
        return None
    if obj in ["bearing","gear","washer"]:
        dia = max(basic['width_mm'], basic['height_mm'])
        return {
            "radius_mm": dia/2,
            "circumference_mm": np.pi * dia,
            "height_mm": basic['height_mm']
        }
    if obj == "crank":
        return {
            "length_mm": basic['length_mm'],
            "width_mm":  basic['width_mm'],
            "height_mm": basic['height_mm'],
            "circle_radius_mm": basic.get('circle_radius_mm',0.0)
        }
    return {
        "length_mm": basic['length_mm'],
        "width_mm":  basic['width_mm'],
        "height_mm": basic['height_mm']
    }

def perform_inspection(meas, stds, obj):
    if not meas or not stds:
        return None
    standard = stds['objects'][obj]['measurements']
    tol = standard['tolerance_mm']
    results = {
        'overall_pass': True,
        'dimension_results': {},
        'summary': []
    }
    for dim, val in meas.items():
        if dim in standard:
            stdv = standard[dim]
            lo, hi = stdv - tol, stdv + tol
            status = "PASS" if lo <= val <= hi else "FAIL"
            if status == "FAIL":
                results['overall_pass'] = False
            results['dimension_results'][dim] = {
                'measured': val, 'standard': stdv,
                'tolerance': tol, 'min_allowed': lo,
                'max_allowed': hi, 'status': status
            }
            results['summary'].append(
                f"{dim.replace('_',' ').title()}: {val:.2f}mm "
                f"(Standard: {stdv:.2f}mm ±{tol:.2f}mm) [{status}]"
            )
    return results

# ─── PDF REPORT GENERATION (mirrors report.py logic) ──────────────────
def generate_pdf_report(obj, num, meas, insp, img_path, ts):
    try:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{obj}{num}Scan.pdf"
        filepath = DOCUMENTS_DIR / filename

        doc    = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        title_style = ParagraphStyle(
            'Title', parent=styles['Heading1'],
            fontSize=24, spaceAfter=30, alignment=TA_CENTER
        )
        story.append(Paragraph(f"Inspection Report - {obj.title()}", title_style))
        story.append(Spacer(1,20))

        # Details
        details = [
            ['Object Name:', obj.title()],
            ['Scan Count for Object:', str(num)],
            ['Date & Time:', ts.strftime("%Y-%m-%d %H:%M:%S")],
            ['Inspection Result:', 'PASS' if insp['overall_pass'] else 'FAIL']
        ]
        tbl = Table(details, colWidths=[2.5*inch,3.5*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(0,-1),colors.lightgrey),
            ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
            ('GRID',(0,0),(-1,-1),1,colors.black),
        ]))
        story.append(tbl)
        story.append(Spacer(1,20))

        # — remove uniform background from scan image — (same as report.py)
        if img_path and Path(img_path).exists():
            try:
                img = PILImage.open(img_path).convert("RGBA")
                bg  = img.getpixel((0,0))
                data= np.array(img)
                r,g,b,_ = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
                mask    = (r==bg[0]) & (g==bg[1]) & (b==bg[2])
                data[mask] = [255,255,255,255]
                PILImage.fromarray(data).save(img_path)
            except Exception:
                pass

            story.append(Paragraph("Scan Visualization", styles['Heading2']))
            # Dynamic sizing per report.py
            img_w, img_h = 6, 3
            if meas and ('length_mm' in meas and 'width_mm' in meas):
                l, wv = meas['length_mm'], meas['width_mm']
                if min(l, wv) > 0 and max(l, wv)/min(l, wv) < 1.1:
                    img_w, img_h = 3, 3
            else:
                img_w, img_h = 3, 3

            story.append(RLImage(str(img_path), width=img_w*inch, height=img_h*inch))
            story.append(Spacer(1,20))

        # Inspection table
        story.append(Paragraph("Inspection Results", styles['Heading2']))
        rows = [['Dimension','Measured (mm)','Standard (mm)','Tolerance','Status']]
        for d,r in insp['dimension_results'].items():
            rows.append([
                d.replace('_',' ').title(),
                f"{r['measured']:.2f}",
                f"{r['standard']:.2f}",
                f"±{r['tolerance']:.2f}",
                r['status']
            ])
        rt = Table(rows, colWidths=[1.5*inch]*5)
        rt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('GRID',(0,0),(-1,-1),1,colors.black),
        ]))
        story.append(rt)
        story.append(Spacer(1,20))

        # Summary
        story.append(Paragraph("Summary", styles['Heading2']))
        for line in insp['summary']:
            story.append(Paragraph(line, styles['Normal']))

        doc.build(story)
        print(f"[✓] PDF report generated: {filepath}")
        return filepath

    except Exception as e:
        print(f"[!] Error generating PDF report: {e}")
        return None

# ─── UI PROMPT FOR INITIAL LABEL ────────────────────-9.212165832519531, 241.2690887451172, -48.746620178222656, 92.18660736083984──────────────────
def get_label():
    root = tk.Tk(); root.withdraw()
    lbl = simpledialog.askstring("Target Object", "Enter label:")
    if not lbl:
        print("[!] No label—exit."); sys.exit(1)
    try:
        root.destroy()
    except Exception:
        pass
    return lbl.strip().lower()

# ─── UR5e ePick Pick & Place (after Accept) — FIXED ───────────────────
def ur_epick_pick_and_place():
    try:
        # Build hover poses
        pick_hover = list(UR_PICK_POSE)
        drop_hover = list(UR_DROP_POSE)
        pick_hover[2] += UR_PNP_LIFT
        drop_hover[2] += UR_PNP_LIFT

        # Go to pick hover → pick → suction ON → lift
        ur_moveL(pick_hover, UR_SPEED, UR_ACCEL, label="PnP: pick hover")
        ur_wait_until_reached(pick_hover, label="PnP: pick hover")
        ur_moveL(UR_PICK_POSE, UR_SPEED, UR_ACCEL, label="PnP: pick")
        ur_wait_until_reached(UR_PICK_POSE, label="PnP: pick")

        print("[*] ePick: suction ON")
        try:
            start_suction()
        except Exception as e:
            print(f"[!] ePick start_suction error: {e}")
        time.sleep(0.3)

        ur_moveL(pick_hover, UR_SPEED, UR_ACCEL, label="PnP: lift from pick")
        ur_wait_until_reached(pick_hover, label="PnP: lift from pick")

        # Transit to drop hover → drop → suction OFF → lift
        ur_moveL(drop_hover, UR_SPEED, UR_ACCEL, label="PnP: drop hover")
        ur_wait_until_reached(drop_hover, label="PnP: drop hover")
        ur_moveL(UR_DROP_POSE, UR_SPEED, UR_ACCEL, label="PnP: drop")
        ur_wait_until_reached(UR_DROP_POSE, label="PnP: drop")

        print("[*] ePick: suction OFF")
        try:
            stop_suction()
        except Exception as e:
            print(f"[!] ePick stop_suction error: {e}")
        time.sleep(0.3)

        ur_moveL(drop_hover, UR_SPEED, UR_ACCEL, label="PnP: lift from drop")
        ur_wait_until_reached(drop_hover, label="PnP: lift from drop")

        # Optional: return to detection pose so the cell is in a known state
        ur_moveL(DETECTION_POSE, UR_SPEED, UR_ACCEL, label="PnP: return to detection")
        ur_wait_until_reached(DETECTION_POSE, label="PnP: return to detection")

        print("[✓] UR5e ePick pick & place completed.")
    except Exception as e:
        print(f"[!] UR5e PnP error: {e}")

# ─── WORKFLOW THREAD: DETECTION → DOBOT PNP → VIEW LOAD → CONVEYOR → ALIGN → SCAN ─
def workflow(label):
    global scan_basic, scan_measurements, scan_inspection, scan_image_path

    # 1) Move UR to detection pose (parallel to the display loop opening the window)
    print("[*] Moving to detection pose...")
    ur_moveL(DETECTION_POSE, UR_SPEED, UR_ACCEL, label="detection")
    ur_wait_until_reached(DETECTION_POSE, label="detection")

    # 2) While the viewing window is loading, give UI time to appear
    print("[*] Waiting for viewing window to load...")
    live_view_ready.wait(timeout=6.0)

    # 2.5) Create ONE Dobot session for everything that follows
    db = DoBotArm(HOME_X, HOME_Y, HOME_Z)
    time.sleep(7)
    # 3) Run pick and place on this SAME session
    dobot_pick_and_place_db(db)

    # 4) Start conveyor on the SAME session
    conveyor_start_db(db, CONVEYOR_SPEED)

    # 5) Wait for YOLO detection
    while not finish_event.is_set() and detected_box == [0, 0]:
        time.sleep(0.05)
    cx, cy = detected_box
    print(f"[*] Detected at {cx},{cy}")

    # 6) PIR trigger using the SAME session
    dType.SetIOMultiplexing(db.api, PIR_PIN, 3, 1)  # PIR as DI
    while dType.GetIODI(db.api, PIR_PIN)[0] != 0:
        time.sleep(0.05)
    print("[?] PIR triggered")
    time.sleep(0.58)

    # 7) Stop conveyor and disconnect Dobot
    conveyor_stop_db(db)
    try:
        db.moveHome()
        db.dobotDisconnect()
    except Exception:
        pass

    # 6) Alignment: MOVE FIRST, WAIT ARRIVAL, THEN SAMPLE
    moved = ur_moveL(ALIGNMENT_POSE, UR_SPEED, UR_ACCEL, label="alignment")
    if moved:
        if not ur_wait_until_reached(ALIGNMENT_POSE, label="alignment"):
            # retry once
            ur_moveL(ALIGNMENT_POSE, UR_SPEED, UR_ACCEL, label="alignment (retry)")
            ur_wait_until_reached(ALIGNMENT_POSE, label="alignment (retry)")
    else:
        print("[!] Skipping alignment move due to UR error.")

    # Depth-based Z adjust ONLY after arrival
    samples = []
    for _ in range(10):
        df = depth_pipe.wait_for_frames().get_depth_frame()
        if df:
            d = df.get_distance(cx, cy) * 1000
            if d >= 177.8:
                samples.append(d)
        time.sleep(0.1)

    if samples:
        avg = sum(samples)/len(samples)
        off = 254.0 - avg
        ALIGNMENT_POSE[2] += off/1000.0
        SCAN_POSE[2]      = ALIGNMENT_POSE[2]
        print(f"[*] Z adjust {off:.1f}mm")
        ur_moveL(ALIGNMENT_POSE, UR_SPEED, UR_ACCEL, label="alignment (adjusted)")
        ur_wait_until_reached(ALIGNMENT_POSE, label="alignment (adjusted)")

    # 7) Move to scan pose and wait
    ur_moveL(SCAN_POSE, UR_SPEED, UR_ACCEL, label="scan pose")
    ur_wait_until_reached(SCAN_POSE, label="scan pose")

    # 8) Gocator scan (guarded)
    meas, img = run_gocator_scan(label)
    scan_basic      = meas
    scan_image_path = img
    scan_measurements = calculate_object_measurements(meas, label) if meas else None
    scan_inspection   = perform_inspection(scan_measurements, standards, label) if scan_measurements else None

    if scan_image_path:
        print(f"[✓] Scan image saved: {scan_image_path}")
    else:
        print("[!] No scan image produced.")

    finish_event.set()

# ─── LIVE DISPLAY & YOLO DETECTION LOOP ──────────────────────────────
def display_loop(label):
    global latest_depth_frame, latest_frame, latest_results
    start_ts = time.time()

    while not finish_event.is_set():
        # depth
        df = depth_pipe.wait_for_frames().get_depth_frame()
        if df:
            with depth_lock:
                latest_depth_frame = df
            dimg    = np.asanyarray(df.get_data(), dtype=float)
            clamped = np.clip(dimg, 100, 500)
            norm    = ((clamped - 100) / 400 * 255).astype(np.uint8)
            dcol    = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        else:
            dcol = np.zeros((480, 640, 3), dtype=np.uint8)

        # RGB & detection
        ret, frame = color_cap.read()
        if not ret:
            continue

        # Update latest frame for inference thread
        with inference_lock:
            latest_frame = frame.copy()

        if detected_box == [0, 0]:
            # Get results from inference thread
            with inference_lock:
                results = latest_results if latest_results is not None else []

            # Adaptive threshold after some waiting time
            elapsed = time.time() - start_ts
            thr = CONF_THRESH if elapsed < CONF_RELAX_AFTER_S else max(CONF_MIN, CONF_THRESH - 0.1)

            want = label.strip().lower()
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    try:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        class_name = yolo_model.names[class_id]
                        score = float(box.conf[0])
                        
                        if class_name.lower() != want:
                            continue
                        if score < thr:
                            continue
                            
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate center point
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cx = np.clip(cx, 0, frame.shape[1]-1)
                        cy = np.clip(cy, 0, frame.shape[0]-1)

                        detected_box[0], detected_box[1] = int(cx), int(cy)
                        print(f"[?] Detection: {want} score={score:.2f} at ({detected_box[0]},{detected_box[1]})")
                        break
                    except Exception:
                        continue
                if detected_box != [0, 0]:
                    break

        # overlay annotation
        if detected_box != [0, 0]:
            cx, cy = detected_box
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
            cv2.putText(frame, f"Detected: {label}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            cv2.putText(frame, f"Detecting: {label}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        combined = np.hstack((frame, dcol))
        cv2.imshow("Live View", combined)
        # Signal once that the live view is ready (replaces fixed 25s sleep)
        if not live_view_ready.is_set():
            live_view_ready.set()
        if cv2.waitKey(1) == 27:
            finish_event.set()
            break

    depth_pipe.stop()
    color_cap.release()
    cv2.destroyAllWindows()

# ─── ALIGNMENT UTILITY FOR POST-SCAN UI ───────────────────────────────
def perform_alignment():
    """Re-run alignment sampling and update Z; move to ALIGNMENT then back to SCAN at new Z."""
    ur_moveL(ALIGNMENT_POSE, UR_SPEED, UR_ACCEL, label="realign: to ALIGNMENT")
    ur_wait_until_reached(ALIGNMENT_POSE, label="realign: to ALIGNMENT")

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(cfg)

    cx, cy = 320, 240
    samples = []
    for _ in range(10):
        frame = pipeline.wait_for_frames().get_depth_frame()
        if frame:
            d = frame.get_distance(cx, cy) * 1000
            if d >= 177.8:
                samples.append(d)
        time.sleep(0.1)
    pipeline.stop()

    if samples:
        avg = sum(samples)/len(samples)
        off = 254.0 - avg
        ALIGNMENT_POSE[2] += off/1000.0
        SCAN_POSE[2]      = ALIGNMENT_POSE[2]
        print(f"[*] Realign: Z adjust {off:.1f} mm")
        ur_moveL(ALIGNMENT_POSE, UR_SPEED, UR_ACCEL, label="realign: apply Z")
        ur_wait_until_reached(ALIGNMENT_POSE, label="realign: apply Z")

# ─── POST-SCAN UI ────────────────────────────────────────────────────
def post_scan_ui(label):
    global scan_basic, scan_measurements, scan_inspection, scan_image_path

    if standards is None:
        load_standard_dimensions()

    root = tk.Tk()
    root.title(f"Post-Scan Review — {label}")
    root.geometry("900x950")

    img_label = tk.Label(root)
    img_label.pack(pady=10)
    # Smaller measurement area: shorter height and no vertical expand
    text      = tk.Text(root, height=12)
    text.pack(fill=tk.X, expand=False, padx=10, pady=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=12)

    def update_ui():
        text.delete("1.0", tk.END)

        # Ensure image exists then load on main thread
        if scan_image_path:
            exists = wait_for_file(scan_image_path, timeout=2.0)
            text.insert(tk.END, f"[i] Scan image path: {scan_image_path}\n")
        else:
            exists = False

        if exists:
            try:
                # Make background white like in the report, in-place (runs once per file)
                ensure_clean_bg(scan_image_path)

                # Dynamic sizing like report.py:
                # square-ish -> 480x480; otherwise -> 800x400
                img_w, img_h = 800, 400
                try:
                    if scan_basic and ('length_mm' in scan_basic and 'width_mm' in scan_basic):
                        l = float(scan_basic['length_mm'])
                        wv = float(scan_basic['width_mm'])
                        if min(l, wv) > 0 and max(l, wv) / min(l, wv) < 1.1:
                            img_w, img_h = 800, 400
                    elif scan_measurements and ('radius_mm' in scan_measurements):
                        img_w, img_h = 800, 400
                except Exception:
                    img_w, img_h = 800, 400

                pil_img = PILImage.open(scan_image_path).resize((int(img_w), int(img_h)), PILImage.LANCZOS)
                tk_img  = ImageTk.PhotoImage(pil_img, master=root)
                img_label.configure(image=tk_img)
                img_label.image = tk_img

            except Exception as e:
                text.insert(tk.END, f"[!] Failed to load image: {e}\n")
        else:
            text.insert(tk.END, "[i] No scan image available. You can Rescan.\n")

        if not scan_measurements or not scan_inspection:
            text.insert(tk.END, "[i] No measurements/inspection available. You can Rescan.\n")
            return

        text.insert(tk.END, "\nMEASUREMENTS:\n")
        for k, v in scan_measurements.items():
            status = scan_inspection["dimension_results"].get(k, {}).get("status", "N/A")
            text.insert(tk.END, f"  • {k}: {v:.2f} mm [{status}]\n")

    def on_rescan():
        global scan_basic, scan_measurements, scan_inspection, scan_image_path
        ur_moveL(SCAN_POSE, UR_SPEED, UR_ACCEL, label="UI: to SCAN for rescan")
        ur_wait_until_reached(SCAN_POSE, label="UI: to SCAN for rescan")
        meas, img = run_gocator_scan(label)
        if not meas:
            messagebox.showwarning("Rescan Failed", "No measurements from scanner. Try again.")
            return
        scan_basic        = meas
        scan_image_path   = img
        scan_measurements = calculate_object_measurements(meas, label)
        scan_inspection   = perform_inspection(scan_measurements, standards, label)
        update_ui()

    def on_realign():
        perform_alignment()
        ur_moveL(SCAN_POSE, UR_SPEED, UR_ACCEL, label="UI: to SCAN after realign")
        ur_wait_until_reached(SCAN_POSE, label="UI: to SCAN after realign")
        # Per user: do NOT auto-rescan; wait for user to hit Rescan.
        update_ui()

    def on_accept():
        if not scan_measurements or not scan_inspection:
            messagebox.showwarning("Cannot Accept", "No valid measurements to report. Rescan first.")
            return
        num = get_next_scan_number(label)
        ts  = datetime.now()
        pdf = generate_pdf_report(label, num, scan_measurements, scan_inspection, scan_image_path, ts)
        if pdf:
            messagebox.showinfo("Report Saved", f"Report: {pdf}")
        else:
            messagebox.showerror("Error", "Failed to generate report.")
            return  # don't proceed to PnP if no report

        # Run UR5e ePick pick & place BEFORE exit
        try:
            on_status = tk.Label(root, text="Running UR5e pick & place...", fg="blue")
            on_status.pack(pady=6)
            root.update_idletasks()
        except Exception:
            pass

        ur_epick_pick_and_place()

        # Done → close UI and exit
        try:
            root.destroy()
        except Exception:
            pass
        os._exit(0)

    for txt, cmd in [("Rescan", on_rescan), ("Realign", on_realign), ("Accept Scan", on_accept)]:
        btn = tk.Button(btn_frame, text=txt, command=cmd, width=16)
        btn.pack(side=tk.LEFT, padx=10)

    update_ui()
    root.mainloop()

# ─── MAIN ENTRYPOINT ─────────────────────────────────────────────────
def main():
    load_standard_dimensions()
    #epick_init()  # NEW: initialize ePick connection once
    label = get_label()
    
    # Start inference worker thread for 30 FPS performance
    threading.Thread(target=inference_worker, daemon=True).start()
    
    # Start workflow thread
    threading.Thread(target=workflow, args=(label,), daemon=True).start()
    
    display_loop(label)          # blocks until finish_event set by workflow
    post_scan_ui(label)          # then show post-scan UI

if __name__ == "__main__":
    main()
