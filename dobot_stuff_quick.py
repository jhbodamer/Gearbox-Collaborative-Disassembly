#!/usr/bin/env python3
"""
scanning_pipeline.py

Combined end-to-end pipeline for:
 1) YOLO + depth live detection with conveyor & PIR trigger
 2) UR5e alignment (arrive → sample 10 depths → offset to 10 in / 254 mm) & Gocator scan
 3) Post-scan review UI (Rescan, Realign, Accept → PDF report)
 4) After Accept → UR5e ePick pick & place → exit

Notes:
- Uses a SINGLE RTDE session with a movement lock.
- Waits for robot ARRIVAL at ALIGNMENT/SCAN poses before sampling/scan.
- Saves scan PNG/PLY to ~/Code/test/Object-Detection/results (absolute paths).
- UI loads the exact saved PNG (waits briefly if needed).
"""

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

# Headless matplotlib for saving images during scan
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
import epick_gripper  # provides start_suction() / stop_suction()

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
SCAN_POSE       = [0.620943295785446, -0.38508224790201964, 0.31697166063512483, 2.243004555383064, 2.168806847593467, -0.024489224430529654]
                   
HOME_X, HOME_Y, HOME_Z = 200, 0, 50
PIR_PIN        = 15
CONVEYOR_SPEED = 60
UR_SPEED       = 0.15   # per user request: speed/accel 0.3
UR_ACCEL       = 0.1
SCANNER_IP     = b"192.168.1.10"
RECEIVE_TIMEOUT= 10000

# ─── UR5e ePick PnP COORDS (provided) ─────────────────────────────────
UR_PICK_POSE = [0.704572722840677, -0.38359319806077835, 0.11480716747680059, 2.0921095716905325, 2.284226773887462, 0.13239764555724826]
UR_DROP_POSE = [0.35408779470312507, -0.6277985314083585, 0.09163414487131588,-2.11324724252493, -2.2819505729846736, -0.02765755486127989]
UR_PNP_LIFT = 0.06  # 60 mm hover for approach/depart


# ─── DOBOT PICK & PLACE COORDINATES (X, Y, Z, R) ──────────────────────
DOBOT_PICK = [2.037076711654663, 324.0500793457031, -37.299278259277344, 89.63982391357422]
DOBOT_DROP = [347.0736999511719, -34.666500091552734, 35.241695404052734, -5.703912734985352]
DOBOT_LIFT_MM = 60.0  # lift amount between pick and place


# ─── HAILO MODEL CONFIG ───────────────────────────────────────────────
INFERENCE_HOST = "@local"
DEVICE_TYPE    = ["HAILORT/HAILO8"]
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



# ====== DOBOT MOVEMENT ======
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

def dobot_movl(db: DoBotArm, x: float, y: float, z: float, r: float = 0.0):
    """Linear move via DLL (includes rHead)."""
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
        dobot_movl(db, px, py, pz, pr)
        # _dobot_set_suction(db, True)
        time.sleep(0.15)
        dobot_movl(db, px, py, pz_hover, pr)

        # Transit to drop (hover) → descend → suction OFF → lift
        dobot_movl(db, dx, dy, dz_hover, dr)
        dobot_movl(db, dx, dy, dz,       dr)
        # _dobot_set_suction(db, False)
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


def main():
	db = DoBotArm(HOME_X, HOME_Y, HOME_Z)
	dobot_pick_and_place_db(db)

