#!/usr/bin/env python3
import time
import ctypes
from ctypes import byref
import os
import sys

# --- 1. SETUP ENVIRONMENT (Fixes the KeyError/OSError) ---
# Modify this path if your SDK is somewhere else
SDK_PATH = "/home/robotics/Documents/human_robot_collab/test/fall_practice/GO_SDK"

# Set the environment variable if it isn't already set correctly
if 'GO_SDK_4' not in os.environ:
    os.environ['GO_SDK_4'] = SDK_PATH

# Check if the .so file actually exists to prevent crashing later
lib_path = os.path.join(SDK_PATH, 'lib', 'linux_arm64', 'libkApi.so')
if not os.path.exists(lib_path):
    print(f"[ERROR] Could not find library at: {lib_path}")
    print("Please check SDK_PATH at the top of the script.")
    sys.exit(1)

# --- 2. IMPORTS ---
# Now we import Gocator modules since the env var is set
try:
    import GoSdk_MsgHandler
    from Gocator import (
        GoSdk, kApi, RecieveData, get_measurement_decision,
        kIpAddress, GoDataSet, GoDataMsg, kNULL
    )
except ImportError as e:
    print(f"[ERROR] Failed to import Gocator wrappers: {e}")
    print("Ensure Gocator.py and GoSdk_MsgHandler.py are in this folder.")
    sys.exit(1)

# --- 3. CONFIGURATION ---
SCANNER_IP = b"192.168.1.10"
RECEIVE_TIMEOUT = 20000  # Increased slightly for safety

def trigger_scanner():
    """
    Connects to the Gocator, triggers a single snapshot, and returns the decision.
    """
    # Create C pointers
    api = ctypes.c_void_p()
    system = ctypes.c_void_p()
    sensor = ctypes.c_void_p()
    dataset = GoDataSet()
    
    # Initialize connection
    print(f"[*] Connecting to Gocator at {SCANNER_IP.decode()}...")
    try:
        GoSdk.GoSdk_Construct(byref(api))
        GoSdk.GoSystem_Construct(byref(system), None)

        ip_addr = kIpAddress()
        kApi.kIpAddress_Parse(byref(ip_addr), SCANNER_IP)
        
        # Connect to sensor
        GoSdk.GoSystem_FindSensorByIpAddress(system, byref(ip_addr), byref(sensor))
        GoSdk.GoSensor_Connect(sensor)
        GoSdk.GoSystem_EnableData(system, True)

        # Setup Message Handler
        mgr = GoSdk_MsgHandler.MsgManager(GoSdk, system, dataset)
        mgr.SetDataHandler(RECEIVE_TIMEOUT, RecieveData)

        # Trigger Scan
        print("[*] Triggering Snapshot...")
        GoSdk.GoSensor_Stop(sensor)
        GoSdk.GoSensor_Snapshot(sensor)

        # Wait for data (Adjust sleep if scan takes longer)
        print("[*] Waiting for data processing...")
        time.sleep(2.5) 
        
        # Cleanup Handler
        mgr.SetDataHandler(RECEIVE_TIMEOUT, kNULL)
        mgr.stop()

        # Get Result
        decision = get_measurement_decision()
        print(f"[*] Scan Complete. Decision Value: {decision}")
        return decision

    except Exception as e:
        print(f"[ERROR] Gocator Exception: {e}")
        import traceback
        traceback.print_exc()
        return -1

if __name__ == "__main__":
    print("--- Starting Gocator Test ---")
    result = trigger_scanner()
    print(f"--- Finished. Final Result: {result} ---")