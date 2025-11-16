#!/usr/bin/env python3

import time
import ctypes
import numpy as np
from DoBotArm import DoBotArm
import DobotDllType as dType

# ─── DOBOT PICK & PLACE COORDINATES (X, Y, Z, R) ──────────────────────
DOBOT_PICK = [2.037076711654663, 324.0500793457031, -37.299278259277344, 89.63982391357422]
DOBOT_DROP = [347.0736999511719, -34.666500091552734, 35.241695404052734, -5.703912734985352]
DOBOT_LIFT_MM = 60.0  # lift amount between pick and place

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

def main():
    db = DoBotArm(5, 5, 5) # the three numbers here seem to not matter
    db.dobotConnect() #the preceding "db" is just because we are creating 
    #an instance and working within that instance, it might not be needed 
    #at all should we make the code within a method
    db.gripperOpen() #changed from toggleGripper because this is more simple
    time.sleep(1)
    db.gripperClose()
    time.sleep(1)
    db.gripperOff()
    
    #working to figure out how to hard code movements, even if the 
    #status light starts green, as soon as any arm movement begins it 
    #hard stops and immediately becomes red again
    
    #doesn't seem to be a range or "out of work envelope" issue because
    #even with newly logged coordinate points it still self terminates
    #during the first actual (non-gripper) movement
    
    # Define points A and B (X, Y, Z, R)
    point_a = (131.198, 187.791, 55.450, 55.060)

    point_b = (130.151, 186.229, 108.779, 55.051) #using 2nd point to avoid dragging the gearbox with the gear
    
    point_c = (153.549, 100.546, 33.175, 31.595)


    # Move linearly from A to B
    dobot_movl(db, *point_a) 
    db.gripperOpen()
    #time.sleep(1)
    db.gripperClose()
    #time.sleep(1.5)
     
    
    dobot_movl(db, *point_b) 
    db.gripperClose()
    #time.sleep(1)
    dobot_movl(db, *point_c)
    db.gripperOpen()
    #time.sleep(1)
    db.gripperOff()
    #lets add some more coordinates so the motion is smoother
   
   
'''
probably going to need new points because the current arm might collide with the ur5e so we're going to need a further default position

the red light doesnt really seem to be an issue considering that the code seems to work regardless of it
'''
if __name__ == "__main__":
    main()
