#!/usr/bin/env python
import time
import sys
from ctypes import *
sys.path.insert(1,'./DLL')
import DobotDllType as dType


"""-------The DoBot Control Class-------
Variables:
suction = Suction is currently on/off
picking: shows if the dobot is currently picking or dropping an item
api = variable for accessing the dobot .dll functions
home% = home position for %
                                  """

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}



#Main control class for the DoBot Magician.
class DoBotArm:
    def __init__(self, homeX, homeY, homeZ):
        self.suction = False
        self.picking = False
        self.api = dType.load()
        self.homeX = homeX
        self.homeY = homeY
        self.homeZ = homeZ
        self.connected = False
        self.dobotConnect()

    def __del__(self):
        self.dobotDisconnect()

    #Attempts to connect to the dobot
    def dobotConnect(self):
        if(self.connected):
            print("You're already connected")
        else:
            state = dType.ConnectDobot(self.api, "", 115200)[0]
            if(state == dType.DobotConnect.DobotConnect_NoError):
                print("Connect status:",CON_STR[state])
                dType.SetQueuedCmdClear(self.api)

                dType.SetHOMEParams(self.api, self.homeX, self.homeY, self.homeZ, 0, isQueued = 1)
                dType.SetPTPJointParams(self.api, 300, 300, 300, 300, 300, 300, 300, 300, isQueued = 1)
                dType.SetPTPCommonParams(self.api, 150, 150, isQueued = 1)

                #dType.SetHOMECmd(self.api, temp = 0, isQueued = 1)
                self.connected = True
                return self.connected
            else:
                print("Unable to connect")
                print("Connect status:",CON_STR[state])
                return self.connected

    #Returns to home location and then disconnects
    def dobotDisconnect(self):
        self.moveHome()
        dType.DisconnectDobot(self.api)

    #Delays commands
    def commandDelay(self, lastIndex):
        dType.SetQueuedCmdStartExec(self.api)
        while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            dType.dSleep(200)
        dType.SetQueuedCmdStopExec(self.api)

    #Toggles suction peripheral on/off
    def toggleSuction(self):
        lastIndex = 0
        if(self.suction):
            lastIndex = dType.SetEndEffectorSuctionCup(self.api, False, False, isQueued = 0)[0]
            self.suction = False
        else:
            lastIndex = dType.SetEndEffectorSuctionCup(self.api, True, True, isQueued = 0)[0]
            self.suction = True
        self.commandDelay(lastIndex)

# Closes the gripper
    def gripperClose(self):
        # True closes the claw mechanically for your gripper
        lastIndex = dType.SetEndEffectorGripper(self.api, True, True, isQueued=0)[0]
        self.suction = True
        self.commandDelay(lastIndex)

# Opens the gripper
    def gripperOpen(self):
        # False opens the claw mechanically for your gripper
        lastIndex = dType.SetEndEffectorGripper(self.api, True, False, isQueued=0)[0]
        self.suction = False
        self.commandDelay(lastIndex)

# Deactivates the gripper
    def gripperOff(self):
        # Turn power off and reset state
        lastIndex = dType.SetEndEffectorGripper(self.api, False, False, isQueued=0)[0]
        self.suction = False
        self.commandDelay(lastIndex)


    #Moves arm to X/Y/Z Location
    def moveArmXY(self,x,y):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, self.homeZ, 0)[0]
        self.commandDelay(lastIndex)

    # Moves arm to X/Y/Z/R in linear mode
    def moveArmXYZ(self, x: float, y: float, z: float):
        """
        Linear move to the specified X, Y, Z and R (end-effector rotation).
        """
        # PTPMOVLXYZMode: linear interpolation, includes rotation
        lastIndex = dType.SetPTPCmd(
            self.api,
            dType.PTPMode.PTPMOVLXYZMode,
            x, y, z
        )[0]
        self.commandDelay(lastIndex)

    #Returns to home location
    def moveHome(self):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, self.homeX, self.homeY, self.homeZ, 0)[0]
        self.commandDelay(lastIndex)

    #Toggles between hover and item level
    def pickToggle(self, itemHeight):
        lastIndex = 0
        positions = dType.GetPose(self.api)
        if(self.picking):
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, positions[0], positions[1], self.homeZ, 0)[0]
            self.picking = False
        else:
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, positions[0], positions[1], itemHeight, 0)[0]
            self.picking = True
        self.commandDelay(lastIndex)

    #Runs the conveyor belt at a specified speed for a given duration
    def runConveyor(self, speed_mm_per_s: float, duration_ms: int = 5000):
        STEP_PER_CIRCLE = 360.0 / 1.8 * 5.0 * 16.0
        MM_PER_CIRCLE = 3.1415926535898 * 32.0
        vel = float(speed_mm_per_s) * STEP_PER_CIRCLE / MM_PER_CIRCLE
        try:
            dType.SetEMotor(self.api, 0, 1, int(vel), True)
            dType.dSleep(duration_ms)
        finally:
            dType.SetEMotor(self.api, 0, 1, 0, True)

    def getPose(self):
        """
        Returns the current pose (x, y, z) as a tuple of floats.
        """
        return dType.GetPose(self.api)[:4]  # Returns (x, y, z)



    def getGripperStatus(self):
        enableCtrl = c_int()
        isOn = c_int()
        rc = self.api.GetEndEffectorGripper(byref(enableCtrl), byref(isOn))
        if rc != 0:
            raise RuntimeError(f"Failed to get gripper status: {rc}")
        return isOn.value

    def getSuctionStatus(self):
        enableCtrl = c_int()
        isOn = c_int()
        rc = self.api.GetEndEffectorSuctionCup(byref(enableCtrl), byref(isOn))
        if rc != 0:
            raise RuntimeError(f"Failed to get suction status: {rc}")
        return isOn.value



    """ EXPERIMENTAL FUNCTIONS, DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING """

    #Enables or disables the built-in infrared sensor
    def setInfrared(self, on: bool, port: int = 0) -> None:
        rc = self.api.SetInfraredSensor(
            c_bool(on),
            c_uint8(port)
        )
        if rc != 0:
            raise RuntimeError(f"IR set failed: {rc}")

    #Reads the raw infrared sensor value
    def getInfrared(self, port: int = 0) -> int:
        val = c_ubyte(0)
        rc = self.api.GetInfraredSensor(
            c_uint8(port),
            byref(val)
        )
        if rc != 0:
            raise RuntimeError(f"IR get failed: {rc}")
        return val.value

    def getColorSensor(self) -> tuple[int, int, int]:
        # kick the Dobot’s internal loop
        dType.PeriodicTask(self.api)
        # optional tiny sleep to let the ADC finish
        dType.dSleep(100)

        r, g, b = dType.GetColorSensor(iself.api)
        return (r, g, b)

    def setColorSensor(self, on: bool) -> None:
        """
        Turn the built-in color sensor on or off.
        """
        # This free-function wrapper will retry internally until success.
        dType.SetColorSensor(self.api, on)

    def testHandHoldTeaching(self):        
    
        STEP_PER_CIRCLE = 360.0 / 1.8 * 5.0 * 16.0
        MM_PER_CIRCLE = 3.1415926535898 * 32.0
        vel = float(10.0) * STEP_PER_CIRCLE / MM_PER_CIRCLE # speed pulses required for a conveyor speed of 10.0[mm/s]
        #vel = float(0) * STEP_PER_CIRCLE / MM_PER_CIRCLE # speed pulses required for a conveyor speed of 0[mm/s]
        #dType.SetEMotor(self.api, 0, 1, int(vel), True)
        dType.SetEMotor(self.api, 0, 1, 1600, True)
        dType.dSleep(5000)

    def testPIRSensor(self):
        dType.SetIOMultiplexing(self.api, 15, 3, 1)
        while (True):
            PIR_status = dType.GetIODI(self.api, 15)
            print(PIR_status)
            dType.dSleep(500) 

    """ END OF EXPERIMENTAL FUNCTIONS """
