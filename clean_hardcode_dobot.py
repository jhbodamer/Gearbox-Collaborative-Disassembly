import time
import sys
from ctypes import *
sys.path.insert(1,'./DLL')
import DobotDllType as dType
from DoBotArm import DoBotArm

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

test = DoBotArm(250,0,50)
test.dobotConnect()

#Start point

test.moveArmXYZ(195.34349060058594, -29.452899932861328, 59.18818664550781, -8.574182510375977)

#first gear
test.moveArmXYZ(182.22520446777344, -9.129701614379883, -0.0670967698097229, -2.8681890964508057)

test.gripperClose()
#dropoff point

test.moveArmXYZ(30.6605281829834, 210.17864990234375, 81.93257141113281, 81.70032501220703)
test.gripperOpen()

#reset to starting position

test.moveArmXYZ(195.34349060058594, -29.452899932861328, 59.18818664550781, -8.574182510375977)

test.dobotDisconnect()

