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

#test file for hard coding dobot path
#sample:
'''
test = DoBotArm(250,0,50)
test.moveArmXYZ(250, 200, 150, 30)
test.dobotConnect()
test.gripperOpen()
time.sleep(1)
test.gripperClose()
time.sleep(1)
test.gripperOff()
time.sleep(1)
test.moveArmXYZ(195.14817810
test.moveArmXYZ(30.66052818298058594, 190.302978515625, 127.11076354980469, 44.27981948852539)
time.sleep(1)
'''

test = DoBotArm(250,0,50)
test.dobotConnect()
test.moveArmXYZ(199.98040771484375, 0, 16.383460998535156, 0)
test.moveArmXYZ(199.98040771484375, -20, 16.383460998535156, 0)
test.moveArmXYZ(199.98040771484375, 20, 16.383460998535156, 0)
#test.dobotDisconnect()

# STEP 1: PICK OUT RIGHTMOST GEAR
#test = DoBotArm(250,0,50)
test.moveArmXYZ(199.98040771484375, 0, 16.383460998535156, 15)
test.moveArmXYZ(199.98040771484375, -13.750436782836914, 16.383460998535156, 15)
test.gripperClose()
test.moveArmXYZ(30.6605281829834, 210.17864990234375, 81.93257141113281, 81.70032501220703)
test.gripperOpen()

# STEP 2: PICK OUT GEAR CLOSEST TO THE DOBOT (WITH PEGS ON THE RIGHT)
test.moveArmXYZ(193.15679931640625, 0, 10.37519645690918, 45)
test.moveArmXYZ(193.15679931640625, 20.68450927734375, 10.37519645690918, 6.11231803894043)
test.gripperClose()
test.moveArmXYZ(30.6605281829834, 210.17864990234375, 81.93257141113281, 81.70032501220703)
test.gripperOpen()

#STEP 3: PICK OUT LAST GEAR (FARTHEST FROM THE DOBOT)

test.gripperClose()
test.moveArmXYZ(30.6605281829834, 210.17864990234375, 81.93257141113281, 81.70032501220703)
test.gripperOpen()
test.dobotDisconnect()

#dropoff [30.6605281829834, 210.17864990234375, 81.93257141113281, 81.70032501220703]
'''

if __name__ == '__main__':
	dobot = DoBotArm(250, 0, 50)
'''
