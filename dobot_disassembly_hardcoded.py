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

def dobot_disassembly_hardcoded(test: DoBotArm):
	try:
		# STEP 1: PICK OUT BIGGEST GEAR
		test.moveArmXYZ(homeX, homeY, homeZ, 75) #home position for the bot to move to every time so that it can either soft reset current position or have somewhere to go to so that
		# it wont collide with any of the gears or the box itself
		test.moveArmXYZ(235.62034606933594, -10.231431007385254, 13.101369857788086, 75) #hovering on the first gear
		test.gripperClose()
		time.sleep(2)
		test.moveArmXYZ(homeX, homeY, homeZ, 75) #moves up to home position so that it doesnt hit anything on accident
		test.moveArmXYZ(146.344970703125, 110.74940490722656, 50.16551971435547, 37.1173210144043) #moves to a nearby location that is within the work envelope because the arm is
		# unable to move outside of a certain range depending on its starting position for whatever reason
		test.gripperOpen() #lets go of the object 
		time.sleep(2)
		test.gripperOff() #turns the claw off so that it isn't running in the meantime on its way to the next gear
		# STEP 2: PICK OUT GEAR CLOSEST TO THE DOBOT (WITH PEGS ON THE RIGHT)
		test.moveArmXYZ(homeX, homeY, homeZ, 75)
		test.moveArmXYZ(199.7819061279297, 21.38075828552246, 13.565299034118652, 75)
		test.gripperClose()
		time.sleep(2)
		test.moveArmXYZ(homeX, homeY, homeZ, 75)
		test.moveArmXYZ(146.344970703125, 110.74940490722656, 50.16551971435547, 37.1173210144043)
		test.gripperOpen()
		time.sleep(2)
		test.gripperOff()
		# STEP 3: PICK OUT LAST GEAR (FARTHEST FROM THE DOBOT)
		test.moveArmXYZ(homeX, homeY, homeZ, 75)
		test.moveArmXYZ(241.5885772705078, 22.94457244873047, 9.806683540344238, 25)
		test.gripperClose()
		time.sleep(2)
		test.moveArmXYZ(homeX, homeY, homeZ, 75)
		test.moveArmXYZ(146.344970703125, 110.74940490722656, 50.16551971435547, 37.1173210144043)
		test.gripperOpen()
		time.sleep(2)
		test.gripperOff()
		
	except KeyboardInterrupt:
		print("\n[!] Stopping log...")
	finally:
		test.dobotDisconnect()
		print("[✓] Disconnected from DOBOT.")

if __name__ == '__main__':
	homeX, homeY, homeZ = 240.1073455810547, 4.5382914543151855, 133.52200317382812
	test = DoBotArm(homeX, homeY, homeZ)
	dobot_disassembly_hardcoded(test)
