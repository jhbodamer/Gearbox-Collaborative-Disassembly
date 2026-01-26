from DoBotArm import *

test = DoBotArm(1,1,1)

# Move joints: base, shoulder, elbow, wrist
test.moveArmJointAngles(1, 40, 0, 90)
time.sleep(3)
for x in range(0,10):
	test.moveJoint4(-120)
	test.moveJoint4(120)
