''' This is a file to demonstrate using some of the 
	functions of the RobotiqGripper class contained
	in AdaptiveGripperControl.py. Note that the 
	activate() method is used to clear flags and get 
	the gripper going after a power cycle.
	Created 1/18/2026
'''

import time

# import the class
from AdaptiveGripperControl import RobotiqGripper

# create the object instance
gripper = RobotiqGripper()

# these parameters for our gripper
HOST = "192.168.1.5"  
PORT = 63352   

# these two should probably always go together
gripper.connect()
gripper.activate()
print("Gripper activating, this might cause it to close and open")

# this automatically finds the true range of the gripper and stores them
gripper.auto_calibrate()

# these two can be used to check if the gripper is fully open or closed
# print(gripper.is_open())
# print(gripper.is_closed())

''' there are two functions for movement. One pauses the code to wait for the 
movement to execute and the other does not. Most of the time the one which
pauses will be better. If it is picking up and object and it cant get to 
the fully closed position it will know this and stop early

The parameters of each are 
(position (0-255), speed(0-255), force(0-255))'''

gripper.move_and_wait_for_pos(0, 100, 100) # fully open
print(f"\ngripper open\n")
time.sleep(1)

print(f"closing with increasing speed\n")
for i in range(255):
	gripper.move(255, i, 0)
	time.sleep(0.005)

time.sleep(1)
print(f"opening with increasing speed\n")
for i in range(255):
	gripper.move(0, i, 0)
	time.sleep(0.005)

time.sleep(1)
print(f"closing at max speed\n")
gripper.move_and_wait_for_pos(255 , 255, 0)
print(f"opening at max speed\n")
gripper.move_and_wait_for_pos(0 , 255, 0)
print(f"closing at min speed\n")
gripper.move_and_wait_for_pos(255 , 0, 0)
print(f"opening at min speed\n")
gripper.move_and_wait_for_pos(0 , 0, 0)

# lastly i wrote these if you don't need the finer control
gripper.close()
gripper.open()

print("demo complete")



