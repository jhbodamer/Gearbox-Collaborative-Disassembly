import socket
import time
from AdaptiveGripperControl import RobotiqGripper

# Socket settings
HOST = "192.168.1.5"  
PORT = 63352         
GRIPPER_ID = 9   
     
gripper = RobotiqGripper()
gripper.connect(HOST, PORT)
gripper.activate()
gripper.auto_calibrate()
for i in range(5):
    gripper.move_and_wait_for_pos(255, 0, 0)
    gripper.move_and_wait_for_pos(0, 0, 0)
time.sleep(7)
time.sleep(2)
gripper.disconnect()

# WRITE VARIABLES (CAN ALSO READ)
ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
ATR = 'ATR'  # atr : auto-release (emergency slow move)
ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
FOR = 'FOR'  # for : force (0-255)
SPE = 'SPE'  # spe : speed (0-255)
POS = 'POS'  # pos : position (0-255), 0 = open
# READ VARIABLES
STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
PRE = 'PRE'  # position request (echo of last commanded position)
OBJ = 'OBJ'  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)
    
def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(f"sid{GRIPPER_ID}\n".encode())  # Select gripper
        s.sendall(command.encode())  # Send command
        response = s.recv(1024).decode().strip()
    
    #print(f"Response for '{command.strip()}': {response}")
    return response

#opens the gripper
def open_gripper():

    # send_command("SET ACT 1\n")

    print("gripper activated")
    # send_command("SET SPE 255 \n")
    # time.sleep(2)
    # send_command("SET POS 0\n")
    # send_command("SET GTO 1\n")
	
def close_gripper():
    # time.sleep(5)
    # print("closing gripper")
    # send_command("SET POS 200\n")
    # send_command("SET SPE 0\n")
    # send_command("SET GTO 1\n")
    # time.sleep(5)
    print("gripper closed")
	
	
#def start_suction():
    '''
    send_command("SET ACT 1\n") 
    time.sleep(1)
    #send_command("SET MOD 0\n") 
    #print("ePick Suction is now ON.")
    '''

#def stop_suction():
    '''
    send_command("SET SPE 0\n")
    send_command("SET ACT 0\n")
    #print("ePick Suction is now OFF and the object is released.")
    '''

if __name__ == "__main__":
	
	open_gripper()
	
	close_gripper()
	'''
    print("starting suction")
    start_suction()
    time.sleep(1) 
    print("done starting suction")

    #stop_suction()
    #print("done stopping suction")
    time.sleep(2)
    print("Starting suction again")
    start_suction()
    '''
