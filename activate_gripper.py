# Used after UR5e power cycle to reactivate the suction gripper
import socket
import time

HOST = "192.168.1.5"  # IP
PORT = 29999            # Dashboard server port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

def pulse_gripper():
	# Load activation program which is on the UR5e
	s.sendall(b"load GripperInit.urp\n")
	time.sleep(1)
	s.sendall(b"play\n")

	time.sleep(0.05)
	s.sendall(b"stop\n")

	s.close()

pulse_gripper()
