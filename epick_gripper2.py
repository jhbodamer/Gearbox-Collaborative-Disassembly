import socket
import time

# Socket settings
HOST = "192.168.1.5"  
PORT = 63352         
GRIPPER_ID = 9        

def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(f"sid{GRIPPER_ID}\n".encode())  # Select gripper
        s.sendall(command.encode())  # Send command
        response = s.recv(1024).decode().strip()
    
    #print(f"Response for '{command.strip()}': {response}")
    return response

def start_suction():
    send_command("SET ACT 1\n") 
    time.sleep(1)
    send_command("SET MOD 0\n") 
    #print("ePick Suction is now ON.")

def stop_suction():
    send_command("SET ATR 1\n")
    #print("ePick Suction is now OFF and the object is released.")

if __name__ == "__main__":
    start_suction()
    time.sleep(5)  
    stop_suction()
