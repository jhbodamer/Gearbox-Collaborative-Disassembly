# log_dobot_values.py
import time
from DoBotArm import DoBotArm

def log_dobot_state(bot: DoBotArm):
    try:
        while True:
            pose = bot.getPose()  # (x, y, z, r)
            gripper = bot.getGripperStatus()  # 1 = open, 0 = closed
            suction = bot.getSuctionStatus()  # 1 = on, 0 = off

            print(f"[Dobot Log] Pose: {pose}, Gripper: {'Open' if gripper else 'Closed'}, Suction: {'On' if suction else 'Off'}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Stopping log...")
    finally:
        bot.dobotDisconnect()
        print("[✓] Disconnected from DOBOT.")

if __name__ == "__main__":
    homeX, homeY, homeZ = 250, 0, 50  # default home position
    bot = DoBotArm(homeX, homeY, homeZ)
    log_dobot_state(bot)
