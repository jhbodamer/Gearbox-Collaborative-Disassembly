# ~/Code/test/Dobot/conveyorAndSensor.py

import time
from DoBotArm import DoBotArm
import DobotDllType as dType

homeX, homeY, homeZ = 250, 0, 50
bot = DoBotArm(homeX, homeY, homeZ)

try:
    port = dType.InfraredPort.PORT_GP1  # Try PORT_GP2 if needed
    print(f"Enabling IR sensor on port {port}")
    bot.setInfrared(True, port)

    print("Running conveyor and monitoring IR sensor...")
    bot.runConveyor(10.0, 30000)  # Run conveyor for 30 seconds max

    timeout = time.time() + 30  # Failsafe 30s timeout
    while time.time() < timeout:
        value = bot.getInfrared(port)
        if value == 1:  # Object detected
            print("[✓] Object detected by IR sensor!")
            break
        else:
            print("[ ] No object yet...")
        time.sleep(0.2)
    else:
        print("[-] Timeout: No object detected.")

finally:
    print("Stopping conveyor and disabling sensor...")
    bot.runConveyor(0.0, 500)  # stop conveyor quickly
    bot.setInfrared(False, port)
