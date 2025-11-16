# Created by Hugo Nolte for course PA1414 - DoBot Magician Project
# 2019

import os
import sys
import time
import threading
import signal
import fcntl
import DoBotArm as Dbt

# --- Robust Ctrl+C handling during time.sleep and blocking I/O ---
def handle_interrupt(signum, frame):
    print("\n[!] KeyboardInterrupt detected. Exiting now.")
    sys.exit(0)

# Setup wakeup pipe to ensure signal interrupts blocking calls like sleep()
rfd, wfd = os.pipe()
flags = fcntl.fcntl(wfd, fcntl.F_GETFL)
fcntl.fcntl(wfd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
signal.set_wakeup_fd(wfd)
signal.signal(signal.SIGINT, handle_interrupt)

def test():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ)
    ctrlBot.testHandHoldTeaching()

def testConveyor():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ)
    ctrlBot.runConveyor(10.0, 5000)
    ctrlBot.runConveyor(0.0, 5000)

def testGripper():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ)
    ctrlBot.toggleGripper()
    time.sleep(2)
    ctrlBot.toggleGripper()
    time.sleep(2)
    ctrlBot.dobotDisconnect()

def test_ir_all_ports(duration_per_port: float = 5.0, interval: float = 0.5):
    bot = Dbt.DoBotArm(250, 0, 50)
    print("Connecting and enabling IR sensor loop...")
    try:
        for port in Dbt.dType.InfraredPort:
            print(f"--- Reading port {port.name} for {duration_per_port}s ---")
            bot.setInfrared(True, port)
            start = time.time()
            while time.time() - start < duration_per_port:
                val = bot.getInfrared(port)
                print(f"{port.name}: {val}")
                time.sleep(interval)
            bot.setInfrared(False, port)
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt during IR test.")
    finally:
        print("Cleaning up IR test and disconnecting.")
        bot.setInfrared(False)
        bot.dobotDisconnect()
        sys.exit(0)

def testIR():
    homeX, homeY, homeZ = 250, 0, 50
    bot = Dbt.DoBotArm(homeX, homeY, homeZ)
    ports = list(Dbt.dType.InfraredPort)
    try:
        while True:
            for port in ports:
                print(f"\n-- Testing port {port.name} ({port.value}) --")
                bot.setInfrared(True, port)
                time.sleep(0.5)
                val = bot.getInfrared(port)
                print(f"Infrared[{port.name}] = {val}")
                bot.setInfrared(False, port)
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[!] Manual IR test interrupted.")
    finally:
        print("Cleaning up...")
        bot.setInfrared(False)
        bot.dobotDisconnect()
        sys.exit(0)

def testColorSensor():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ)
    status = ctrlBot.dobotConnect()
    print("Connect status:", status)
    try:
        print("Enabling color sensor...")
        ctrlBot.setColorSensor(True)
        for i in range(5):
            rgb = ctrlBot.getColorSensor()
            print(f"Reading {i+1}: R={rgb[0]}  G={rgb[1]}  B={rgb[2]}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[!] Color sensor test interrupted.")
    finally:
        print("Disabling color sensor and disconnecting…")
        ctrlBot.setColorSensor(False)
        ctrlBot.dobotDisconnect()
        sys.exit(0)

def functions():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ)
    ctrlBot.moveArmXY(250, 100)
    ctrlBot.pickToggle(-40)
    ctrlBot.toggleSuction()
    ctrlBot.pickToggle(-40)
    ctrlBot.moveHome()
    ctrlBot.pickToggle(-40)
    ctrlBot.toggleSuction()
    ctrlBot.pickToggle(-40)

def manualMode():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ)
    print("---Manual Mode---")
    print("move to move to location")
    print("pick - toggles picking at certain height")
    print("suct - toggles suction on and off")
    print("q - exit manual mode")
    try:
        while True:
            inputCoords = input("$ ")
            inputCoords = inputCoords.split(",")
            if(inputCoords[0] == "move"):
                x = int(inputCoords[1])
                y = int(inputCoords[2])
                ctrlBot.moveArmXY(x,y)
            elif(inputCoords[0] == "pick"):
                height = int(inputCoords[1])
                ctrlBot.pickToggle(height)
            elif(inputCoords[0] == "suct"):
                ctrlBot.toggleSuction()
            elif(inputCoords[0] == "q"):
                break
            else:
                print("Unrecognized command")
    except KeyboardInterrupt:
        print("\n[!] Manual mode interrupted.")
    finally:
        ctrlBot.dobotDisconnect()
        sys.exit(0)

#--Main Program--
def main():
    try:
        #manualMode()
        #functions()
        #test()
        #testConveyor()
        #testGripper()
        #testIR()
        test_ir_all_ports()
        #testColorSensor()
    except KeyboardInterrupt:
        print("\n[!] Exiting main on keyboard interrupt.")
        sys.exit(0)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

main()
