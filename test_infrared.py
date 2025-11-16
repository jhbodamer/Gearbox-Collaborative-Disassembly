#!/usr/bin/env python3
"""
test_infrared.py
Simple test harness for the DoBot Magician IR sensor (two-arg API)
"""
import sys
import time

# adjust import path if necessary
from DoBotArm import DoBotArm
from DobotDllType import InfraredPort


def test_ir_all_ports(duration_per_port: float = 5.0, interval: float = 0.5):
    """
    Cycle through each InfraredPort, reading values for duration_per_port seconds.
    Uses the two-argument Set/GetInfraredSensor API.
    """
    # Initialize arm at home position
    bot = DoBotArm(250, 0, 50)
    print("Connected to DoBot. Beginning IR port test...")

    ports = [InfraredPort.PORT_GP1.value,
             InfraredPort.PORT_GP2.value,
             InfraredPort.PORT_GP4.value,
             InfraredPort.PORT_GP5.value]

    try:
        for port in ports:
            print(f"\n--- Testing IR port {port} for {duration_per_port} seconds ---")
            # Enable sensor on this port
            bot.setInfrared(True, port)
            start = time.time()
            while time.time() - start < duration_per_port:
                val = bot.getInfrared(port)
                print(f"Port {port}: {val}")
                time.sleep(interval)
            # Disable before moving on
            bot.setInfrared(False, port)

    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt detected. Exiting test.")
    finally:
        # ensure sensor off and clean disconnect
        print("Disabling IR and disconnecting...")
        try:
            bot.setInfrared(False, ports[0])
        except Exception:
            pass
        bot.dobotDisconnect()
        sys.exit(0)


if __name__ == '__main__':
    test_ir_all_ports()
