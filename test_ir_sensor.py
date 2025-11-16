import DobotDllType as dType
from DobotDllType import InfraredPort

api = dType.load()
result = dType.ConnectDobot(api, "/dev/ttyACM0", 115200)  # Use actual device path
print("Connect result:", result)

if result[0] == 0:
    try:
        dType.SetInfraredSensor(api, InfraredPort.PORT_GP1, True, 0, False)
        print("Infrared sensor enabled.")

        val = dType.GetInfraredSensor(api, InfraredPort.PORT_GP1)
        print("Infrared sensor value:", val)

    except Exception as e:
        print("Error:", e)

    dType.DisconnectDobot(api)
else:
    print("Failed to connect.")
