#!/usr/bin/env python3
import os, platform
from ctypes import CDLL, cdll, RTLD_GLOBAL

# Load exactly the same way your DobotDllType.load() does:
base = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Windows":
    lib = CDLL(os.path.join(base, "DobotDll.dll"), RTLD_GLOBAL)
elif platform.system() == "Darwin":
    lib = CDLL(os.path.join(base, "libDobotDll.dylib"), RTLD_GLOBAL)
else:
    lib = cdll.LoadLibrary(os.path.join(base, "libDobotDll.so"))

# Declare the argtypes/restype you think are correct:
from ctypes import c_int, c_bool, c_uint8, c_uint64, POINTER, c_ubyte
lib.SetInfraredSensor.argtypes = [c_int, c_int, c_bool, c_uint8, c_uint8, c_bool, POINTER(c_uint64)]
lib.SetInfraredSensor.restype  = c_int
lib.GetInfraredSensor.argtypes = [c_int, c_int, c_uint8, POINTER(c_ubyte)]
lib.GetInfraredSensor.restype  = c_int

# Now print them back out:
print("SetInfraredSensor:")
print("  argtypes:", lib.SetInfraredSensor.argtypes)
print("  restype: ", lib.SetInfraredSensor.restype)
print()
print("GetInfraredSensor:")
print("  argtypes:", lib.GetInfraredSensor.argtypes)
print("  restype: ", lib.GetInfraredSensor.restype)
