import os
import sys
import time
import threading
import signal
import fcntl
import DoBotArm as Dbt

def test():
    homeX, homeY, homeZ = 250, 0, 50                 #home position
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ)      #centers robot
    ctrlBot.moveArmXY(10,10)                         #moves arm           
    print(ctrlBot.getSuctionStatus())

def main():
    test()

main()
