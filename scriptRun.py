# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/14 15:25
desc: 运行测试脚本
'''
import csv
import multiprocessing
from multiprocessing.dummy import Process
import time

import numpy as np
import matplotlib.pyplot as plt

from readData import ReadData
from mahonyPredictor import MahonyPredictor
from predictorViewer import track3D


def main():
    snesorDict = {'imu': 'LSM6DS3TR-C'}
    readObj = ReadData(snesorDict)
    # outputDataSigma = multiprocessing.Array('f', [0] * len(snesorDict) * 24)
    outputDataSigma = None
    magBg = multiprocessing.Array('f', [0] * 6)
    outputData = multiprocessing.Array('f', [0] * len(snesorDict) * 24)

    state = multiprocessing.Array('f', [0, 0, 0, 1, 0, 0, 0])

    # Wait a second to let the port initialize
    # readObj.send()
    # receive data in a new process
    pRec = Process(target=readObj.receive, args=(outputData, magBg, outputDataSigma))
    pRec.daemon = True
    pRec.start()

    pTrack3D = multiprocessing.Process(target=track3D, args=(state,))
    pTrack3D.daemon = True
    pTrack3D.start()

    mp = MahonyPredictor(q=state[3:], Kp=100, Ki=0.01, dt=0.002)
    while True:
        # print("a={}, w={}".format(np.round(outputData[:3], 2), np.round(outputData[3:6], 2)))
        mp.getGyroOffset(outputData[3:6])
        mp.IMUupdate(outputData[:3], outputData[3:6])
        state[3:] = mp.q
        time.sleep(0.08)


if __name__ == '__main__':
    main()