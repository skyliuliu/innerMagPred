# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/11/6 17:07
desc: 使用EKF进行IMU的位姿解算
'''
import multiprocessing
from multiprocessing.dummy import Process
import time

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

from readData import ReadData
from predictorViewer import track3D


class QEKF(ExtendedKalmanFilter):
    def __init__(self):
        super().__init__(4, 3)
        self.dt = 0.001
        # self.w = np.zeros(3)
        # self.bw = np.zeros(3)
        self.wb = np.ones(3)

        self.x = np.array([1, 0, 0, 0])
        self.P = np.eye(self.dim_x) * 0.5
        self.Q = np.eye(self.dim_x) * 0.5
        self.R = np.eye(self.dim_z) * 0.1

        self.F = self.Fx(self.dt, self.wb)

    def Fx(self, dt, wb):
        return np.array([[1,           -0.5*dt*wb[0], -0.5*dt*wb[1], -0.5*dt*wb[2]],
                         [0.5*dt*wb[0], 1,             0.5*dt*wb[2], -0.5*dt*wb[1]],
                         [0.5*dt*wb[1], -0.5*dt*wb[2], 1,             0.5*dt*wb[0]],
                         [0.5*dt*wb[2], 0.5*dt*wb[1],  -0.5*dt*wb[0], 1]])


def Hx(x, *args):
    q = x[:]
    return np.array([2*q[1]*q[3] - 2*q[0]*q[2],
                     2*q[2]*q[3] + 2*q[0]*q[1],
                     q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]])

def HJacobian(x, *args):
    q = x[:]
    return 2 * np.array([[-q[2], q[3], -q[0], q[1]],
                         [q[1],  q[0], q[3],  q[2]],
                         [q[0], -q[1], -q[2], q[3]]])

def main():
    snesorDict = {'imu': 'LSM6DS3TR-C'}
    readObj = ReadData(snesorDict)
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
    time.sleep(0.5)

    pTrack3D = multiprocessing.Process(target=track3D, args=(state,))
    pTrack3D.daemon = True
    pTrack3D.start()

    i = 0
    bw = np.zeros(3)
    qEKF = QEKF()
    while True:
        for j in range(4):
            # print("w={}".format(np.round(outputData[3+6*j:6*(j+1)], 2)))
            if i < 100:
                bw += outputData[3+6*j: 6*(j+1)]
                i += 1
                if i == 100:
                    bw /= i
                    qEKF.bw = bw
                    print("get gyroscope bias:{}deg/s".format(bw))
            else:
                w = outputData[3+6*j: 6*(j+1)]
                wb = w - bw
                qEKF.F = qEKF.Fx(qEKF.dt, wb)
                print('time={:.4f}: wb={}, q={}'.format(time.time(), np.round(qEKF.wb, 2), np.round(qEKF.x, 3)))
                qEKF.predict()
                qNorm = np.linalg.norm(qEKF.x)
                qEKF.x = qEKF.x / qNorm
                state[3: 7] = qEKF.x[:]

                aNorm = np.linalg.norm(outputData[6 * j: 6 * j + 3])
                qEKF.z = np.array(outputData[6 * j: 6 * j + 3]) / aNorm
                qEKF.update(qEKF.z, HJacobian, Hx, qEKF.R)
                qNorm = np.linalg.norm(qEKF.x)
                qEKF.x = qEKF.x / qNorm
                state[3: 7] = qEKF.x[:]
            time.sleep(0.037)

if __name__ == '__main__':
    main()