# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/15 14:02
desc:使用UKF+互补滤波算法进行内置式磁定位
'''
import datetime

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.stats import plot_covariance

from mahonyPredictor import MahonyPredictor
from predictorViewer import plotP, plotPos, q2R
from mainLM import h

SLAVES = 2
MOMENT = 2169
DISTANCE = 0.02
SENSORLOC = np.array([[0, 0, DISTANCE]]).T
EPM = np.array([[0, 0, 1]]).T

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class InMagPredictor:
    stateNum = 10  # x, y, z, q0, q1, q2, q3, wx, wy, wz
    points = MerweScaledSigmaPoints(n=stateNum, alpha=0.3, beta=2., kappa=3-stateNum)
    dt = 0.03     # 时间间隔[s]
    t0 = datetime.datetime.now()  # 初始化时间戳

    def __init__(self):
        self.mp = MahonyPredictor()

        self.ukf = UKF(dim_x=self.stateNum, dim_z=SLAVES*3, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
        self.ukf.x = np.array([0, 0, -0.5, 1, 0, 0, 0, 0, 0, 0])   # 初始值
        self.em = self.q2R(self.ukf.x[3: 7])[-1]

        self.ukf.P = np.eye(self.stateNum) * 0.01
        for i in range(3, 7):
            self.ukf.P[i, i] = 0.2
        for i in range(7, self.stateNum):
            self.ukf.P[i, i] = 0.002

        self.ukf.Q = np.eye(self.stateNum) * 0.01 * self.dt  # 将速度作为过程噪声来源，Qi = [v*dt]
        for i in range(3, 7):
            self.ukf.Q[i, i] = 0.5  # 四元数的过程误差
        for i in range(7, self.stateNum):
            self.ukf.Q[i, i] = 0.01   # 角速度的过程误差


    def f(self, x, dt):
        wx, wy, wz = self.ukf.x[-3:]
        A = np.eye(self.stateNum)
        A[3:7, 3:7] = np.eye(4) + 0.5 * dt * np.array([[0, -wx, -wy, -wz],
                                                       [wx, 0, wz, -wy],
                                                       [wy, -wz, 0, wx],
                                                       [wz, wy, -wx, 0]])
        return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))

    def h0(self, state):
        '''
        使用互补滤波预测的结果(q)来估算预估量的四元数
        :param state: 胶囊的位姿状态
        :return: 预估量的四元数
        '''
        H = np.zeros((7, 4))
        for i in range(3):
            H[i+4, i] = 0
        return np.dot(state, H)

    def q2R(self, q):
        '''
        从四元数求旋转矩阵
        :param q: 四元数
        :return: R 旋转矩阵 (3, 3)
        '''
        q0, q1, q2, q3 = q / np.linalg.norm(q)
        R = np.array([
            [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
            [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
            [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2]
        ])
        return R

    def h(self, state):
        '''
        以外部大磁体为参考系，得出胶囊内sensor的读数
        :param state: 胶囊的位姿状态
        :param EPM: 外部大磁体的朝向
        :return: 两个sensor的读数 (6, )
        '''
        EPMNorm = np.linalg.norm(EPM)
        eEPM = EPM / EPMNorm

        pos, q = state[0: 3], state[3: 7]
        R = q2R(q)
        d = np.dot(R.T, SENSORLOC * 0.5)  # 胶囊坐标系下的sensor位置矢量转换到EPM坐标系

        r1 = pos.reshape(3, 1) + d
        r2 = pos.reshape(3, 1) - d
        r1Norm = np.linalg.norm(r1)
        r2Norm = np.linalg.norm(r2)
        er1 = r1 / r1Norm
        er2 = r2 / r2Norm

        # 每个sensor的B值[mGs]
        B1 = MOMENT * np.dot(r1Norm ** (-3), np.subtract(3 * np.dot(np.inner(er1, eEPM), er1), eEPM))
        B2 = MOMENT * np.dot(r2Norm ** (-3), np.subtract(3 * np.dot(np.inner(er2, eEPM), er2), eEPM))

        B1s = np.dot(R, B1)
        B2s = np.dot(R, B2)

        return np.vstack((B1s, B2s)).reshape(-1)

    def run(self, z):
        pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        timeCost = (datetime.datetime.now() - self.t0).total_seconds()
        self.em = q2R(self.ukf.x[3: 7])[-1]
        print(r'pos={}m, em={}, w={}, timeCost={:.3f}s'.format(pos, np.round(self.em, 3),  np.round(self.ukf.x[-3:], 3), timeCost))
        self.t0 = datetime.datetime.now()

        # 使用IMU的数据更新滤波器
        for i in range(20):
            self.mp.IMUupdate(z[6: 9], z[-3:])
            emq = q2R(self.mp.q)[-1]
            # print(r'IMU update: pos={}m, em={}, w={}'.format(pos, np.round(emq, 3), np.round(z[6: 9], 3)))
        self.ukf.x[3: 7] = self.mp.q
        # self.ukf.x[7:] = z[6: 9]

        # 使用磁传感器的数据更新滤波器
        self.ukf.predict()
        self.ukf.R = np.eye(SLAVES*3) * 5
        self.ukf.update(z[:6])


    def generate_data(self, num_data, state):
        """
        生成模拟数据
        :param num_data: 数据维度
        :param state: 胶囊的位姿状态
        :return: 模拟的B值, (num_data, )
        """
        Bmid = self.h(state[:7])  # 模拟数据的中间值
        std = 3
        Bsim = np.zeros(num_data)
        for j in range(num_data):
            Bsim[j] = np.random.normal(Bmid[j], std, 1)

        em0 = q2R(state[3: 7])[-1]
        accSim = em0
        gyroSim = np.array([0, 0, 0])
        print('truth: pos={}m, e_moment={}\n'.format(state[:3], np.round(em0, 3)))
        # return Bsim
        return np.concatenate((Bsim, gyroSim, accSim))

    def sim(self, plotType):
        state = np.array([0.1, 0.1, -0.4, 1, 1, 0, 0, 0, 0, 0])
        simData = self.generate_data(SLAVES*3, state)

        n = 20
        for i in range(n):
            print('=========={}=========='.format(i))
            plt.ion()
            plotP(self, state, i, plotType)
            if i == n - 1:
                plt.ioff()
                plt.show()

            self.run(simData)


if __name__ == '__main__':
    pr = InMagPredictor()
    pr.sim(plotType=(1, 2))
