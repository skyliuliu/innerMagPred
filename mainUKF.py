# coding=utf-8
# /usr/bin/env python
"""
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/15 14:02
desc:使用UKF+互补滤波算法进行内置式磁定位
"""
import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.stats import plot_covariance

from mahonyPredictor import MahonyPredictor
from predictorViewer import plotP, plotPos, q2R, plotErr
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
    measureNum = SLAVES * 3 + 3  # sensor*3 + IMU
    points = MerweScaledSigmaPoints(n=stateNum, alpha=0.3, beta=2., kappa=3 - stateNum)
    dt = 0.03  # 时间间隔[s]
    t0 = datetime.datetime.now()  # 初始化时间戳

    def __init__(self, sensor_std, state0, state):
        """
        初始化UKF滤波器
        :param sensor_std:【float】sensor噪声标准差 [mG]
        :param state0:【np.array】初始值 (stateNum,)
        :param state: 【np.array】真实值 (stateNum,)
        """
        # self.mp = MahonyPredictor()

        self.ukf = UKF(dim_x=self.stateNum, dim_z=self.measureNum, dt=self.dt, points=self.points, fx=self.f, hx=h)
        self.ukf.x = state0.copy()  # 初始值
        q0i, q1i, q2i, q3i = state0[3: 7]
        q0, q1, q2, q3 = state[3: 7]

        for i in range(6):
            self.ukf.R[i, i] = sensor_std
        for i in range(7, self.measureNum):
            self.ukf.R[i, i] = 0.001

        self.ukf.P = np.eye(self.stateNum) * 0.001
        for i in range(3):
            self.ukf.P[i, i] = 1.5 * (state0[i] - state[i]) ** 2  # 位置初始值的误差
        self.ukf.P[3: 7, 3: 7] = 1.5 * np.array([   # 姿态四元数初始值的误差
            [(q0i - q0) ** 2,         (q0i - q0) * (q1i - q1), (q0i - q0) * (q2i - q2), (q0i - q0) * (q3i - q3)],
            [(q0i - q0) * (q1i - q1), (q1i - q1) ** 2,         0,                       0],
            [(q0i - q0) * (q2i - q2), 0,                       (q2i - q2) ** 2,         0],
            [(q0i - q0) * (q3i - q3), 0,                       0,                       (q3i - q3) ** 2]
                                                ])
        self.ukf.P += np.eye(self.stateNum) * 0.0001

        self.ukf.Q = np.eye(self.stateNum) * 0.05 * self.dt  # 将速度作为过程噪声来源，Qi = [v*dt]
        Qqii, Qqij = 0.05, 0.005
        self.ukf.Q[3: 7, 3: 7] = np.array([   # 精细化定义姿态(四元数)的过程误差
            [Qqii, Qqij, Qqij, Qqij],
            [Qqij, Qqii, 0,     0],
            [Qqij, 0,    Qqii,  0],
            [Qqij, 0,    0,     Qqij]
        ])
        for i in range(7, self.stateNum):
            self.ukf.Q[i, i] = 0.001  # 角速度的过程误差

    def f(self, x, dt):
        wx, wy, wz = self.ukf.x[-3:]
        A = np.eye(self.stateNum)
        A[3:7, 3:7] = np.eye(4) + 0.5 * dt * np.array([[0, -wx, -wy, -wz],
                                                       [wx, 0, wz, -wy],
                                                       [wy, -wz, 0, wx],
                                                       [wz, wy, -wx, 0]])
        return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))

    def h0(self, state):
        """
        使用互补滤波预测的结果(q)来估算预估量的四元数
        :param state: 胶囊的位姿状态
        :return: 预估量的四元数
        """
        H = np.zeros((7, 4))
        for i in range(3):
            H[i + 4, i] = 0
        return np.dot(state, H)

    def run(self, z, printBool):
        pos = np.round(self.ukf.x[:3], 3)
        self.em = q2R(self.ukf.x[3: 7])[:, -1]
        timeCost = (datetime.datetime.now() - self.t0).total_seconds()
        if printBool:
            print(r'pos={}m, em={}, w={}, timeCost={:.3f}s'.format(pos, np.round(self.em, 3),
                                                                   np.round(self.ukf.x[-3:], 3), timeCost))
        self.t0 = datetime.datetime.now()

        # 使用IMU的数据更新滤波器
        # for i in range(20):
        #     self.mp.IMUupdate(z[6: 9], z[-3:])
        #     emq = q2R(self.mp.q)[-1]
        # print(r'IMU update: pos={}m, em={}, w={}'.format(pos, np.round(emq, 3), np.round(z[6: 9], 3)))
        # self.ukf.x[3: 7] = self.mp.q
        # self.ukf.x[7:] = z[6: 9]

        # 使用磁传感器的数据更新滤波器
        self.ukf.predict()
        self.ukf.update(z)

    def generate_data(self, state, sensor_std, printBool):
        """
        生成模拟数据
        :param state: 【np.array】模拟的胶囊状态 (m,)
        :param sensor_std：【float】磁传感器的噪声标准差 [mG]
        :param printBool: 【bool】是否打印输出
        :return: 【np.array】模拟的B值 + 加速度计的数值, (num_data, )
        """
        Bmid = h(state)[:-3]  # 模拟B值数据的中间值
        Bsim = np.zeros(SLAVES * 3)
        for j in range(SLAVES * 3):
            Bsim[j] = np.random.normal(Bmid[j], sensor_std, 1)

        R = q2R(state[3: 7])
        accSim = R[:, -1]
        if printBool:
            print('Bmid={}'.format(np.round(Bmid, 0)))
            print('Bsim={}'.format(np.round(Bsim, 0)))
            print('truth: pos={}m, e_moment={}\n'.format(state[:3], np.round(accSim, 3)))
        return np.concatenate((Bsim, accSim))

    def sim(self, states, sensor_std, plotType, plotBool, printBool, maxIter=20):
        """
        使用模拟的观测值验证算法的准确性
        :param states: 【list】模拟的真实状态，可以有多个不同的状态
        :param sensor_std: 【float】sensor的噪声标准差[mG]
        :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
        :param plotBool: 【bool】是否绘图
        :param printBool: 【bool】是否打印输出
        :param maxIter: 【int】最大迭代次数
        :return: 【tuple】 位置[x, y, z]和姿态四元数[q0, q1, q2, q3]的误差百分比
        """
        # self.ukf.x = state0.copy()  # 初始值
        state = states[0]  # 真实值
        simData = self.generate_data(state, sensor_std, printBool)

        err_pos, err_em = (0, 0)
        for i in range(maxIter):
            if plotBool:
                print('=========={}=========='.format(i))
                plt.ion()
                plotP(self, state, i, plotType)
                if i == maxIter - 1:
                    plt.ioff()
                    plt.show()
            self.run(simData, printBool)

            posTruth, emTruth = state[:3], q2R(state[3: 7])[:, -1]
            pos, em = self.ukf.x[:3], q2R(self.ukf.x[3: 7])[:, -1]
            err_pos = np.linalg.norm(pos - posTruth) / np.linalg.norm(posTruth)
            err_em = np.linalg.norm(em - emTruth)     # 方向矢量本身是归一化的
        print('err_pos={:.0%}, err_em={:.0%}'.format(err_pos, err_em))
        return (err_pos, err_em)

def simErrDistributed(contourBar, sensor_std=10, pos_or_ori=1):
    """
    模拟误差分布
    :param contourBar: 【np.array】等高线的刻度条
    :param sensor_std: 【float】sensor的噪声标准差[mG]
    :param pos_or_ori: 【int】选择哪个输出 0：位置，1：姿态
    :return:
    """
    n = 20
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, n), np.linspace(-0.1, 0.1, n))
    state0Dist = np.array([0, 0, -0.4, 1, 0, 0, 0, 0, 0, 0])
    statesDist = [np.array([0, 0, -0.4, 0.5 * math.sqrt(3), 0.5, 0, 0, 0, 0, 0])]
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            state0Dist[0] = x[i, j]
            state0Dist[1] = y[i, j]
            pr = InMagPredictor(sensor_std, state0Dist, statesDist[0])
            z[i, j] = pr.sim(statesDist, sensor_std, plotBool=False, plotType=(0, 1), printBool=False)[pos_or_ori]

    plotErr(x, y, z, contourBar, titleName='sensor_std={}'.format(sensor_std))


if __name__ == '__main__':
    sensor_std = 10
    state0 = np.array([0.15, 0.15, -0.4, 1, 0, 0, 0, 0, 0, 0])  # 初始值
    states = [np.array([0, 0, -0.4, 0.5 * math.sqrt(3), 0.5, 0, 0])]  # 真实值
    # pr = InMagPredictor(sensor_std, state0, states[0])
    # err = pr.sim(states, plotType=(1, 2), sensor_std=sensor_std, printBool=True, plotBool=True)

    simErrDistributed(contourBar=np.linspace(0, 0.1, 11) ,sensor_std=sensor_std, pos_or_ori=0)
