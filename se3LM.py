# coding=utf-8
# /usr/bin/env python3
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/12/16 14:31
desc: 基于李代数LM非线性优化，实现内置式磁定位
'''
import datetime
import math

from Lie import *


class Predictor:
    EPM_ORI = np.array([[1, 0, 0]]).T  # 外部大磁体的N极朝向
    EPM_POS = np.array([[0, 0, 0.5]]).T  # 外部大磁体的坐标[m]
    MOMENT = 2169  # 外部大磁体的磁矩[A*m^2]

    COIL_POS = np.array([[0, 0.2, -0.2]]).T  # 下线圈的中心位置[m]
    COIL_INNER_R = 0.16 / 2   # 下线圈的内半径[m]
    COIL_TURNS = 510   # 下线圈的匝数
    COIL_LAYERS = 20   # 下线圈的层数
    COIL_OUTTER_R = 0.227 / 2   # 下线圈的外半径[m]
    COIL_CURRENT = 5   # 下线圈的电流
    COIL_d = (COIL_OUTTER_R - COIL_INNER_R) / (COIL_TURNS / COIL_LAYERS - 1)   # 线圈导线的径向间距[m]
    COIL_NR = COIL_TURNS // COIL_LAYERS   # 径向圈数
    COIL_AREA = 0
    #COIL_AREA = COIL_TURNS // COIL_LAYERS * math.pi * sum([(COIL_INNER_R + COIL_d * k) ** 2 for k in range(COIL_LAYERS)]) / 1000000

    def __init__(self, state, stateX):
        '''
        初始化定位类
        :param state: 【se3】初始位姿
        :param stateX: 【se3】真实位姿
        '''
        self.state = state
        self.stateX = stateX
        self.m = 9
        self.n = 6

        for i in range(self.COIL_LAYERS):
            self.COIL_AREA += self.COIL_NR * math.pi * (self.COIL_INNER_R + self.COIL_d * i) ** 2
        self.coil_moment = self.COIL_AREA * self.COIL_CURRENT

    def h(self, state):
        '''
        观测函数
        :return: 【np.array】观测值
        '''
        pos = state.exp().matrix()[:3, 3]
        R = state.exp().matrix()[:3, :3]

        # 外部大磁体
        EPMNorm = np.linalg.norm(self.EPM_ORI)
        eEPM = self.EPM_ORI / EPMNorm

        r = pos.reshape(3, 1) + self.EPM_POS
        rNorm = np.linalg.norm(r)
        er = r / rNorm

        # 下线圈
        eCOIL = np.array([[0, 0, 1]]).T
        rCOIL = pos.reshape(3, 1) + self.COIL_POS
        rCOILNorm = np.linalg.norm(rCOIL)
        erCOIL = rCOIL / rCOILNorm

        # EPM坐标系下sensor的B值[Gs]
        B0 = self.MOMENT * np.dot(rNorm ** (-3), np.subtract(3 * np.dot(np.inner(er, eEPM), er), eEPM)) / 1000
        Bi = self.coil_moment * np.dot(rCOILNorm ** (-3), np.subtract(3 * np.dot(np.inner(erCOIL, eCOIL), erCOIL), eCOIL)) / 1000
        # 变换到胶囊坐标系下的sensor读数
        B0s = np.dot(R, B0)
        B0s[-1] *= -1

        Bis = np.dot(R, Bi)
        Bis[-1] *= -1

        # 加速度计的读数
        a_s = R[2]  # 重力矢量在胶囊坐标系下的坐标

        return np.concatenate((a_s, B0s.reshape(-1), Bis.reshape(-1)))

    def generateData(self, std):
        '''
        生成模拟数据
        :param Std: 传感器输出值的标准差
        :return:
        '''
        midData = self.h(self.stateX)
        pos = self.stateX.exp().matrix()[:3, 3]
        ez = self.stateX.exp().matrix()[2, :3]
        print('turth: pos={}, ez={}'.format(np.round(pos, 3), np.round(ez, 3)))
        simData = np.zeros(self.m)
        for j in range(self.m):
            simData[j] = np.random.normal(midData[j], std, 1)
        self.measureData = simData
        print('sensor_mid={}'.format(np.round(midData[:], 3)))
        print('sensor_sim={}'.format(np.round(simData[:], 3)))

    def residual(self, state):
        '''
        计算残差
        :param measureData:  【np.array】观测值
        :return: 【np.array】观测值 - 预测值
        '''
        eastData = self.h(state)
        return self.measureData - eastData

    def derive(self, param_index):
        """
        指定状态量的偏导数
        :param param_index: 第几个状态量
        :return: 偏导数 (m, )
        """
        state1 = self.state.vector().copy()
        state2 = self.state.vector().copy()
        delta = 0.001
        state1[param_index] += delta
        state2[param_index] -= delta
        data_est_output1 = self.h(se3(vector=state1))
        data_est_output2 = self.h(se3(vector=state2))
        return 0.5 * (data_est_output1 - data_est_output2) / delta

    def jacobian(self):
        '''
        计算预估状态的雅可比矩阵
        :return: 【np.array (m, n)】雅可比矩阵
        '''
        J = np.zeros((self.m, self.n))
        for pi in range(self.n):
            J[:, pi] = self.derive(pi)
        return J

    def get_init_u(self, A, tao):
        """
        确定u的初始值
        :param A: 【np.array】 J.T * J (n, n)
        :param tao: 【float】常量
        :return: 【int】u
        """
        Aii = []
        for i in range(0, self.n):
            Aii.append(A[i, i])
        u = tao * max(Aii)
        return u

    def LM(self, maxIter, printBool):
        """
        Levenberg–Marquardt优化算法的主体
        :param maxIter: 最大迭代次数
        :return: 【np.array】优化后的状态 (7, )
        """
        tao = 1e-3
        eps_stop = 1e-9
        eps_step = 1e-6
        eps_residual = 1e-3

        t0 = datetime.datetime.now()
        res = self.residual(self.state)
        J = self.jacobian()
        A = J.T.dot(J)
        g = J.T.dot(res)
        u = self.get_init_u(A, tao)  # set the init u
        # u = 100
        v = 2
        rou = 0
        mse = 0

        for i in range(maxIter):
            i += 1
            while True:

                Hessian_LM = A + u * np.eye(self.n)  # calculating Hessian matrix in LM
                step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
                if np.linalg.norm(step) <= eps_step:
                    self.stateOut(t0, i, mse, 'threshold_step', printBool)
                    return self.state
                newState = se3(vector=self.state.vector() + step)   # 先将se3转换成数组相加，再变回李代数，这样才符合LM算法流程
                newRes = self.residual(newState)
                mse = np.linalg.norm(res) ** 2
                newMse = np.linalg.norm(newRes) ** 2
                rou = (mse - newMse) / (step.T.dot(u * step + g))
                if rou > 0:
                    self.state = newState
                    res = newRes
                    J = self.jacobian()
                    A = J.T.dot(J)
                    g = J.T.dot(res)
                    u *= max(1 / 3, 1 - (2 * rou - 1) ** 3)
                    v = 2

                    stop = (np.linalg.norm(g, ord=np.inf) <= eps_stop) or (mse <= eps_residual)
                    if stop:
                        if np.linalg.norm(g, ord=np.inf) <= eps_stop:
                            self.stateOut(t0, i, mse, 'threshold_stop', printBool)
                        if mse <= eps_residual:
                            self.stateOut(t0, i, mse, 'threshold_residual', printBool)
                        return self.state
                    else:
                        break
                else:
                    u *= v
                    v *= 2

            self.stateOut(t0, i, mse, '', printBool)

    def stateOut(self, t0, i, mse, printStr, printBool):
        '''
        输出算法的中间结果
        :param t0: 【float】 时间戳
        :param i: 【int】迭代步数
        :param mse: 【float】残差
        :return:
        '''
        if not printBool:
            return
        print(printStr)
        timeCost = (datetime.datetime.now() - t0).total_seconds()
        pos = self.state.exp().matrix()[:3, 3]
        em = self.state.exp().matrix()[2, :3]
        pos = np.round(pos, 3)
        em = np.round(em, 3)
        print('i={}, pos={}m, em={}, timeConsume={:.3f}s, cost={:.2e}'.format(i, pos, em, timeCost, mse))

    def sim(self):
        '''
        使用模拟的观测值验证算法的准确性
        :return:
        '''
        measureData = self.generateData(0.01)
        state = self.LM(100, True)

        posX = self.stateX.exp().matrix()[:3, 3]
        emX = self.stateX.exp().matrix()[2, :3]
        pos = self.state.exp().matrix()[:3, 3]
        em = self.state.exp().matrix()[2, :3]
        err_pos = np.linalg.norm(pos - posX) / np.linalg.norm(posX)
        err_em = np.linalg.norm(em - emX)      # 方向矢量本身是归一化的
        print('turth: pos={}, ez={}, err_pos={:.0%}, err_em={:.0%}'.format(np.round(posX, 3), np.round(emX, 3), err_pos, err_em))

if __name__ == '__main__':
    state = se3(vector=np.array([0, 0, 0, 0, 0, 0.01]))
    stateX = se3(vector=np.array([1.04722, 0, 0, 0.2, 0.014, 0.2337]))

    pred = Predictor(state, stateX)
    pred.sim()
