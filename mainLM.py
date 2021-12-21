# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/17 10:24
desc:使用LM非线性优化+互补滤波算法进行内置式磁定位
'''
import datetime
import math
import multiprocessing
from multiprocessing.dummy import Process
import time

import numpy as np
import matplotlib.pyplot as plt

from predictorViewer import q2R, q2Euler, plotErr, plotLM, plotPos, track3D
from readData import ReadData

tao = 1e-3
delta = 0.0001
eps_stop = 1e-9
eps_step = 1e-6
eps_residual = 0.001
residual_memory = []
us = []
poss = []
ems = []

SLAVES = 2
MOMENT = 2169
g0 = 9.8
DISTANCE = 0.0138
SENSORLOC = np.array([[0, 0, DISTANCE]]).T
EPM_ORI = np.array([[0, 1, 0]]).T    # 外部大磁体的N极朝向
EPM_POS = np.array([[0, 0, 0.5]]).T  # 外部大磁体的坐标

COIL_POS = np.array([[0, 0.2, -0.2]]).T  # 下线圈的中心位置[m]
COIL_INNER_R = 0.16 / 2   # 下线圈的内半径[m]
COIL_TURNS = 510   # 下线圈的匝数
COIL_LAYERS = 20   # 下线圈的层数
COIL_OUTTER_R = 0.227 / 2   # 下线圈的外半径[m]
COIL_CURRENT = 5   # 下线圈的电流
COIL_d = (COIL_OUTTER_R - COIL_INNER_R) / (COIL_TURNS / COIL_LAYERS - 1)   # 线圈导线的径向间距[m]
COIL_NR = COIL_TURNS // COIL_LAYERS
COIL_AREA = COIL_NR * math.pi * sum([(COIL_INNER_R + COIL_d * k) ** 2 for k in range(COIL_LAYERS)])
COIL_MOMENT = COIL_AREA * COIL_CURRENT

def h(state):
    '''
    以外部大磁体为参考系，得出胶囊内sensor的读数
    :param state: 胶囊的位姿状态
    :return: 两个sensor的读数 + 胶囊z轴朝向 (9, )
    '''
    EPMNorm = np.linalg.norm(EPM_ORI)
    eEPM = EPM_ORI / EPMNorm

    pos, q = state[0: 3], state[3: 7]
    q /= np.linalg.norm(q)
    R = q2R(q)
    emz = R[:, -1]  # 重力矢量在胶囊坐标系下的坐标
    d = np.dot(R.T, SENSORLOC * 0.5)  # 胶囊坐标系下的sensor位置矢量转换到EPM坐标系

    r1 = pos.reshape(3, 1) + d + EPM_POS
    r2 = pos.reshape(3, 1) - d + EPM_POS
    r1Norm = np.linalg.norm(r1)
    r2Norm = np.linalg.norm(r2)
    er1 = r1 / r1Norm
    er2 = r2 / r2Norm

    # EPM坐标系下每个sensor的B值[Gs]
    B1 = MOMENT * np.dot(r1Norm ** (-3), np.subtract(3 * np.dot(np.inner(er1, eEPM), er1), eEPM)) / 1000
    B2 = MOMENT * np.dot(r2Norm ** (-3), np.subtract(3 * np.dot(np.inner(er2, eEPM), er2), eEPM)) / 1000
    # 变换到胶囊坐标系下的sensor读数
    B1s = np.dot(R, B1)
    B1s[-1] *= -1
    B2s = np.dot(R, B2)

    # 加速度计的读数
    a_s = emz * g0

    return np.concatenate((a_s, B1s.reshape(-1), B2s.reshape(-1)))


def h0(state):
    '''
    胶囊只有内一个磁sensor
    :param state: 胶囊的位姿状态
    :param EPM: 外部大磁体的朝向
    :return: 胶囊内imu+磁传感器的读数
    '''
    pos, q = state[0: 3], state[3: 7]
    q /= np.linalg.norm(q)
    R = q2R(q)
    emz = R[:, -1]  # 重力矢量在胶囊坐标系下的坐标

    # 外部大磁体
    EPMNorm = np.linalg.norm(EPM_ORI)
    eEPM = EPM_ORI / EPMNorm
    r = pos.reshape(3, 1) + EPM_POS
    rNorm = np.linalg.norm(r)
    er = r / rNorm

     # 下线圈
    eCOIL = np.array([[0, 0, 1]]).T
    rCOIL = pos.reshape(3, 1) + COIL_POS
    rCOILNorm = np.linalg.norm(rCOIL)
    erCOIL = rCOIL / rCOILNorm

    # EPM坐标系下sensor的B值[Gs]
    B0 = MOMENT * np.dot(rNorm ** (-3), np.subtract(3 * np.dot(np.inner(er, eEPM), er), eEPM)) / 1000
    Bi = COIL_MOMENT * np.dot(rCOILNorm ** (-3), np.subtract(3 * np.dot(np.inner(erCOIL, eCOIL), erCOIL), eCOIL)) / 1000

    # 变换到胶囊坐标系下的sensor读数
    B0s = np.dot(R, B0)
    B0s[-1] *= -1

    Bis = np.dot(R, Bi)
    Bis[-1] *= -1

    # 加速度计的读数
    a_s = emz

    return np.concatenate((a_s, B0s.reshape(-1), Bis.reshape(-1)))


def derive(state, param_index):
    """
    指定状态量的偏导数
    :param state: 预估的状态量 (n, )
    :param param_index: 第几个状态量
    :return: 偏导数 (m, )
    """
    state1 = state.copy()
    state2 = state.copy()
    if param_index < 3:
        delta = 0.0001
    else:
        delta = 0.0001
    state1[param_index] += delta
    state2[param_index] -= delta
    data_est_output1 = h0(state1)
    data_est_output2 = h0(state2)
    return 0.5 * (data_est_output1 - data_est_output2) / delta


def jacobian(state, m):
    """
    计算预估状态的雅可比矩阵
    :param state: 【np.array】预估的状态量 (n, )
    :param m: 【int】观测量的个数
    :return: 【np.array】J (m, n)
    """
    n = state.shape[0]
    J = np.zeros((m, n))
    for pi in range(0, n):
        J[:, pi] = derive(state, pi)
    return J


def residual(state, output_data):
    """
    计算残差
    :param state: 【np.array】预估的状态量 (n, )
    :param output_data: 【np.array】观测量 (m, )
    :return:【np.array】 residual (m, )
    """
    data_est_output = h0(state[:7])
    res = output_data - data_est_output
    return res


def get_init_u(A, tao):
    """
    确定u的初始值
    :param A: 【np.array】 J.T * J (n, n)
    :param tao: 【float】常量
    :return: 【int】u
    """
    m = np.shape(A)[0]
    Aii = []
    for i in range(0, m):
        Aii.append(A[i, i])
    u = tao * max(Aii)
    return u


def LM(state2, output_data, n, maxIter, printBool):
    """
    Levenberg–Marquardt优化算法的主体
    :param state2: 【np.array】预估的状态量 (n, ) + [moment, costTime]
    :param output_data: 【np.array】观测量 (m, )
    :param n: 【int】状态量的维度
    :param maxIter: 【int】最大迭代次数
    :return: None
    """
    output_data = np.array(output_data)
    state = np.array(state2)[:n]
    t0 = datetime.datetime.now()
    m = output_data.shape[0]
    res = residual(state, output_data)
    J = jacobian(state, m)
    A = J.T.dot(J)
    g = J.T.dot(res)
    u = get_init_u(A, tao)  # set the init u
    # u = 100
    v = 2
    rou = 0
    mse = 0

    for i in range(maxIter):
        poss.append(state[:3])
        ems.append(state[3:])
        i += 1
        while True:
            Hessian_LM = A + u * np.eye(n)  # calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
            if np.linalg.norm(step) <= eps_step:
                # state2[:] = state
                stateOut(state, state2, t0, i, mse, 'threshold_step', printBool)
                return
            newState = state + step
            newRes = residual(newState, output_data)
            mse = np.linalg.norm(res) ** 2
            newMse = np.linalg.norm(newRes) ** 2
            rou = (mse - newMse) / (step.T.dot(u * step + g))
            if rou > 0:
                state = newState
                res = newRes
                J = jacobian(state, m)
                A = J.T.dot(J)
                g = J.T.dot(res)
                u *= max(1 / 3, 1 - (2 * rou - 1) ** 3)
                v = 2
                stop = (np.linalg.norm(g, ord=np.inf) <= eps_stop) or (mse <= eps_residual)
                us.append(u)
                residual_memory.append(mse)
                if stop:
                    # state2[:] = state
                    if np.linalg.norm(g, ord=np.inf) <= eps_stop:
                        stateOut(state, state2, t0, i, mse, 'threshold_stop', printBool)
                    if mse <= eps_residual:
                        stateOut(state, state2, t0, i, mse, 'cost stop', printBool)
                    return
                else:
                    break
            else:
                u *= v
                v *= 2
                us.append(u)
                residual_memory.append(mse)
        # state2[:] = state
        stateOut(state, state2, t0, i, mse, ' ', printBool)


def stateOut(state, state2, t0, i, mse, printStr, printBool):
    '''
    输出算法的中间结果
    :param state:【np.array】 位置和姿态:x, y, z, q0, q1, q2, q3 (7,)
    :param state2: 【np.array】位置、姿态、磁矩、单步耗时和迭代步数 (10,)
    :param t0: 【float】 时间戳
    :param i: 【int】迭代步数
    :param mse: 【float】残差
    :return:
    '''
    if not printBool:
        return
    print(printStr)
    timeCost = (datetime.datetime.now() - t0).total_seconds()
    state2[:] = np.concatenate((state, np.array([MOMENT, timeCost, i])))  # 输出的结果
    pos = np.round(state[:3], 3)
    R = q2R(state[3:7])
    emx = np.round(R[:, 0], 3)
    emz = np.round(R[:, -1], 3)
    sensor_pred = h0(state)[:6]
    print('i={}, pos={}m, q={}, emz={}, sensor_pred={}'.format(i, pos, np.round(state[3:7], 3), emz, np.round(sensor_pred, 2)))


def generate_data(num_data, state, sensor_std, printBool):
    """
    生成模拟数据
    :param num_data: 【int】观测数据的维度
    :param state: 【np.array】模拟的胶囊状态 (n,)
    :param sensor_std：【float】磁传感器的噪声标准差 [mG]
    :param printBool: 【bool】是否打印输出
    :return: 【np.array】模拟的B值 + 加速度计的数值, (num_data, )
    """
    mid = h0(state)  # 模拟B值数据的中间值
    sim = np.zeros(num_data)
    for j in range(num_data):
        sim[j] = np.random.normal(mid[j], sensor_std, 1)

    if printBool:
        print('sensor_mid={}'.format(np.round(mid[:], 3)))
        print('sensor_sim={}'.format(np.round(sim[:], 3)))
    return sim


def sim(states, state0, sensor_std, plotType, plotBool, printBool, maxIter=100):
    '''
    使用模拟的观测值验证算法的准确性
    :param states: 模拟的真实状态
    :param state0: 模拟的初始值
    :param sensor_std: sensor的噪声标准差[mG]
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :param plotBool: 【bool】是否绘图
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return: 【tuple】 位置[x, y, z]和姿态ez的误差百分比
    '''
    m, n = 9, 7
    for i in range(1):
        # run
        output_data = generate_data(m, states[i], sensor_std, printBool)
        LM(state0, output_data, n, maxIter, printBool)

        if plotBool:
            # plot pos and em
            # 最大化窗口
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            iters = len(poss)
            for j in range(iters):
                state00 = np.concatenate((poss[j], ems[j]))
                plt.ion()
                plotPos(state00, states[i], j, plotType)
                if j == iters - 1:
                    plt.ioff()
                    plt.show()
            # plotLM(residual_memory, us)

        posTruth, emTruth = states[0][:3], q2R(states[0][3: 7])[:, -1]
        err_pos = np.linalg.norm(poss[-1] - posTruth) / np.linalg.norm(posTruth)
        err_em = np.linalg.norm(q2R(ems[-1])[:, -1] - emTruth)  # 方向矢量本身是归一化的
        print('pos={}， emz={}: err_pos={:.0%}, err_em={:.0%}'.format(np.round(posTruth, 3), np.round(emTruth, 3), err_pos, err_em))
        residual_memory.clear()
        us.clear()
        poss.clear()
        ems.clear()
        return (err_pos, err_em)


def simErrDistributed(contourBar, sensor_std=10, pos_or_ori=1):
    '''
    模拟误差分布
    :param contourBar: 【np.array】等高线的刻度条
    :param sensor_std: 【float】sensor的噪声标准差[mG]
    :param pos_or_ori: 【int】选择哪个输出 0：位置，1：姿态
    :return:
    '''
    n = 20
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, n), np.linspace(-0.2, 0.2, n))
    state0 = np.array([0, 0, -0.5, 1, 0, 0, 0, MOMENT, 0, 0])
    states = [np.array([0, 0, -0.5, 1, 0, 0, 0])]
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            state0[0] = x[i, j]
            state0[1] = y[i, j]
            # states[0][0] = x[i, j]
            # states[0][1] = y[i, j]
            z[i, j] = sim(states, state0, sensor_std, plotBool=False, plotType=(0, 1), printBool=False)[pos_or_ori]

    plotErr(x, y, z, contourBar, titleName='sensor_std={}'.format(sensor_std))


def runReadData(printBool, maxIter=50):
    '''
    跑实际的数据来实现定位
    :param printBool: 【bool】是否打印输出
    :param maxIter: 【int】最大迭代次数
    :return:
    '''
    snesorDict = {'imu': 'LSM6DS3TR-C', 'magSensor': 'AK09970d'}
    readObj = ReadData(snesorDict)    # 创建读取数据的对象

    outputData = multiprocessing.Array('f', [0] * len(snesorDict) * 24)
    magBg = multiprocessing.Array('f', [0] * 6)
    state0 = multiprocessing.Array('f', [0, 0, 0.01, 1, 0, 0, 0])

    readObj.send()
    pRec = Process(target=readObj.receive, args=(outputData, magBg, None))
    # pRec.daemon = True
    pRec.start()
    time.sleep(2)

    pTrack3D = multiprocessing.Process(target=track3D, args=(state0, ))
    pTrack3D.daemon = True
    pTrack3D.start()

    while True:
        measureData = np.concatenate((outputData[:3], outputData[6: 9]))
        LM(state0, measureData, 7, maxIter, printBool)
        time.sleep(0.1)


if __name__ == '__main__':
    state0 = np.array([0, 0, 0.01, 1, 0, 0, 0, MOMENT, 0, 0])  # 初始值

    # 仿真模拟
    # states = [np.array([0.2, -0.1, 0.2, 0.5 * math.sqrt(3), 0.5, 0, 0])]    # 真实值
    states = [np.array([0, 0.1, 0.2, 1, 2, 3, 0])]  # 真实值
    err = sim(states, state0, sensor_std=0.01, plotBool=False, plotType=(1, 2), printBool=True)
    # simErrDistributed(contourBar=9, sensor_std=10, pos_or_ori=0)

    # 实际运行
    # runReadData(printBool=False)
