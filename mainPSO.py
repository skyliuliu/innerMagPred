# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/12/4 10:42
desc: 使用PSO（粒子群优化）算法进行内置式磁定位
'''
import math
import random

import numpy as np

from predictorViewer import q2R
from mainLM import h0, generate_data

ws = 0.9
we = 0.5
c1 = 1.5
c2 = 1.5
pN = 1000
dim = 7
max_iter = 100

class PSO:

    def __init__(self):
        self.X = np.zeros((pN, dim))    # 粒子的位姿
        self.V = np.zeros((pN, dim))    # 粒子位姿的增加量

        self.posMax = 0.3    # 位置的上限
        self.posMin = -0.3    # 位置的下限
        self.qMax = 1    # 四元数的上限
        self.qMin = 0    # 四元数的下限
        self.Vdelta = 0.005   # 位置增量上限
        self.qdelta = 0.1    # 四元数增量上限

        self.pbest = np.zeros((pN, dim))   # 个体经历的最佳位置
        self.gbest = np.zeros((1, dim))    # 全局最佳位置
        self.pCost = np.zeros(pN)    # 每个粒子的cost
        self.gCost = float('inf')    # 全局cost


    def initParticles(self, output_data):
        for pi in range(pN):
            for j in range(dim):
                if j < 3:
                    self.X[pi][j] = random.uniform(self.posMin, self.posMax)
                    self.V[pi][j] = random.uniform(0, self.Vdelta)
                else:
                    self.X[pi][j] = random.uniform(self.qMin, self.qMax)
                    self.V[pi][j] = random.uniform(-self.qdelta, self.qdelta)

            self.pbest[pi] = self.X[pi]
            tmp = self.fun(self.X[pi], output_data)
            self.pCost[pi] = tmp
            if tmp < self.gCost:
                self.gCost = tmp
                self.gbest = self.X[pi]

    def fun(self, state, output_data):
        """
        以观测量与预估量之差的二范数作为目标函数
        :param state: 【np.array】预估的状态量 (n, )
        :param output_data: 【np.array】观测量 (m, )
        :return:【np.array】 residual (m, )
        """
        data_est_output = h0(state[:7])
        res = output_data - data_est_output
        return np.linalg.norm(res)

    def update(self, output_data):
        fitness = []
        for it in range(max_iter):
            w = ws + (we - ws) * (it / max_iter)
            for pi in range(pN):
                # 更新每个粒子的位置
                self.V[pi] = w * self.V[pi] + c1 * random.uniform(0, 1) * (self.pbest[pi] - self.X[pi]) + c2 * random.uniform(0, 1) * (self.gbest - self.X[pi])
                self.X[pi] += self.V[pi]
                # 检查是否越界
                for j in range(3):
                    self.X[pi][j] = max(self.posMin, self.X[pi][j])
                    self.X[pi][j] = min(self.posMax, self.X[pi][j])

                # 更新pbest和gbest
                tmp = self.fun(self.X[pi], output_data)
                if tmp < self.pCost[pi]:
                    self.pCost[pi] = tmp
                    self.pbest[pi] = self.X[pi]
                if tmp < self.gCost:
                    self.gCost = tmp
                    self.gbest = self.X[pi]

            fitness.append(self.gCost)
            pos = self.gbest[:3]
            ez = q2R(self.gbest[3:7])[:,-1]
            print('i={}: pos={}, ez={}, cost={}'.format(it + 1, np.round(pos, 3), np.round(ez, 3), self.gCost))

def sim():
    state = np.array([0, -0.05 , 0.15, 1, 2, 3, 0])
    simData = generate_data(6, state, 0.1, True)

    pso = PSO()
    pso.initParticles(simData)
    pso.update(simData)

if __name__ == '__main__':
    sim()