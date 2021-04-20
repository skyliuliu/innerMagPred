# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/14 11:35
desc:
'''
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

class CapsulePosPredictor(EKF):

    def __init__(self, dim_x, dim_z):
        super().__init__(dim_x, dim_z)
        self.P *= 1000
        self.x = np.array([1, 0, 0, 0, 0, 0, 0])
        self.Q *= 0.5
        self.R *= 0.001
