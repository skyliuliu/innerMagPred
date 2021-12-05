# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/14 10:40
desc: 互补滤波算法实现类，引用Mahony提出的互补滤波算法
'''
import math
import multiprocessing
from multiprocessing.dummy import Process
import time

from readData import ReadData
from predictorViewer import track3D

#from capsulePosPredictor import CapsulePosPredictor

class MahonyPredictor:
    offset_cnt = 0
    max_time = 20
    wx_offset, wy_offset, wz_offset = 0, 0, 0
    exInt, eyInt, ezInt = (0, 0, 0)

    def __init__(self, Kp=20, Ki=0.008, dt=0.01, yaw=0, pitch=0, roll=0, q=None):
        self.q = [1, 0, 0, 0] if q is None else q
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.dt = dt
        self.Ki = Ki
        self.Kp = Kp

    def getGyroOffset(self, w):
        '''
        消除陀螺仪原始数据的零偏，静止时取前20个陀螺仪数据取平均值，之后每次采集到陀螺仪数据时减去该偏差
        :param w: 【float】 陀螺仪输出的角速度[deg/s]
        :return:  【bool】 消除零偏是否ok，0：ok； 1：不ok
        '''
        if self.offset_cnt < self.max_time:
            self.wx_offset += w[0]
            self.wy_offset += w[1]
            self.wz_offset += w[2]
            self.offset_cnt += 1
            print("Calibrating gyroscope offset...")
            return False
        elif self.offset_cnt == self.max_time:
            self.wx_offset /= self.max_time
            self.wy_offset /= self.max_time
            self.wz_offset /= self.max_time
            self.offset_cnt += 1
            print("Calibrate OK!!")
        else:
            return True

    def autoPI_byGyro(self, w):
        '''
        基于陀螺仪数据的自适应互补滤波
        :param w: 【float】 陀螺仪输出的角速度[deg/s]
        :return: 修改Kp和Ki
        '''
        w1, w2, w3 = 207.24, 1792.8, 2000
        Kp0, Kp1, Ki0 = 20, 30, 0.008
        wNorm = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        if wNorm < w1:
            self.Kp = Kp0
            self.Ki = Ki0
        elif w1 < wNorm < w2:
            self.Kp = 20 + 10 / w2 * (wNorm - w1)
            self.Ki = Ki0
        elif w2 < wNorm < w3:
            self.Kp = Kp1
            self.Ki = Ki0
        else:
            return

    def IMUupdate(self, a, w, b=None):
        '''
        互补滤波算法主体
        :param w: 【float】 陀螺仪输出的角速度[deg/s]
        :param a: 【float】 加速度计输出的加速度[m/s^2]
        :return:  修改欧拉角
        '''
        # 消除零偏
        # if not self.getGyroOffset(w):
        #     return
        w[0] -= self.wx_offset
        w[1] -= self.wy_offset
        w[2] -= self.wz_offset

        # 自适应调节PI
        #self.autoPI_byGyro(w)

        # 先减小角速度计零偏
        # GyroErrorPredictor.EKF_updateIMU(w, a)
        # todo

        # 互补滤波核心过程
        # 1、对加速度数据进行归一化
        aNorm = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        ax, ay, az = (ai / aNorm for ai in a) if aNorm else (0, 0, 1)

        # 2、载体坐标转为世界坐标
        # 重力加速度在载体坐标系下的方向矢量
        vx = 2 * (self.q[1] * self.q[3] - self.q[0] * self.q[2])
        vy = 2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3])
        vz = self.q[0] * self.q[0] - self.q[1] * self.q[1] -self.q[2] * self.q[2] + self.q[3] * self.q[3]



        # 3、在机体坐标系下做向量叉积得到补偿数据
        if b:
            bNorm = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
            bx = -b[0] / bNorm if bNorm else 1
            by = b[1] / bNorm if bNorm else 0
            bz = -b[2] / bNorm if bNorm else 0

            # 磁场北极方向在载体坐标系下的方向矢量
            nx = bx * (0.5 - self.q[2] * self.q[2] - self.q[3] * self.q[3]) + bz * (self.q[1] * self.q[3] - self.q[0] * self.q[2])
            ny = bx * (self.q[1] * self.q[2] - self.q[0] * self.q[3]) + bz * (self.q[0] * self.q[1] + self.q[2] * self.q[3])
            nz = bx * (self.q[0] * self.q[2] + self.q[1] * self.q[3]) + bz * (0.5 - self.q[1] * self.q[1] - self.q[2] * self.q[2])

            ex = ay * vz - az * vy + by * nz - bz * ny
            ey = az * vx - ax * vz + bz * nx - bx * nz
            ez = ax * vy - ay * vx + bx * ny - by * nx
        else:
            ex = ay * vz - az * vy
            ey = az * vx - ax * vz
            ez = ax * vy - ay * vx

        # 4、对误差进行PI计算，补偿角速度
        self.exInt += ex * self.Ki
        self.eyInt += ey * self.Ki
        self.ezInt += ez * self.Ki
        w[0] += ex * self.Kp + self.exInt
        w[1] += ey * self.Kp + self.eyInt
        w[2] += ez * self.Kp + self.ezInt

        # 5、按照四元数微分公式进行四元数更新
        qTemp = self.q[:]
        self.q[0] += (-qTemp[1] * w[0] - qTemp[2] * w[1] - qTemp[3] * w[2]) * 0.5 * self.dt
        self.q[1] += (qTemp[0] * w[0] + qTemp[2] * w[2] - qTemp[3] * w[1]) * 0.5 * self.dt
        self.q[2] += (qTemp[0] * w[1] - qTemp[1] * w[2] + qTemp[3] * w[0]) * 0.5 * self.dt
        self.q[3] += (qTemp[0] * w[2] + qTemp[1] * w[1] - qTemp[2] * w[0]) * 0.5 * self.dt
        qNorm = math.sqrt(self.q[0] * self.q[0] + self.q[1] * self.q[1] + self.q[2] * self.q[2] + self.q[3] * self.q[3])
        self.q = [qi / qNorm for qi in self.q]

        # 6、四元数转欧拉角
        eps = 0.005
        qq = self.q[0] * self.q[2] - self.q[1] * self.q[3]

        if abs(qq) > 0.5 - eps:  # 奇异姿态,roll为±90°
            sign = -1 if qq < 0 else 1
            self.yaw = -2 * sign * math.atan2(self.q[1], self.q[0])
            self.roll = sign * math.pi * 0.5
            self.pitch = 0
        else:
            self.pitch = math.atan2(2 * self.q[0] * self.q[1] + 2 * self.q[2] * self.q[3], 1 - 2 * self.q[1] * self.q[1] - 2 * self.q[2] * self.q[2])
            self.roll = math.asin(2 * self.q[0] * self.q[2] - 2 * self.q[3] * self.q[1])
            self.yaw = math.atan2(2 * self.q[0] * self.q[3] + 2 * self.q[1] * self.q[2], 1 - 2 * self.q[2] * self.q[2] - 2 * self.q[3] * self.q[3])

def main():
    snesorDict = {'imu': 'LSM6DS3TR-C', 'magSensor': 'AK09970d'}
    readObj = ReadData(snesorDict)
    outputDataSmooth = None
    outputDataSigma = None
    magBg = multiprocessing.Array('f', [0] * 6)
    outputData = multiprocessing.Array('f', [0] * len(snesorDict) * 24)

    state = multiprocessing.Array('f', [0, 0, 0, 1, 0, 0, 0])

    # Wait a second to let the port initialize
    readObj.send()
    # receive data in a new process
    pRec = Process(target=readObj.receive, args=(outputData, outputDataSmooth, magBg, outputDataSigma))
    pRec.daemon = True
    pRec.start()

    pTrack3D = multiprocessing.Process(target=track3D, args=(state,))
    pTrack3D.daemon = True
    pTrack3D.start()

    mp = MahonyPredictor(q=state[3:], Kp=100, Ki=0.01, dt=0.002)
    while True:
        print("pitch={:.0f}, roll={:.0f}".format(mp.pitch * 57.3, mp.roll * 57.3))
        mp.getGyroOffset(outputData[3:6])
        mp.IMUupdate(outputData[:3], outputData[3:6], outputData[6:9])
        state[3:] = mp.q
        time.sleep(0.08)


if __name__ == '__main__':
    main()