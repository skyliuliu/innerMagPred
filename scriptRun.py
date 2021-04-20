# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/14 15:25
desc: 运行测试脚本
'''
import csv
from datetime import datetime
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt

from mahonyPredictor import MahonyPredictor

data_pwd = r"E:\胶囊\胶囊姿态\IMU数据\IMU_data0.csv"
data = []
data_lines = 0
with open(data_pwd, 'r') as fr:
    data_reader = csv.reader(fr)
    for row in islice(data_reader, 1, None):
        data_lines += 1
        dataRow = {"date": row[0],
                   "Time": row[1],
                   "ax": float(row[2]),
                   "ay": float(row[3]),
                   "az": float(row[4]),
                   "wx": float(row[5]),
                   "wy": float(row[6]),
                   "wz": float(row[7])}
        data.append(dataRow)


pitch = np.zeros(data_lines)
roll = np.zeros(data_lines)
yaw = np.zeros(data_lines)
t = []
deg2rad = 57.3

predictor = MahonyPredictor()

for i in range(data_lines):
    d = data[i]
    w = [d["wx"], d["wy"], d["wz"]]
    a = [d["ax"], d["ay"], d["az"]]
    predictor.IMUupdate(w, a)
    pitch[i] = predictor.pitch * 57.3
    roll[i] = predictor.roll * 57.3
    yaw[i] = predictor.yaw * 57.3
    t.append(datetime.strptime(d["date"] + d["Time"], "%Y/%m/%d %H:%M:%S.%f"))
    #print("pitch={:.3f}, roll={:.3f}, yaw={:.3f}".format(predictor.pitch * deg2rad, predictor.roll * deg2rad , predictor.yaw * deg2rad))

with open("output.csv", 'w') as fw:
    writer = csv.writer(fw)
    for i in range(data_lines):
        writer.writerow((t[i], pitch[i], roll[i], yaw[i]))

# plt.plot(t, pitch, label='pitch', color='green')
# plt.plot(t, roll, label='roll', color='blue')
# plt.plot(t, yaw, label='yaw', color='red')
# plt.grid()
# plt.legend()
# plt.show()