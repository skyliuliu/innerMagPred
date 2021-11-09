# coding=utf-8
# /usr/bin/env python
'''
Author: Zhao Zhangsong, Liu Liu
Email: zs.zhao@sibionics.com, l.liu@sibionics.com
DateTime: 2021/10/26 10:00
desc: 从串口读取胶囊上的IMU和2个AKM磁传感器的数据

通信协议：
    name：       head/type/size/objid/instid/timestamp/data/crc
    len(byte) :   1 / 1  / 2(从head-crc) /  4   /   2  / 2(该位上发不存在) / 0~64/ 1
'''
import json
import multiprocessing
from multiprocessing.dummy import Process
import time
import struct
import ctypes
import binascii
from queue import Queue

import numpy as np
import serial
import serial.tools.list_ports
from filterpy.kalman import FixedLagSmoother as FLS

from predictorViewer import plotSensor


class ReadData:
    # 验证字段
    UAVTALK_SYNC_VAL = 0x3c
    UAVTALK_TYPE_MASK = 0x78
    UAVTALK_TYPE_VER = 0x20
    UAV_OBJ_SENSOR = 0x5F9FFBCA
    # 重力加速度【m/s^2】
    CONST_g0 = 9.8
    # AKMsensor的灵敏度【mGs/LSB】
    magSensitivity = 0.031

    def __init__(self, snesorDict):
        # 串口端口号
        port = list(serial.tools.list_ports.comports())[0][0]
        self.ser = serial.Serial(port, 230400, timeout=0.5)
        if self.ser.isOpen():
            print("open {} success!\n".format(port))
        else:
            raise RuntimeError("open failed")
        # 传感器种类
        self.sensorDict = snesorDict
        self.outDataNum = len(snesorDict) * 6
        # 存储所有sensor的所有输出，用于计算标准差std
        self.sensorAll = []
        for sensor_i in range(self.outDataNum):
            self.sensorAll.append(Queue())
        # 读取的数据
        self.imuSensorData = np.zeros((6, 4), dtype='float32')
        self.magSensorData = np.zeros((6, 4))
        self.timedata = np.zeros(4, dtype='uint32')
        # 用于计算原始数据的sigma
        self.sensorDataSigma = np.zeros((self.outDataNum, 4), dtype='float32')
        # 扣除背景磁场时的计数
        self.n = 0
        # 是否读取本地保存的背景磁场
        self.offset = True
        # 用于平滑磁传感器的数据
        self.magSmooth = np.zeros((6, 4), dtype='float32')
        # 固定区间平滑器,暂时只对磁传感器进行平滑
        self.fls = FLS(dim_x=6, dim_z=6, N=4)
        self.fls.P = 1
        self.fls.R = 0.05
        self.fls.Q = 0.01

    def PIOS_CRC_updateByte(self, crc, data) :
        crc_table = [
            0x00, 0x07, 0x0e, 0x09, 0x1c, 0x1b, 0x12, 0x15, 0x38, 0x3f, 0x36, 0x31, 0x24, 0x23, 0x2a, 0x2d,
            0x70, 0x77, 0x7e, 0x79, 0x6c, 0x6b, 0x62, 0x65, 0x48, 0x4f, 0x46, 0x41, 0x54, 0x53, 0x5a, 0x5d,
            0xe0, 0xe7, 0xee, 0xe9, 0xfc, 0xfb, 0xf2, 0xf5, 0xd8, 0xdf, 0xd6, 0xd1, 0xc4, 0xc3, 0xca, 0xcd,
            0x90, 0x97, 0x9e, 0x99, 0x8c, 0x8b, 0x82, 0x85, 0xa8, 0xaf, 0xa6, 0xa1, 0xb4, 0xb3, 0xba, 0xbd,
            0xc7, 0xc0, 0xc9, 0xce, 0xdb, 0xdc, 0xd5, 0xd2, 0xff, 0xf8, 0xf1, 0xf6, 0xe3, 0xe4, 0xed, 0xea,
            0xb7, 0xb0, 0xb9, 0xbe, 0xab, 0xac, 0xa5, 0xa2, 0x8f, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9d, 0x9a,
            0x27, 0x20, 0x29, 0x2e, 0x3b, 0x3c, 0x35, 0x32, 0x1f, 0x18, 0x11, 0x16, 0x03, 0x04, 0x0d, 0x0a,
            0x57, 0x50, 0x59, 0x5e, 0x4b, 0x4c, 0x45, 0x42, 0x6f, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7d, 0x7a,
            0x89, 0x8e, 0x87, 0x80, 0x95, 0x92, 0x9b, 0x9c, 0xb1, 0xb6, 0xbf, 0xb8, 0xad, 0xaa, 0xa3, 0xa4,
            0xf9, 0xfe, 0xf7, 0xf0, 0xe5, 0xe2, 0xeb, 0xec, 0xc1, 0xc6, 0xcf, 0xc8, 0xdd, 0xda, 0xd3, 0xd4,
            0x69, 0x6e, 0x67, 0x60, 0x75, 0x72, 0x7b, 0x7c, 0x51, 0x56, 0x5f, 0x58, 0x4d, 0x4a, 0x43, 0x44,
            0x19, 0x1e, 0x17, 0x10, 0x05, 0x02, 0x0b, 0x0c, 0x21, 0x26, 0x2f, 0x28, 0x3d, 0x3a, 0x33, 0x34,
            0x4e, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5c, 0x5b, 0x76, 0x71, 0x78, 0x7f, 0x6a, 0x6d, 0x64, 0x63,
            0x3e, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2c, 0x2b, 0x06, 0x01, 0x08, 0x0f, 0x1a, 0x1d, 0x14, 0x13,
            0xae, 0xa9, 0xa0, 0xa7, 0xb2, 0xb5, 0xbc, 0xbb, 0x96, 0x91, 0x98, 0x9f, 0x8a, 0x8d, 0x84, 0x83,
            0xde, 0xd9, 0xd0, 0xd7, 0xc2, 0xc5, 0xcc, 0xcb, 0xe6, 0xe1, 0xe8, 0xef, 0xfa, 0xfd, 0xf4, 0xf3 ]

        return crc_table[crc ^ data]

    def crc8Calculate(self, curCrc, data):
        val = curCrc
        for i in range(len(data)) :
            val = self.PIOS_CRC_updateByte(val, data[i])
        return val


    def sensorUnpack(self, data, outputData, outputDataSmooth, magBg, outputDataSigma):
        '''
        对读到的原始数据进行解包处理
        :param data:
        :return:
        '''
        objId = int.from_bytes(data[0:4], 'little')
        # print(objId)
        if objId == self.UAV_OBJ_SENSOR :
            instId = int.from_bytes(data[4:6], 'little')
            for i in range(4) :
                # 加速度计换算后的单位为[m/s^2]
                self.imuSensorData[0, i] = np.asarray(struct.unpack('<f', data[6+i*4:10+i*4])) * 0.001 * self.CONST_g0
                self.imuSensorData[1, i] = np.asarray(struct.unpack('<f', data[22+i*4:26+i*4])) * 0.001 * self.CONST_g0
                self.imuSensorData[2, i] = np.asarray(struct.unpack('<f', data[38+i*4:42+i*4])) * 0.001 * self.CONST_g0
                # 陀螺仪输出的单位为[deg/s]
                self.imuSensorData[3, i] = np.asarray(struct.unpack('<f', data[54+i*4:58+i*4]))
                self.imuSensorData[4, i] = np.asarray(struct.unpack('<f', data[70+i*4:74+i*4]))
                self.imuSensorData[5, i] = np.asarray(struct.unpack('<f', data[86+i*4:90+i*4]))
                # AKM磁传感器换算后的单位为[Gs]
                for j in range(6):
                    self.magSensorData[j, i] = np.asarray(
                        struct.unpack('<h', data[102 + 8*j + 2*i: 104 + 8*j + 2*i])) * self.magSensitivity
                # 时间戳
                self.timedata[i] = np.asarray(struct.unpack('<i', data[150+i*4:154+i*4]))

                # 存储所有sensor的所有输出，用于计算标准差std
                if outputDataSigma:
                    if self.n > 100:
                        for sensor_i in range(self.outDataNum):
                            self.sensorAll[sensor_i].get()
                    for sensor_i in range(6):
                        if 'imu' not in self.sensorDict.keys():
                            self.sensorAll[sensor_i].put(self.magSensorData[sensor_i, i])
                            self.sensorDataSigma[sensor_i][i] = np.array(self.sensorAll[sensor_i].queue).std()
                        elif 'magSensor' not in self.sensorDict.keys():
                            self.sensorAll[sensor_i].put(self.imuSensorData[sensor_i, i])
                            self.sensorDataSigma[sensor_i][i] = np.array(self.sensorAll[sensor_i].queue).std()
                        else:
                            self.sensorAll[sensor_i].put(self.imuSensorData[sensor_i, i])
                            self.sensorAll[sensor_i + 6].put(self.magSensorData[sensor_i, i])
                            self.sensorDataSigma[sensor_i][i] = np.array(self.sensorAll[sensor_i].queue).std()
                            self.sensorDataSigma[sensor_i + 6][i] = np.array(self.sensorAll[sensor_i + 6].queue).std()

                # 对磁传感器的读数进行平滑
                self.fls.smooth(self.magSensorData[:, i])
                self.magSmooth[:, i] = np.array(self.fls.xSmooth[-1])[0, :]

            if (not self.offset) and self.n < 25:
                for i in range(6):
                    magBg[i] =self.magSensorData[i].sum() + magBg[i]
            elif (not self.offset) and self.n == 25:
                for i in range(6):
                    magBg[i] = magBg[i] / self.n / 4
                self.offset = True
                print('Calibrate magnetic filed ok!')
                # 保存背景磁场到本地json文件
                bg = {}
                for row in range(6):
                    bg['B{}'.format(row)] = magBg[row]
                f = open('bg.json', 'w')
                json.dump(bg, f, indent=4)
                f.close()
            else:
                for i in range(6):
                    self.magSensorData[i] = self.magSensorData[i] - magBg[i]
                    self.magSmooth[i] = self.magSmooth[i] - magBg[i]

            if 'magSensor' not in self.sensorDict.keys():
                outputData[:] = np.hstack(np.stack(self.imuSensorData, axis=1))
            elif 'imu' not in self.sensorDict.keys():
                outputData[:] = np.hstack(np.stack(self.magSensorData, axis=1))
                outputDataSmooth[:] = np.hstack(np.stack(self.magSmooth, axis=1))
            else:
                outputData[:] = np.hstack(np.stack(np.vstack((self.imuSensorData, self.magSensorData)), axis=1))
                outputDataSmooth[:] = np.hstack(np.stack(self.magSmooth, axis=1))
            if outputDataSigma:
                outputDataSigma[:] = np.hstack(np.stack(self.sensorDataSigma, axis=1))
            # print("outputData={}".format(np.round(outputData, 2)))
            # print("outputDataSmooth={}".format(np.round(outputDataSmooth, 2)))


    def receive(self, outputData, outputDataSmooth, magBg, outputDataSigma=None):
        '''
        读串口
        :param outputDataSigma: 输出数据的标准差
        :return:
        '''
        # Wait a second to let the port initialize
        time.sleep(0.01)
        viodFlag = b''

        # 读取本地保存的背景磁场
        if self.offset:
            f = open('bg.json', 'r')
            bg = json.load(f)
            for row in range(6):
                magBg[row] = bg.get('B{}'.format(row), 0)
            f.close()
            print('get background B OK!')
        else:
            print('start calibrate magnetic field--------------')
        while True:
            data = self.ser.read()   # read data from serial_port

            if len(data) > 0 :
                syncVal = int.from_bytes(data,'little')   # 将字节串转换为整数（反序）
                if syncVal == self.UAVTALK_SYNC_VAL :   # 验证UAV
                    crc8 = self.PIOS_CRC_updateByte(0, syncVal)
                    dataType = int.from_bytes(self.ser.read(),'little')
                    if dataType == self.UAVTALK_TYPE_VER:   # 验证数据类型
                        crc8 = self.PIOS_CRC_updateByte(crc8, dataType)
                        dataLen = int.from_bytes(self.ser.read(2),'little')
                        _dataLen = ctypes.c_short(dataLen)
                        high_8 = (_dataLen.value & 0xff00) >> 8
                        crc8 = self.PIOS_CRC_updateByte(crc8, high_8)
                        low_8 = (_dataLen.value & 0x00ff)
                        crc8 = self.PIOS_CRC_updateByte(crc8, low_8)
                        readLen = dataLen + 4 + 2 + 1    # 有用数据的长度
                        dataBuff = self.ser.read(readLen)   # 读取有用数据
                        objId = int.from_bytes(dataBuff[0:4], 'little')

                        if len(dataBuff) > 0 and objId == self.UAV_OBJ_SENSOR:    # 验证sensor obj id的UAV
                            crc8 = self.crc8Calculate(crc8, dataBuff)
                            crc8Val = dataBuff[-1]
                            decData = binascii.b2a_hex(dataBuff).decode('utf-8')

                            self.sensorUnpack(dataBuff, outputData, outputDataSmooth, magBg, outputDataSigma)
                            self.n += 1


    def send(self):
        '''
        向串口发命令
        :param serial_port: 串口端口号
        :return:
        '''
        time.sleep(0.5)

        initCmd1 = '< J.L0?..rf allinit 23...................................................??'
        setsCmd = '< J.L0?..sets capsule 0x96761133 0x0DB7CE15..............................8.'
        # initCmd = initCmd1.strip('\n')
        cmdBuf = initCmd1.encode('utf-8')

        initStr = '3C 20 4A 00 4C 30 D5 02 00 00 72 66 20 61 6C 6C 69 6E 69 74 20 32 33 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 B1'
        initBytes = binascii.a2b_hex(initStr.replace(' ', ''))
        setsStr = '3C 20 4A 00 4C 30 D5 02 00 00 73 65 74 73 20 63 61 70 73 75 6C 65 20 30 78 39 36 37 36 31 31 33 33 20 30 78 30 44 42 37 43 45 31 35 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 38'
        setsBytes = binascii.a2b_hex(setsStr.replace(' ', ''))

        self.ser.write(initBytes)
        print(initBytes)
        time.sleep(1)

        self.ser.write(setsBytes)
        print(setsBytes)
        time.sleep(1)


if __name__ == '__main__':
    # snesorDict = {'imu': 'LSM6DS3TR-C', 'magSensor': 'AK09970d'}
    snesorDict = {'magSensor': 'AK09970d'}
    # snesorDict = {'imu': 'LSM6DS3TR-C'}
    readObj = ReadData(snesorDict)    # 创建读取数据的对象

    # def data struct
    outputData = multiprocessing.Array('f', [0] * len(snesorDict) * 24)
    outputDataSmooth = multiprocessing.Array('f', [0] * len(snesorDict) * 24)
    # outputDataSigma = multiprocessing.Array('f', [0] * len(snesorDict) * 24)
    outputDataSigma = None
    magBg = multiprocessing.Array('f', [0] * 6)

    # Wait a second to let the port initialize
    readObj.send()
    # receive data in a new process
    pRec = Process(target=readObj.receive, args=(outputData, outputDataSmooth, magBg, outputDataSigma))
    pRec.daemon = True
    pRec.start()
    time.sleep(1)
    # plot sensor data
    plotSensor(snesorDict, outputData, outputDataSmooth, outputDataSigma)
