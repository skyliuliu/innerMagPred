# coding=utf-8
# /usr/bin/env python
'''
Author: Liu Liu
Email: Nicke_liu@163.com
DateTime: 2021/4/19 16:00
desc: 定位结果的显示
'''
import copy
import math
from queue import Queue
import sys

import matplotlib.pyplot as plt
import numpy as np
from filterpy.stats import plot_covariance
import OpenGL.GL as ogl
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui



class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, X, Y, Z, text):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(int(self.X), int(self.Y), int(self.Z), self.text)


class Custom3DAxis(gl.GLAxisItem):
    """Class defined to extend 'gl.GLAxisItem'."""

    def __init__(self, parent, color=(1, 2, 3, 4)):
        gl.GLAxisItem.__init__(self)
        self.parent = parent
        self.c = color
        self.ticks = [-20, -10, 0, 10, 20]
        self.setSize(x=40, y=40, z=40)
        self.add_labels()
        self.add_tick_values(xticks=self.ticks, yticks=self.ticks, zticks=[0, 10, 20, 30, 40])
        self.addArrow()

    def add_labels(self):
        """Adds axes labels."""
        x, y, z = self.size()
        x *= 0.5
        y *= 0.5
        # X label
        self.xLabel = CustomTextItem(X=x + 0.5, Y=-y / 10, Z=-z / 10, text="X(cm)")
        self.xLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.xLabel)
        # Y label
        self.yLabel = CustomTextItem(X=-x / 10, Y=y + 0.5, Z=-z / 10, text="Y(cm)")
        self.yLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.yLabel)
        # Z label
        self.zLabel = CustomTextItem(X=-x / 10, Y=-y / 10, Z=z + 1, text="Z(cm)")
        self.zLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.zLabel)

    def add_tick_values(self, xticks=None, yticks=None, zticks=None):
        """Adds ticks values."""
        x, y, z = self.size()
        xtpos = np.linspace(-0.5 * x, 0.5 * x, len(xticks))
        ytpos = np.linspace(-0.5 * y, 0.5 * y, len(yticks))
        ztpos = np.linspace(0, z, len(zticks))
        # X label
        for i, xt in enumerate(xticks):
            val = CustomTextItem(X=xtpos[i], Y=2, Z=0, text=str(xt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)
        # Y label
        for i, yt in enumerate(yticks):
            val = CustomTextItem(X=2, Y=ytpos[i], Z=0, text=str(yt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)
        # Z label
        for i, zt in enumerate(zticks):
            val = CustomTextItem(X=0, Y=2, Z=ztpos[i], text=str(zt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)

    def addArrow(self):
        # add X axis arrow
        arrowXData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowX = gl.GLMeshItem(meshdata=arrowXData, color=(0, 0, 1, 0.6), shader='balloon', glOptions='opaque')
        arrowX.rotate(90, 0, 1, 0)
        arrowX.translate(20, 0, 0)
        self.parent.addItem(arrowX)
        # add Y axis arrow
        arrowYData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowY = gl.GLMeshItem(meshdata=arrowXData, color=(1, 0, 1, 0.6), shader='balloon', glOptions='opaque')
        arrowY.rotate(270, 1, 0, 0)
        arrowY.translate(0, 20, 0)
        self.parent.addItem(arrowY)
        # add Z axis arrow
        arrowZData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowZ = gl.GLMeshItem(meshdata=arrowXData, color=(0, 1, 0, 0.6), shader='balloon', glOptions='opaque')
        arrowZ.translate(0, 0, 40)
        self.parent.addItem(arrowZ)

    def paint(self):
        self.setupGLState()
        if self.antialias:
            ogl.glEnable(ogl.GL_LINE_SMOOTH)
            ogl.glHint(ogl.GL_LINE_SMOOTH_HINT, ogl.GL_NICEST)
        ogl.glBegin(ogl.GL_LINES)

        x, y, z = self.size()
        # Draw Z
        ogl.glColor4f(0, 1, 0, 10.6)  # z is green
        ogl.glVertex3f(0, 0, 0)
        ogl.glVertex3f(0, 0, z)
        # Draw Y
        ogl.glColor4f(1, 0, 1, 10.6)  # y is grape
        ogl.glVertex3f(0, -0.5 * y, 0)
        ogl.glVertex3f(0, 0.5 * y, 0)
        # Draw X
        ogl.glColor4f(0, 0, 1, 10.6)  # x is blue
        ogl.glVertex3f(-0.5 * x, 0, 0)
        ogl.glVertex3f(0.5 * x, 0, 0)
        ogl.glEnd()


def track3D(state):
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    # w.setWindowTitle('3d trajectory')
    w.resize(600, 500)
    # instance of Custom3DAxis
    axis = Custom3DAxis(w, color=(0.6, 0.6, 0.2, .6))
    w.addItem(axis)
    w.opts['distance'] = 75
    w.opts['center'] = pg.Vector(0, 0, 15)
    # add xy grid
    gx = gl.GLGridItem()
    gx.setSize(x=40, y=40, z=10)
    gx.setSpacing(x=5, y=5)
    w.addItem(gx)
    # trajectory line
    pos0 = np.array([[0, 0, 0]])
    pos, q = np.array(state[:3]), state[3:7]
    uAxis, angle = q2ua(q)
    track0 = np.concatenate((pos0, pos.reshape(1, 3)))
    plt = gl.GLLinePlotItem(pos=track0, width=2, color=(1, 0, 0, 0.6))
    w.addItem(plt)
    # orientation arrow
    sphereData = gl.MeshData.sphere(rows=20, cols=20, radius=0.6)
    sphereMesh = gl.GLMeshItem(meshdata=sphereData, smooth=True, shader='shaded', glOptions='opaque')
    w.addItem(sphereMesh)
    ArrowData = gl.MeshData.cylinder(rows=20, cols=20, radius=[0.5, 0], length=1.5)
    ArrowMesh = gl.GLMeshItem(meshdata=ArrowData, smooth=True, color=(1, 0, 0, 0.6), shader='balloon',
                              glOptions='opaque')
    ArrowMesh.rotate(angle, uAxis[0], uAxis[1], uAxis[2])
    w.addItem(ArrowMesh)
    w.setWindowTitle('position={}cm'.format(np.round(pos * 100, 1)))
    w.show()

    i = 1
    pts = pos.reshape(1, 3)

    def update():
        '''update position and orientation'''
        nonlocal i, pts, state
        pos, q = np.array(state[:3]) * 100, state[3:7]
        uAxis, angle = q2ua(q)
        pt = (pos).reshape(1, 3)
        if pts.size < 150:
            pts = np.concatenate((pts, pt))
        else:
            pts = np.concatenate((pts[-50:, :], pt))
        plt.setData(pos=pts)
        ArrowMesh.resetTransform()
        sphereMesh.resetTransform()
        ArrowMesh.rotate(angle, uAxis[0], uAxis[1], uAxis[2])
        ArrowMesh.translate(*pos)
        sphereMesh.translate(*pos)
        w.setWindowTitle('position={}cm'.format(np.round(pos, 1)))
        i += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def q2R(q):
    '''
    从四元数求旋转矩阵
    :param q: 【np.array】四元数
    :return: R 旋转矩阵
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3,     2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3,     1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2,     2 * q2 * q3 + 2 * q0 * q1,     1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])
    return R

def q2ua(q):
    '''
    从四元数求旋转向量和旋转角
    :param q:
    :return:
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    angle = 2 * math.acos(q0)
    u = np.array([q1, q2, q3]) / math.sin(0.5 * angle) if angle else np.array([0, 0, 1])
    return u, angle * 57.3

def q2Euler(q):
    '''
    从四元数求欧拉角
    :param q: 【np.array】四元数
    :return: 【np.array】 [pitch, roll, yaw]
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    pitch = math.atan2(2 * q0 * q1 + 2 * q2 * q3, 1 - 2 * q1 * q1 - 2 * q2 * q2)
    roll = math.asin(2 * q0 * q2 - 2 * q3 * q1)
    yaw = math.atan2(2 * q0 * q3 + 2 * q1 * q2, 1 - 2 * q2 * q2 - 2 * q3 * q3)
    return np.array([pitch, roll, yaw]) * 57.3

def plotLM(residual_memory, us):
    '''
    描绘LM算法的残差和u值（LM算法的参数）曲线
    :param residual_memory: 【list】残差列表
    :param us: 【list】u值列表
    :return:
    '''
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # plt.plot(residual_memory)
    for ax in [ax1, ax2]:
        ax.set_xlabel("iter")
    ax1.set_ylabel("residual")
    ax1.semilogy(residual_memory)
    ax2.set_xlabel("iter")
    ax2.set_ylabel("u")
    ax2.semilogy(us)
    plt.show()


# plt.axis('auto')   # 坐标轴自动缩放

def plotP(predictor, state, index, plotType):
    '''
    描绘UKF算法中误差协方差yz分量的变化过程
    :param state0: 【np.array】预测状态 （7，）
    :param state: 【np.array】真实状态 （7，）
    :param index: 【int】算法的迭代步数
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :return:
    '''
    x, y = plotType
    state_copy = state.copy()  # 浅拷贝真实值，因为后面会修改state
    xtruth = state_copy[:3]  # 获取坐标真实值
    mtruth = q2R(state_copy[3: 7])[:, -1]  # 获取姿态真实值，并转换为z方向的矢量

    pos, q = predictor.ukf.x[:3].copy(), predictor.ukf.x[3: 7]  # 获取预测值，浅拷贝坐标值
    em = q2R(q)[:, -1]
    if plotType == (0, 1):
        plt.ylim(-0.2, 0.4)
        plt.axis('equal')  # 坐标轴按照等比例绘图
    elif plotType == (1, 2):
        xtruth[1] += index * 0.1
        pos[1] += index * 0.1
    else:
        raise Exception("invalid plotType")

    P = predictor.ukf.P[x: y+1, x: y+1]  # 坐标的误差协方差
    plot_covariance(mean=pos[x: y+1], cov=P, fc='g', alpha=0.3, title='胶囊定位过程仿真')
    plt.text(pos[x], pos[y], int(index), fontsize=9)
    plt.plot(xtruth[x], xtruth[y], 'ro')  # 画出真实值
    plt.text(xtruth[x], xtruth[y], int(index), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos[x] + em[x] * scale, pos[y] + em[y] * scale), xytext=(pos[x], pos[y]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[x] + mtruth[x] * scale, xtruth[y] + mtruth[y] * scale),
                 xytext=(xtruth[x], xtruth[y]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))

    # 添加坐标轴标识
    plt.xlabel('{}/m'.format('xyz'[x]))
    plt.ylabel('{}/m'.format('xyz'[y]))
    # 添加网格线
    plt.gca().grid(b=True)
    # 增加固定时间间隔
    plt.pause(0.05)

def plotPos(state0, state, index, plotType):
    '''
    描绘预测位置的变化过程
    :param state0: 【np.array】预测状态 （7，）
    :param state: 【np.array】真实状态 （7，）
    :param index: 【int】算法的迭代步数
    :param plotType: 【tuple】描绘位置的分量 'xy' or 'yz'
    :return:
    '''
    x, y = plotType
    state_copy = state.copy()  # 浅拷贝真实值，因为后面会修改state
    xtruth = state_copy[:3]  # 获取坐标真实值
    mtruth = q2R(state_copy[3: 7])[:, -1]  # 获取姿态真实值，并转换为z方向的矢量

    pos, q = state0[:3].copy(), state0[3:]    # 获取预测值，浅拷贝坐标值
    em = q2R(q)[:, -1]
    if plotType == (0, 1):
        # 添加坐标轴标识
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.axis('equal')    # 坐标轴按照等比例绘图
        plt.ylim(-0.2, 0.5)
        # plt.gca().set_aspect('equal', adjustable='box')
    elif plotType == (1, 2):
        xtruth[1] += index
        pos[1] += index
        # 添加坐标轴标识
        plt.xlabel('y/m')
        plt.ylabel('z/m')
    else:
        raise Exception("invalid plotType")

    plt.plot(pos[x], pos[y], 'b+')  # 仅描点
    plt.text(pos[x], pos[y], int(index), fontsize=9)
    plt.plot(xtruth[x], xtruth[y], 'ro')  # 画出真实值
    plt.text(xtruth[x], xtruth[y], int(index), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos[x] + em[x] * scale, pos[y] + em[y] * scale), xytext=(pos[x], pos[y]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[x] + mtruth[x] * scale, xtruth[y] + mtruth[y] * scale),
                 xytext=(xtruth[x], xtruth[y]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))

    plt.gca().grid(b=True)
    plt.pause(0.05)

def plotErr(x, y, z, contourBar, titleName):
    '''
    描绘误差分布的等高线图
    :param x: 【np.array】误差分布的x变量 (n, n)
    :param y: 【np.array】误差分布的y变量 (n, n)
    :param z: 【np.array】误差分布的结果 (n, n)
    :param contourBar: 【np.array】等高线的刻度条
    :param titleName: 【string】图的标题名称
    :return:
    '''
    plt.title(titleName)
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.tick_params(labelsize=10)
    plt_contourf = plt.contourf(x, y, z, contourBar, cmap='jet', extend='both')    # 填充等高线内区域
    cmap = copy.copy(plt_contourf.get_cmap())
    cmap.set_over('red')     # 超过contourBar的上限就填充为red
    cmap.set_under('blue')     # 低于contourBar的下限就填充为blue
    plt_contourf.changed()

    cntr = plt.contour(x, y, z, contourBar, colors='black', linewidths=0.5)    # 描绘等高线轮廓
    plt.clabel(cntr, inline_spacing=1, fmt='%.2f', fontsize=8, colors='black')     # 标识等高线的数值
    plt.show()


def plotSensor(sensorDict, data0, data0Sigma, dataSmooth=None):
    '''
    对sensor读取的结果进行绘图
    :param sensorDict: 【dict】sensor名字的列表
    :param data0:  【Array】sensor原始数据的数组
    :param data0Sigma: 【Array】sensor原始数据标准差的数组
    :param dataSmooth: 【Array】平滑过的sensor原始数据
    :return:
    '''
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Sensor Viewer")
    win.resize(900, 800)
    win.setWindowTitle(str(sensorDict)[1: -1])
    pg.setConfigOptions(antialias=True)

    n = Queue()
    curves = []
    datas = []  # [s1_x_Origin, s1_x_Smooth, s1_y_Origin, s1_y_Smooth, s1_z_Origin, s1_z_Smooth, ... ]
    curveSigma = []
    dataSigma = []  # [s1_x_sigma, s1_y_sigma, s1_z_sigma, ...]

    def multiCurve(sensorName):
        '''
        绘制多线图, 观察原始数据和标准差
        :param sensorName: 【string】 sensor名称
        :return:
        '''
        colours = {'x': 'b', 'y': 'g', 'z': 'r'}
        if sensorName == 'accelerometer':
            units = 'm/s^2'
            label = 'a'
        elif sensorName == 'gyroscope':
            units = 'deg/s'
            label = 'w'
        elif sensorName.startswith('magSensor'):
            units = 'Gs'
            label = 'B'
        else:
            raise NameError("sensor output is not correct!")

        for i in range(2):
            if i == 0:
                p = win.addPlot(title=sensorName)
            elif i and data0Sigma:
                p = win.addPlot(title=sensorName + '_std')
            else:
                return
            p.addLegend(offset=(1, 1))
            p.setLabel('left', label, units=units)
            p.setLabel('bottom', 'points', units='1')
            p.showGrid(x=True, y=True)
            for axis in ['x', 'y', 'z']:
                if i and data0Sigma:
                    cSigma = p.plot(pen=colours[axis], name=axis)
                    curveSigma.append(cSigma)
                    dataSigma.append(Queue())
                elif i == 0:
                    cOrigin = p.plot(pen=colours[axis], name=axis)
                    # cPredict = p.plot(pen='g', name='Smooth')
                    curves.append(cOrigin)
                    # curves.append(cPredict)
                    datas.append(Queue())  # origin
                    # datas.append(Queue())  # smooth

    def singleCurve(sensorName):
        '''
        绘制单线图，观察原始数据和平滑数据
        :param sensorName: 【string】 sensor名称
        :return:
        '''
        colours = {'x': 'b', 'y': 'g', 'z': 'r'}
        if sensorName == 'accelerometer':
            units = 'm/s^2'
            label = 'a'
        elif sensorName == 'gyroscope':
            units = 'deg/s'
            label = 'w'
        elif sensorName.startswith('magSensor'):
            units = 'Gs'
            label = 'B'
        else:
            raise NameError("sensor output is not correct!")

        for axis in ['x', 'y', 'z']:
            p = win.addPlot(name=sensorName, title=sensorName + '_' + axis)
            # p.addLegend()
            p.setLabel('left', label, units=units)
            p.setLabel('bottom', 'points', units='1')
            p.showGrid(x=True, y=True)
            cOrigin = p.plot(pen=colours[axis])
            # cPredict = p.plot(pen='g', name='Smooth')
            curves.append(cOrigin)
            # curves.append(cPredict)
            datas.append(Queue())  # origin
            # datas.append(Queue())  # smooth

    if 'magSensor1' in sensorDict.keys():
        multiCurve('magSensor1')
        multiCurve('magSensor2')
        win.nextRow()
    if 'imu' in sensorDict.keys():
        multiCurve('accelerometer')
        # win.nextRow()
        multiCurve('gyroscope')

    i = 1
    def update():
        nonlocal i

        for _ in range(4):
            n.put(i)
            i += 1
        sensorNum = len(data0) // 4
        for dataRow in range(sensorNum):
            for dataCol in range(4):
                datas[dataRow].put(data0[dataRow + dataCol * sensorNum])
                if data0Sigma:
                    dataSigma[dataRow].put(data0Sigma[dataRow + dataCol * sensorNum])

        if i > 200:
            for _ in range(4):
                n.get()
                for q in datas:
                    q.get()
                for qs in dataSigma:
                    qs.get()
        for (curve, data) in zip(curves, datas):
            curve.setData(n.queue, data.queue)
        if data0Sigma:
            for (curve, data) in zip(curveSigma, dataSigma):
                curve.setData(n.queue, data.queue)

    timer = pg.Qt.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(100)

    if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    track3D(np.array([0, 0, 0.2, 1, 2, 1, 0]))