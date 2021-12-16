# innerMagPred

## 1. 简介
   　　基于python3的内置式磁定位方法，使用内置于胶囊的磁传感器+IMU（惯性测量单元），实现对已知状态的外部大磁体（EPM）的定位，反向确定胶囊在世界坐标系下的状态。

## 2. 磁传感器
+ AK09970d：AKM生产，16bit，1.35 x 1.35 x 0.57(mm)

## 3. IMU
+ LSM6DS3TR-C：ST生产，三轴加速度计+三轴陀螺仪，2.5 mm x 3 mm x 0.83 mm

## 4. 文件组成


| 文件名              | 简介                                     |
|:-------------------|:----------------------------------------|
| mahonyPredictor.py | Mahony的互补滤波算法                       |
| scriptRun.py       | 运行互补滤波算法的脚本                      |
| mainLM.py          | 使用LM算法实现的胶囊定位方法                 |
| mainUKF.py         | 使用UKF算法实现的胶囊定位方法                |
| predictorViewer.py | 绘图工具，包括sensor读数、定位过程、误差分布等 |
| output.csv         | 互补滤波算法脚本的输出结果                   |
| readData.py        | 读取串口数据，并进行绘图                    |
| allanFit.py        | 计算真实数据的allan方差，并进行拟合          |
| mainPSO.py         | 使用PSO算法实现的胶囊定位方法                |
| Lie.py             | 李代数的定义类                             |
| se3LM.py           | 基于李代数的LM优化算法                      |


## 5. 软件安装
+ python > 3.8.3
+ 依赖包的版本要求如requirements.txt所示
+ 依赖包安装方法：`pip install -r requirements.txt`
