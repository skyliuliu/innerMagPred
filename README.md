# innerMagPred

## 1. 简介
   　　基于python3的内置式磁定位方法，使用内置于胶囊的两个sensor（三轴磁传感器）+一个IMU（惯性测量单元），实现对已知状态的外部大磁体（EPM）的定位，反向确定胶囊在世界坐标系下的状态。

## 2. sensor
+ AK09970d：AKM生产，16bit，1.35*1.35*0.57(mm)
+ TLI493D-W2BW: Infineon生产，12bit，1.13*0.93*0.59(mm)

## 3. IMU
+ LSM6DS3TR-C：ST生产，三轴加速度计+三轴陀螺仪，2.5 mm x 3 mm x 0.83 mm

## 4. 文件组成


| 文件名              | 简介                           |
|:-------------------|:-------------------------------|
| mahonyPredictor.py | Mahony的互补滤波算法             |
| scriptRun.py       | 运行互补滤波算法的脚本            |
| mainLM.py          | 使用LM算法实现的胶囊定位方法      |
| mainUKF.py         | 使用UKF算法实现的胶囊定位方法     |
| predictorViewer.py | 绘图工具，包括定位过程，误差分布等 |
| output.csv         | 互补滤波算法脚本的输出结果        |
