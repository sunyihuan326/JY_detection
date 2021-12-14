# -*- coding: utf-8 -*-
# @Time    : 2021/12/9
# @Author  : sunyihuan
# @File    : bboxes_undistort_test.py
'''
直线畸变矫正查看
'''
from xdsj_detection.distance_and_angle import *
import xdsj_detection.core.utils as utils
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = []
    y = []
    x0 = []
    y0 = []
    for x_ in range(int(0), int(300), 3):
        b_up = np.array([float(x_), float(200)])
        print(b_up)
        b0 = bboxes_undistort(b_up)
        print(b0)
        x.append(x_), y.append(200)
        x0.append(b0[0]), y0.append(b0[1])
    plt.figure()
    plt.plot(np.array(x), np.array(y), color="red")
    plt.plot(x0, y0, color="blue")
    plt.show()
