# -*- coding: utf-8 -*-
# @Time    : 2021/11/24
# @Author  : sunyihuan
# @File    : distance_and_angle.py
import numpy as np


def distance_to_camera(img_y):
    """
    计算物体距离
    :param img_y: 测量点的图像y坐标
    :return: 测量点到光心在光轴上的投影距离/m
    """
    # 该反比例函数根据小孔成像及畸变模型推算得出，参数为采集实际数据点后拟合得出
    res = 19.9342652 / (img_y - 240.527121) + 0.00589017564
    return res


def compute_angle(img_x):
    """
    计算物体角度
    :param img_x: 测量点的图像x坐标
    :return: 光心与测量点连线与光轴的夹角在水平方向的投影角度/°
    """
    # 该反正切函数根据小孔成像及畸变模型推算得出，参数为采集实际数据点后拟合得出
    res = np.arctan((img_x - 207.93938873) / 215.84899151) / np.pi * 180
    return res
