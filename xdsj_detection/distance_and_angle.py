# -*- coding: utf-8 -*-
# @Time    : 2021/11/24
# @Author  : sunyihuan
# @File    : distance_and_angle.py
import numpy as np
import cv2
import math


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


Matrix = np.array([[601.62816651, 0., 630.85174465], [0., 601.0869061, 363.24358673], [0., 0., 1.]])
dist = np.array([[-3.35058664e-01, 1.47858607e-01, 7.00833616e-04, -2.55791177e-04, -3.66712540e-02]])


def image_undistort(image):
    '''
    图片畸形矫正，输入尺寸为1280x720

    !!!!!!!!!image size: 1280x720
    :param image:

    :return:
    '''
    dst = cv2.undistort(image, Matrix, dist, None, Matrix)
    # dst = cv2.resize(dst, [448, 448])
    return dst


def bboxes_undistort(bb):
    b = []
    dst_bb = cv2.undistortPoints(bb, Matrix, dist, None, Matrix)

    # 限制取值
    b.append(np.clip(dst_bb[0][0][0], 0, 1280))
    b.append(np.clip(dst_bb[0][0][1], 0, 720))
    return b
