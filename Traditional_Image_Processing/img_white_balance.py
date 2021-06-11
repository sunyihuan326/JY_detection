# -*- encoding: utf-8 -*-

"""
单张图片白平衡

@File    : img_white_balance.py
@Time    : 2019/11/19 9:56
@Author  : sunyihuan
"""
import cv2
import numpy as np
import time


def Color_temperature_wb(img):
    '''
    色温估计白平衡
    :param img:
    :return:
    '''
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def gray_average_wb(img):
    '''
    灰度平均白平衡
    :param img:
    :return:
    '''
    dst = np.zeros(img.shape, img.dtype)

    imgB, imgG, imgR = cv2.split(img)

    bAve = cv2.mean(imgB)[0]
    gAve = cv2.mean(imgG)[0]
    rAve = cv2.mean(imgR)[0]

    Ave = (bAve + gAve + rAve) / 3

    KB = Ave / bAve
    KG = Ave / gAve
    KR = Ave / rAve

    # 3使用增益系数
    imgB = imgB * KB  # 向下取整
    imgG = imgG * KG
    imgR = imgR * KR

    imgB = np.clip(imgB, 0, 255)
    imgG = np.clip(imgG, 0, 255)
    imgR = np.clip(imgR, 0, 255)

    dst[:, :, 0] = imgB
    dst[:, :, 1] = imgG
    dst[:, :, 2] = imgR

    return dst


def perfect_flect_wb(img):
    '''
    全反射白平衡
    :param img:
    :return:
    '''
    height, width = img.shape[:2]
    thresh = height * width * 0.1
    sum_array = np.sum(img.copy(), axis=2)
    MaxVal = np.max(sum_array)
    HistRGB = np.bincount(sum_array.reshape(1, -1)[0])
    HistRGB_Sum = np.add.accumulate(HistRGB[::-1])
    Threshold = np.argwhere(HistRGB_Sum > thresh)[0][0]

    Thresh_array = np.where(sum_array > Threshold, 1, 0)
    cnt = np.count_nonzero(Thresh_array)

    Thresh_array = Thresh_array[:, :, np.newaxis].repeat(3, axis=2)
    sumBGR = np.sum(np.multiply(img, Thresh_array), axis=(0, 1))
    AvgB = sumBGR[0] / cnt
    AvgG = sumBGR[1] / cnt
    AvgR = sumBGR[2] / cnt


    dst = np.zeros_like(img, dtype=np.float64)
    dst[:, :, 0] = np.divide(np.multiply(img[:, :, 0], MaxVal), AvgB)
    dst[:, :, 1] = np.divide(np.multiply(img[:, :, 1], MaxVal), AvgG)
    dst[:, :, 2] = np.divide(np.multiply(img[:, :, 2], MaxVal), AvgR)

    # dst = np.where(dst > 255, 255, 0).astype(np.uint8)

    return dst


if __name__ == '__main__':
    img_path = "C:/Users/sunyihuan/Desktop/t/11.jpg"

    start_time = time.time()  # 开始时间
    img = cv2.imread(img_path, 1)
    cv2.imshow("img", img)
    image = perfect_flect_wb(img)
    cv2.imshow("image", image)
    cv2.waitKey(0)
