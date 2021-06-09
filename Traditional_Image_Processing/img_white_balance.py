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
    img = img.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.sum(img, 2)
    su= sum_.flatten()
    print(su)
    sum_ = np.sort(su)
    print(sum_)
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1

    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r

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


if __name__ == '__main__':
    img_path = "C:/Users/sunyihuan/Desktop/t/11.jpg"

    start_time = time.time()  # 开始时间
    img = cv2.imread(img_path, 1)
    Color_temperature_wb(img)
