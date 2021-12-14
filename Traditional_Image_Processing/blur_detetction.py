# -*- coding: utf-8 -*-
# @Time    : 2021/11/29
# @Author  : sunyihuan
# @File    : blur_detetction.py

import numpy
import cv2
import os
import xlwt
import time
import numpy as np


def detect_blur_fft(image, size=60, thresh=10):
    '''
    值越小越模糊
    :param image:
    :param size:
    :param thresh:
    :return:
    '''
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return mean, mean <= thresh


def diff(pre, img):
    '''
    帧差法
    :param pre:
    :param img:
    :return:
    '''
    dif = cv2.absdiff(pre, img)
    return dif


def estimate_blur(image: numpy.array):
    '''
    值越小越模糊
    :param image:
    :return:
    '''
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imageVar = cv2.Laplacian(image, cv2.CV_64F).var()

    return imageVar


if __name__ == '__main__':
    image_dir = "F:/clear"

    wb = xlwt.Workbook()
    sh = wb.add_sheet("clear_score")
    sh.write(0, 0, "image")
    sh.write(0, 1, "score")
    sh.write(0, 2, "blurry")

    i = 0
    for img in os.listdir(image_dir):
        image_path = image_dir + "/" + img
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st = time.time()
        # imageVar = estimate_blur(image)
        (mean, blurry) = detect_blur_fft(gray)
        en = time.time()
        print("耗时：：：：：：", en - st)
        print(mean)
        sh.write(i + 1, 0, img)
        sh.write(i + 1, 1, mean)
        bb = "no"
        if blurry: bb = "yes"
        sh.write(i + 1, 2, str(bb))
        i += 1
    wb.save("F:/clear_score.xls")
