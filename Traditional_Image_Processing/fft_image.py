# -*- coding: utf-8 -*-
# @Time    : 2021/8/3
# @Author  : sunyihuan
# @File    : fft_image.py

import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def fft_img(iamge_path):
    img = cv.imread(iamge_path, 0)  # 直接读为灰度图像
    f = np.fft.fft2(img)  # 做频率变换
    fshift = np.fft.fftshift(f)  # 转移像素做幅度普
    s1 = np.log(np.abs(fshift))  # 取绝对值：将复数变化成实数取对数的目的为了将数据变化到0-255
    return s1


if __name__ == "__main__":
    img_dir = "F:/robots_images_202107/20210719/10/use"
    save_dir = "F:/robots_images_202107/20210719/10/use_fft"
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    for im in tqdm(os.listdir(img_dir)):
        image_path = img_dir + "/" + im
        s = fft_img(image_path)
        plt.imshow(s)  # Needs to be in row,col order
        plt.savefig(save_dir + "/" + im)
