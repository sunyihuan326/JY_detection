# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : sunyihuan
# @File    : image_gamma.py
import numpy as np
import cv2


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


if __name__ == "__main__":
    img_path = "C:/Users/sunyihuan/Desktop/t/20210409102138_pizzacut.jpg"
    img = cv2.imread(img_path, 1)
    cv2.imshow("img", img)
    image = gamma_trans(img,0.8)
    cv2.imshow("image", image)
    cv2.waitKey(0)
