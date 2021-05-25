# -*- coding: utf-8 -*-
# @Time    : 2021/5/20
# @Author  : sunyihuan
# @File    : fugai.py
'''
覆盖率测算
'''
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img_path = "F:/fugai/89.67.jpg"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# b, g, r = cv2.split(img)
# print(sum(b))
# print(sum(sum(g)))
# print(sum(r))
ret, thresh1 = cv2.threshold(img, 212, 255, cv2.THRESH_BINARY)
plt.imshow(thresh1, 'gray')
plt.show()
