# -*- coding: utf-8 -*-
# @Time    : 2021/12/27
# @Author  : sunyihuan
# @File    : img2mp4.py
import numpy as np
import cv2
import os
from tqdm import tqdm

size = (1280, 720)
mp4_path = ""
videowrite = cv2.VideoWriter(r'F:/carpet.mp4', -1, 1, size)  # 20是帧数，size是图片尺寸
img_array = []

img_dir = "F:/carpet"
for filename in os.listdir(img_dir):
    img = cv2.imread(img_dir + "/" + filename)
    if img is None:
        print(filename + " is error!")
        continue
    img_array.append(img)
for i in range(len(os.listdir(img_dir))):
    videowrite.write(img_array[i])
print('end!')