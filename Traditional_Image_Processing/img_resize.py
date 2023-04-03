# -*- coding: utf-8 -*-
# @Time    : 2020/6/11
# @Author  : sunyihuan
# @File    : img_resize.py
import os

from PIL import Image

img_dir = "F:/model_data/FOOD/2023/03/aug_data/samples"

for im in os.listdir(img_dir):
    img_path = img_dir + "/" + im
    img = Image.open(img_path)
    img_new = img.resize((1000, 1000), Image.ANTIALIAS)  # 图片尺寸变化
    if img_new.mode == "RGBA": img_new = img_new.convert('RGB')
    img_new.save(img_path)
