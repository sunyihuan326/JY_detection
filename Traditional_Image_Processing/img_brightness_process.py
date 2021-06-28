# -*- coding: utf-8 -*-
# @Time    : 2021/5/25
# @Author  : sunyihuan
# @File    : img_brightness_process.py
'''
判断图片亮度，并对过亮的和过暗的调整

'''

from PIL import Image, ImageStat, ImageEnhance
import numpy as np
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def get_bright(img):
    '''
    获取该区域亮度
    :param img:
    :param crop_size:
    :return:
    '''
    crop_size = [160, 310, 900, 745]
    img = img.crop(crop_size)
    img = img.convert("YCbCr")
    start = ImageStat.Stat(img)
    return start.mean[0]


def adjust_bri(img_path):
    '''
    根据亮度值调整
    :param img_path:
    :return:
    '''
    img = Image.open(img_path)
    b = get_bright(img)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if b > 100:  # 针对亮度值b，过大调整
        g = round(b / 100, 2)
        img = gamma_trans(img, g)
    # g
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def dir_bri_pro(src_dir, dst_dir):
    '''
    针对文件中图片变换后保存
    :param src_dir:
    :param dst_dir:
    :return:
    '''
    for jpg in os.listdir(src_dir):
        if jpg.endswith(".jpg"):
            img_path = src_dir + "/" + jpg
            img = adjust_bri(img_path)
            img.save(dst_dir + "/" + jpg)


if __name__ == "__main__":
    jpg_root = "F:/Test_set/ZG/202104_test"
    save_root = "F:/Test_set/ZG/202104_test_gamma"
    if not os.path.exists(save_root): os.mkdir(save_root)
    for jpg_dir in tqdm(os.listdir(jpg_root)):
        if not jpg_dir.endswith(".xls"):
            jpg_dir_ = jpg_root + "/" + jpg_dir
            save_dir_ = save_root + "/" + jpg_dir
            if not os.path.exists(save_dir_): os.mkdir(save_dir_)
            dir_bri_pro(jpg_dir_, save_dir_)
