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
import os


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
    # print(start.mean[0])
    return start.mean[0]


def adjust_bri(img_path):
    '''
    根据亮度值调整
    :param img_path:
    :return:
    '''
    img = Image.open(img_path)
    b = get_bright(img)
    print(b)
    if b > 110:  # 针对亮度值b，过大调整
        g = round(110 / b, 2)
        enh_bri = ImageEnhance.Brightness(img)
        img = enh_bri.enhance(g)
        # img = exposure.adjust_gamma(np.array(img), g)
    elif b < 60:
        g = round(60 / b, 2)
        enh_bri = ImageEnhance.Brightness(img)
        img = enh_bri.enhance(g)
    return img


if __name__ == "__main__":
    jpg_dir = "F:/Test_set/ZG/testset_results/no_result_multi_0517_75_score80_di_wan(终版)_all_original"
    save_dir = "F:/Test_set/ZG/testset_results/no_result_multi_0517_75_score80_di_wan(终版)_all_original_adjust"
    if not os.path.exists(save_dir):os.mkdir(save_dir)
    for jpg in os.listdir(jpg_dir):
        img_path = jpg_dir + "/" + jpg
        # img_path = "./temp_img/20210118141758_babycabbage.jpg"
        img = adjust_bri(img_path)
        # img = Image.fromarray(img)
        img.save(save_dir + "/" + jpg)
