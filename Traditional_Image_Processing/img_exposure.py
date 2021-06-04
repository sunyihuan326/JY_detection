# -*- coding: utf-8 -*-
# @Time    : 2021/6/4
# @Author  : sunyihuan
# @File    : img_exposure.py

import cv2
from PIL import Image
import numpy as np
import os

#
#
# def gamma_trans(img, gamma):  # gamma函数处理
#     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
#     return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
#
#
# def nothing(x):
#     pass
#
# data_base_dir = "F:/no_result_multi_0517_75_all_original"  # 输入文件夹的路径
# outfile_dir = "F:/no_result_multi_0517_75_all_original_exposure"# 输出文件夹的路径
#
#
# for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
#     read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
#     image = cv2.imread(read_img_name)  # 读入图片
#
#     while (1):
#         value_of_gamma = cv2.getTrackbarPos('Value of Gamma', 'demo')  # gamma取值
#         value_of_gamma = value_of_gamma * 0.01  # 压缩gamma范围，以进行精细调整
#         image_gamma_correct = gamma_trans(image, value_of_gamma)  # 2.5为gamma函数的指数值，大于1曝光度下降，大于0小于1曝光度增强
#         cv2.imshow("demo", image_gamma_correct)
#         k = cv2.waitKey(1)
#         if k == 13:  # 按回车键确认处理、保存图片到输出文件夹和读取下一张图片
#             out_img_name = outfile_dir + '//' + file.strip()
#             cv2.imwrite(out_img_name, image_gamma_correct)
#             break
from PIL import Image
import matplotlib.pyplot as plt


# 获取图片
def getimg(image_path):
    return Image.open(image_path)


# 显示图片
def showimg(img, isgray=False):
    plt.axis("off")
    if isgray == True:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

def histeq(imarr):
  hist, bins = np.histogram(imarr, 255)
  cdf = np.cumsum(hist)
  cdf = 255 * (cdf/cdf[-1])
  res = np.interp(imarr.flatten(), bins[:-1], cdf)
  res = res.reshape(imarr.shape)
  return res, hist


def rgb_histeq1(im):
    imarr = np.array(im)
    imarr2 = imarr.flatten()
    hist, bins = np.histogram(imarr2, 255)
    cdf = np.cumsum(hist)
    cdf = 255 * (cdf / cdf[-1])
    imarr3 = np.interp(imarr2, bins[:-1], cdf)
    imarr3 = imarr3.reshape(imarr.shape)
    return Image.fromarray(imarr3.astype('uint8'), mode='RGB')


def rgb_histeq2(im):
    imarr = np.array(im)
    r_arr = imarr[..., 0]
    g_arr = imarr[..., 1]
    b_arr = imarr[..., 2]

    r_res, r_hist = histeq(r_arr)
    g_res, g_hist = histeq(g_arr)
    b_res, b_hist = histeq(b_arr)

    new_imarr = np.zeros(imarr.shape, dtype='uint8')
    new_imarr[..., 0] = r_res
    new_imarr[..., 1] = g_res
    new_imarr[..., 2] = b_res

    return Image.fromarray(new_imarr, mode='RGB')


def rgb_histeq3(im):
    imarr = np.array(im)
    r_arr = imarr[..., 0]
    g_arr = imarr[..., 1]
    b_arr = imarr[..., 2]

    # 取三个通道的平均值
    imarr2 = np.average(imarr, axis=2)
    hist, bins = np.histogram(imarr2, 255)
    cdf = np.cumsum(hist)
    cdf = 255 * (cdf / cdf[-1])

    r_res = np.interp(r_arr, bins[:-1], cdf)
    g_res = np.interp(g_arr, bins[:-1], cdf)
    b_res = np.interp(b_arr, bins[:-1], cdf)

    new_imarr = np.zeros(imarr.shape, dtype="uint8")
    new_imarr[..., 0] = r_res
    new_imarr[..., 1] = g_res
    new_imarr[..., 2] = b_res

    return Image.fromarray(new_imarr, mode='RGB')

image_path = "F:/no_result_multi_0517_75_all_original"
image_save = "F:/no_result_multi_0517_75_all_original_exposure"
for im in os.listdir(image_path):
    print(im)
    img_ = image_path + "/" + im
    img = getimg(img_)
    img2 = rgb_histeq3(img)
    img2.save(image_save + "/" + im)
