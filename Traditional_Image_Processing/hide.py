# -*- coding: utf-8 -*-
# @Time    : 2023/1/9
# @Author  : sunyihuan
# @File    : hide.py
'''
遮挡判断
'''
import cv2
import os

"""
        calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) 
        images参数表示输入图像，传入时应该用中括号[ ]括起来
        channels参数表示传入图像的通道，如果是灰度图像，那就不用说了，只有一个通道，值为0，
        如果是彩色图像（有3个通道），那么值为0,1,2,中选择一个，对应着BGR各个通道。这个值也得用[ ]传入。
        mask参数表示掩膜图像。如果统计整幅图，那么为None。
        主要是如果要统计部分图的直方图，就得构造相应的掩膜来计算。
        histSize参数表示灰度级的个数，需要中括号，比如[256]
        ranges参数表示像素值的范围，通常[0,256]。此外，假如channels为[0,1],ranges为[0,256,0,180],
        则代表0通道范围是0-256,1通道范围0-180。
        hist参数表示计算出来的直方图。
"""


def hist_rgb_no0_nums(img):
    color = ('blue', 'green', 'red')  # 图像三通道
    hist = []
    for i, color in enumerate(color):
        hist.append(cv2.calcHist([img], [i], None, [256], [0, 256]))  # 绘制各个通道的直方图

    c = {}
    for i in range(3):
        c[i] = 0
        for j in range(256):
            if hist[i][j] > 2000:
                c[i] += 1
    return c, hist


def hist_gray_no0_nums(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    c = 0
    for i in range(256):
        if hist[i] > 2000:
            c += 1
    return c, hist


targ = "gray"
# img_root = "F:/RobotProgram/data/test_data_yolov5m/running/img1217_old"
img_root = "F:/RobotProgram/data/test_data_yolov5m/running/img_old"
for im in os.listdir(img_root):
    img = cv2.imread(img_root + "/" + im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    c, his = hist_gray_no0_nums(img)
    if c < 100:
        print(im, ":", c)
# plt.hist(src.ravel(),256,color="red")
# plt.show()
#
# cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("input image", src)
#
# plt.show()
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
