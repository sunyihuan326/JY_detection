# -*- coding: utf-8 -*-
# @Time    : 2021/11/29
# @Author  : sunyihuan
# @File    : blur_detetction.py

import numpy
import cv2
import os
import xlwt
import time


def estimate_blur(image: numpy.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    return blur_map, score, bool(score < threshold)


if __name__ == '__main__':
    image_dir = "F:/blur"

    wb = xlwt.Workbook()
    sh = wb.add_sheet("blur_score")
    sh.write(0, 0, "image")
    sh.write(0, 1, "score")

    i = 0
    for img in os.listdir(image_dir):
        image_path = image_dir + "/" + img
        image = cv2.imread(str(image_path))
        st = time.time()
        blur_map, score, blurry = estimate_blur(image, 100)
        en = time.time()
        print("耗时：：：：：：", en - st)
        print(score)
        sh.write(i + 1, 0, img)
        sh.write(i + 1, 1, score)
        i += 1
    wb.save("F:/blur_score.xls")
