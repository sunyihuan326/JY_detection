# -*- coding: utf-8 -*-
# @Time    : 2021/12/14
# @Author  : sunyihuan
# @File    : img_undisort_dir.py
'''
文件夹下图片畸变矫正

'''

import os
from xdsj_detection.distance_and_angle import *
from tqdm import tqdm


def dir_undistort(filedir, savedir):
    for img in os.listdir(filedir):
        img_name = filedir + "/" + img
        if img.endswith("jpg"):
            image = cv2.imread(img_name)
            image_un = image_undistort(image)
            cv2.imwrite(savedir + "/" + img.split(".jpg")[0] + "_undistort.jpg", image_un)


if __name__ == "__main__":
    img_root = "G:/pic/JPGImages/12"
    save_root = "G:/pic/JPGImages/12_undistort"
    for c in tqdm(os.listdir(img_root)):
        img_dir = img_root + "/" + c
        save_dir = save_root + "/" + c
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        dir_undistort(img_dir, save_dir)
