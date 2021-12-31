# -*- coding: utf-8 -*-
# @Time    : 2021/12/29
# @Author  : sunyihuan
# @File    : img_undistorted.py

import os
from tqdm import tqdm
from xdsj_detection.distance_and_angle import *

if __name__ == "__main__":
    img_root = "F:/Test_set/STSJ/saved_pictures_202112"  # 图片文件地址
    save_root = "F:/Test_set/STSJ/saved_pictures_202112_undistorted"  # 图片预测后保存地址
    if not os.path.exists(save_root): os.mkdir(save_root)

    for k, c in enumerate(os.listdir(img_root)):
        img_dir = img_root + "/" + c  # 类别文件夹
        save_dir = save_root + "/" + c  # 预测后类别文件夹
        if not os.path.exists(save_dir): os.mkdir(save_dir)

        for img in tqdm(os.listdir(img_dir)):
            img_path = img_dir + "/" + img  # 图片路径
            image = cv2.imread(img_path)  # 图片读取
            image_undis = image_undistort(image)  # 畸变矫正
            cv2.imwrite(save_dir + "/" + img, image_undis)  # 保存畸变矫正图片
