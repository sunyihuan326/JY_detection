# -*- coding: utf-8 -*-
# @Time    : 2021/5/17
# @Author  : sunyihuan
# @File    : jpg_reisze.py
'''
批量resize图片
'''
import shutil, os
import cv2
from tqdm import tqdm


def resize_dir(jpg_dir, target_size, save_dir):
    for jpg in os.listdir(jpg_dir):
        if jpg.endswith("png") or jpg.endswith("jpg"):
            img = cv2.imread(jpg_dir + "/" + jpg)
            img = cv2.resize(img, target_size)
            cv2.imwrite(save_dir + "/{}".format(jpg), img)


if __name__ == "__main__":
    img_root = "F:/Test_set/ZG/testset"
    target_size = (800, 600)
    save_root = "F:/Test_set/ZG/testset_800"
    if not os.path.exists(save_root): os.mkdir(save_root)
    for img_d in tqdm(os.listdir(img_root)):
        jpg_dir = img_root + "/" + img_d
        save_dir = save_root + "/" + img_d
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        resize_dir(jpg_dir, target_size, save_dir)
