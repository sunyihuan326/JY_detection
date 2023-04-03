# -*- coding: utf-8 -*-
# @Time    : 2022/12/29
# @Author  : sunyihuan
# @File    : imgLight.py


import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm


def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst


if __name__ == "__main__":
    root_dir = "F:/model_data/XDSJ/all_data/20221115"
    img_dir = root_dir + "/JPGImages"
    xml_dir = root_dir + "/Annotations"

    img_save_dir = root_dir + "/JPGImages_light"
    xml_save_dir = root_dir + "/Annotations_light"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)

    for f in tqdm(os.listdir(img_dir)):
        f_name = f.split(".")[0]
        if not os.path.exists(img_save_dir + "/" + f_name + "_light.jpg"):
            img_path = img_dir + "/" + f
            xml_path = xml_dir + "/" + f_name + ".xml"

            img = cv2.imread(img_path)
            dst = imgBrightness(img, 1.3, 0)

            cv2.imwrite(img_save_dir + "/" + f_name + "_light.jpg", dst)

            shutil.copy(xml_dir + "/" + f_name + ".xml", xml_save_dir + "/" + f_name + "_light.xml")
