# -*- coding: utf-8 -*-
# @Time    : 2021/8/11
# @Author  : sunyihuan
# @File    : move_jpg_from_dir.py
import os
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    image_root = "F:/robots_images_202107/20210722/12"

    image_fen = "F:/robots_images_202107/20210722/12/fen"
    if not os.path.exists(image_fen): os.mkdir(image_fen)

    fen_img_list = []
    for d in ["白", "黑", "糊", "清晰-已筛选"]:
        for img in os.listdir(image_root + "/" + d):
            fen_img_list.append(img)
    for k in tqdm(os.listdir(image_root)):
        if k.endswith(".jpg"):
            if k in fen_img_list:
                shutil.move(image_root + "/" + k, image_fen + "/" + k)