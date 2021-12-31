# -*- coding: utf-8 -*-
# @Time    : 2021/12/29
# @Author  : sunyihuan
# @File    : copy_one_class_img_from_txt.py

'''
从txt文件中，copy某一类所有图片
'''
import shutil
from tqdm import tqdm
import os


def from_model_txt_copy_jpg(txt_path, target_label, jpg_save_dir):
    '''
    从训练使用txt文件中，拷贝对应的图片至jpg_save_dir中
    :param txt_path:
    :param jpg_save_dir:
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()

    for f in tqdm(txt_files):
        f = f.strip()
        img_path = f.split(" ")[0]
        for b in f.split(" ")[1:]:
            if int(b[-1]) == target_label:
                try:
                    shutil.copy(img_path, jpg_save_dir + "/" + img_path.split("/")[-1])
                    print("copy!")
                except:
                    print(img_path)


if __name__ == "__main__":
    txt_path = "E:/JY_detection/xdsj_detection/data/dataset/train9_1216.txt"
    save_dir = "F:/all_line"
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    from_model_txt_copy_jpg(txt_path, 2, save_dir)
