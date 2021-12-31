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


def from_model_txt_copy_jpg(txt_path, jpg_save_dir, xml_save_dir):
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
        image_name = img_path.split("/")[-1]
        xml_name = image_name.split(".jpg")[0] + ".xml"
        xml_path = img_path.split("JPGImages")[0] + "Annotations/" + xml_name
        try:
            shutil.copy(xml_path, xml_save_dir + "/" + xml_name)
            shutil.copy(img_path, jpg_save_dir + "/" + image_name)
        except:
            print(img_path)


if __name__ == "__main__":
    txt_path = "E:/JY_detection/xdsj_detection/data/dataset/test9_1222.txt"
    jpg_save_dir = "F:/model_data/XDSJ/all_data/20211231/test/JPGImages"
    xml_save_dir = "F:/model_data/XDSJ/all_data/20211231/test/Annotations"
    if not os.path.exists(jpg_save_dir): os.mkdir(jpg_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)

    from_model_txt_copy_jpg(txt_path, jpg_save_dir, xml_save_dir)
