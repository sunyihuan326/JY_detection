# -*- coding: utf-8 -*-
# @Time    : 2021/10/20
# @Author  : sunyihuan
# @File    : from_txt_copy_xml.py
import shutil
from tqdm import tqdm


def from_model_txt_copy_xml(txt_path, jpg_save_dir):
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
        xml_path = img_path.split("JPGImages")[0] + "Annotations" + img_path.split("JPGImages")[1].split(".jpg")[
            0] + ".xml"
        # print(xml_path)
        shutil.copy(xml_path, jpg_save_dir + img_path.split("JPGImages")[1].split(".jpg")[0] + ".xml")
        # if "aug" in img_path:  # 拷贝图片条件
        #     shutil.copy(img_path, jpg_save_dir + img_path.split("/")[-1])


if __name__ == "__main__":
    txt_path = "F:/model_data/XDSJ/test8_allnew.txt"
    jpg_save_dir = "F:/model_data/XDSJ/test_annotations"
    from_model_txt_copy_xml(txt_path, jpg_save_dir)
