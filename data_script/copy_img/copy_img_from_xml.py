# -*- coding: utf-8 -*-
# @Time    : 2021/11/26
# @Author  : sunyihuan
# @File    : copy_img_from_xml.py
'''
将所有已经标注的有xml文件的img数据拷贝出（主要针对很多模糊图片不标注）

文件格式：
20210915
    10                  -----img文件根目录地址
      1
      2
      3
    10_annotations      -----xml文件根目录地址
      1
      2
      3
'''

import os
import shutil
from tqdm import tqdm

def copy_img2all(img_root, xml_root, img_save_root, xml_save_root):
    '''

    :param img_root:
    :param xml_root:
    :param img_save_root:
    :param xml_save_root:
    :return:
    '''
    for xml_dir in tqdm(os.listdir(xml_root)):
        xml_dir_path = xml_root + "/" + xml_dir
        for xx in os.listdir(xml_dir_path):
            if xx.endswith(".xml"):
                shutil.copy(xml_dir_path + "/" + xx, xml_save_root + "/" + xx)
                shutil.copy(img_root + "/" + xml_dir + "/" + xx.split(".xml")[0]+".jpg",
                            img_save_root + "/" + xx.split(".xml")[0]+".jpg")


if __name__ == "__main__":
    img_root = "F:/robots_images_202109/20210916/10"
    xml_root = "F:/robots_images_202109/20210916/10_annotations"
    img_save_root = "F:/robots_images_202109/20210916/10_use/JPGImages"
    xml_save_root = "F:/robots_images_202109/20210916/10_use/Annotations"
    if not os.path.exists(img_save_root): os.mkdir(img_save_root)
    if not os.path.exists(xml_save_root): os.mkdir(xml_save_root)

    copy_img2all(img_root, xml_root, img_save_root, xml_save_root)
