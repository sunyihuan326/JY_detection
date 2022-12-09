# -*- coding: utf-8 -*-
# @Time    : 2022/11/11
# @Author  : sunyihuan
# @File    : copy_and_sub_img.py

import os
import shutil
from tqdm import tqdm


def copy_xmlimg2use(img_dir, xml_dir, img_all_dir):
    '''
    若数据已标注，img_dir中没有，则从总文件夹中拷贝至img_dir文件夹中
    :param img_dir:
    :param xml_root:
    :param img_all_dir:
    :return:
    '''
    img_list = [i.split(".")[0] for i in os.listdir(img_dir)]
    for xml_ in tqdm(os.listdir(xml_dir)):
        fname = xml_.split(".")[0]
        if fname not in img_list:
            img_name = fname + ".jpg"
            try:
                shutil.move(img_all_dir + "/" + img_name, img_dir + "/" + img_name)
            except:
                print(fname)


if __name__ == "__main__":
    img_dir = "F:/RobotProgram/data/chair_leg/use"
    xml_dir = "F:/RobotProgram/data/chair_leg/annotations"
    img_all_dir = "F:/RobotProgram/data/chair_leg/office"
    copy_xmlimg2use(img_dir, xml_dir, img_all_dir)
