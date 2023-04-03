# -*- coding: utf-8 -*-
# @Time    : 2022/11/7
# @Author  : sunyihuan
# @File    : extract_data_from_all.py

'''
从所有数据中取部分
'''
import os
import shutil


def extract_data(all_data_dir, save_dir, k):
    '''
    所有数据中，隔k个取一个，保存到save_dir中
    :param all_data_dir:
    :param save_dir:
    :param k:
    :return:
    '''
    data_list = os.listdir(all_data_dir)
    for i, d in enumerate(data_list):
        if i % k == 0:
            shutil.move(all_data_dir + "/" + d, save_dir + "/" + d)


def extract_img_and_xml(root_dir, save_dir, k):
    '''

    :param root_dir:
    :param save_dir:
    :param k:
    :return:
    '''
    img_dir = root_dir + "/JPGImages"
    xml_dir = root_dir + "/Annotations"

    save_img_dir = save_dir + "/JPGImages"
    save_xml_dir = save_dir + "/Annotations"
    if not os.path.exists(save_img_dir): os.mkdir(save_img_dir)
    if not os.path.exists(save_xml_dir): os.mkdir(save_xml_dir)

    img_list = os.listdir(img_dir)
    for i, d in enumerate(img_list):
        if i % k == 0:
            # 拷贝图片
            shutil.copy(img_dir + "/" + d, save_img_dir + "/" + d)
            # 拷贝xml
            shutil.copy(xml_dir + "/" + d.split(".")[0] + ".xml",
                        save_xml_dir + "/" + d.split(".")[0] + ".xml")


if __name__ == "__main__":
    all_data_dir = "F:/model_data/XDSJ/all_data/20221115/data_light"
    save_dir = "F:/model_data/XDSJ/all_data/20221115/data_light_use"
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    k = 5
    extract_img_and_xml(all_data_dir, save_dir, k)
