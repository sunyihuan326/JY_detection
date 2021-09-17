# -*- coding: utf-8 -*-
# @Time    : 2021/9/8
# @Author  : sunyihuan
# @File    : generate_aug_txt.py
'''
生成的图片名称为：xxxx_menting.jpg
结合ImageSets\Main\train.txt和已有train8.txt生成数据增强后的train8_aug.txt
'''
import os


def generate_txt(txt_path, src_txtpath, dst_txtpath, image_aug_dir):
    '''
    :param txt_path: 原txt文件路径
    :param src_txtpath: 更改后txt保存路径
    :param file_path: 图片新地址
    :param typ:
    :return:
    '''
    image_aug_list = os.listdir(image_aug_dir)

    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    dst_file = open(dst_txtpath, "r")
    dst_files = dst_file.readlines()
    print(len(dst_files))

    src_all_list = []
    for txt_file_one in txt_files:
        # image_aug_name = ""
        # src_f = ""
        img_path_name = txt_file_one.strip()
        for image_aug in image_aug_list:
            if img_path_name in image_aug:
                image_aug_name = image_aug  # 拿到aug数据名字
        for dst_name in dst_files:
            if img_path_name in dst_name:
                src_f = dst_name  # 获取原标注数据
        # print(src_f)
        label_data = src_f.split(".jpg")[1]
        # print(label_data)
        src_f = image_aug_dir + "/" + image_aug_name + label_data
        # print(src_f)
        src_all_list.append(src_f)
    #
    file = open(src_txtpath, "w")
    for i in src_all_list:
        file.write(i)


if __name__ == "__main__":
    txt_path = "F:/model_data/XDSJ/202107/ImageSets/Main/train.txt"
    src_txtpath = "F:/model_data/XDSJ/202107/train8_aug.txt"
    dst_txtpath = "F:/model_data/XDSJ/202107/train8.txt"
    image_aug_dir = "F:/model_data/XDSJ/202107/JPGImages_aug"
    generate_txt(txt_path, src_txtpath, dst_txtpath, image_aug_dir)
