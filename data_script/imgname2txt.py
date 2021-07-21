# -*- coding: utf-8 -*-
# @Time    : 2021/7/2
# @Author  : sunyihuan
# @File    : imgname2txt.py

'''
将文件夹中图片名字写入txt
'''
import os


def imgname2traintxt(image_dir, txt_path):
    train_all_list = []
    for img in os.listdir(image_dir):
        train_all_list.append(img.split(".")[0] + '\n')

    file = open(txt_path, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == "__main__":
    image_root = "F:/model_data/XDSJ/2020_data_bai/test/JoyRobot_1/JPEGImages_classes"
    ImageSets_txt = "F:/model_data/XDSJ/2020_data_bai/test/JoyRobot_1/ImageSets/Main"
    for c in os.listdir(image_root):
        image_dir = image_root + "/" + c
        txt_path = ImageSets_txt + "/" + "{}_test.txt".format(c)
        imgname2traintxt(image_dir,txt_path)
