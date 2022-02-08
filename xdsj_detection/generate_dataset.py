# -*- coding: utf-8 -*-
# @Time    : 2022/1/7
# @Author  : sunyihuan
# @File    : generate_dataset.py
'''
生成pb2rknn所需dataset.txt
'''
import os


def generate_txt(img_dir, txt_name):
    src_all_list = []
    for im in os.listdir(img_dir):
        src_all_list.append("./test/{}".format(im)+"\n")

    file = open(txt_name, "w")
    for i in src_all_list:
        file.write(i)


if __name__ == "__main__":
    img_dir = "F:/VMware_ubuntu64/multi_yolov3_predict-20210129/test"
    dataset_path = "F:/VMware_ubuntu64/multi_yolov3_predict-20210129/dataset0.txt"
    generate_txt(img_dir, dataset_path)
