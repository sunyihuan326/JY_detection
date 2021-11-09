# -*- coding: utf-8 -*-
# @Time    : 2021/9/22
# @Author  : sunyihuan
# @File    : split_data_nums.py
'''
将文件夹中数为分为若干个小文件夹
'''
import os
import random
import math
import shutil
from tqdm import tqdm

def split_data_nums(data_root, one_nums):
    '''
    将data_root中文件，按one_nums个分为小文件夹
    :param data_root:
    :param one_nums:
    :return:
    '''
    if not os.path.exists(data_root):
        print("cannot find such directory: " + data_root)
        exit()

    total_xml = []
    for a in os.listdir(data_root):
        if a.endswith(".jpg"):
            total_xml.append(a)

    random.shuffle(total_xml)  # 打乱total_xml
    num = len(total_xml)  # 文件总数

    dir_num = math.ceil(num / one_nums)  # 小文件夹数
    for i in tqdm(range(dir_num)):
        if not os.path.exists(data_root + "/" + str(i + 1)): os.mkdir(data_root + "/" + str(i + 1))
        for j in range(one_nums):
            shutil.move(data_root + "/" + total_xml[i * one_nums + j],
                        data_root + "/" + str(i + 1) + "/" + total_xml[i * one_nums + j])


if __name__ == "__main__":
    data_root = "F:/robots_images_202109/20210916/12"
    split_data_nums(data_root, 500)
