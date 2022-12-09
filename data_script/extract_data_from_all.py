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


if __name__ == "__main__":
    all_data_dir = "F:/RobotProgram/data/line"
    save_dir = "F:/RobotProgram/data/line_use"
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    k = 5
    extract_data(all_data_dir, save_dir, k)
