# -*- coding: utf-8 -*-
# @Time    : 2022/12/22
# @Author  : sunyihuan
# @File    : split_old_img.py
import os
import shutil
from tqdm import tqdm


def shutil_img_file(src_dir, dst_dir):
    '''
    拷贝数据
    :param a_dir:
    :param b_dir:
    :param c_dir:
    :return:
    '''
    # a_list = os.listdir(a_dir)  # a文件夹下无层级目录
    for c in tqdm(os.listdir(src_dir)):

        if "old" in c:
            shutil.move(src_dir + "/" + c, dst_dir + "/" + c)


if __name__ == "__main__":
    a_dir = "F:/RobotProgram/data/test_data_yolov5m/running/img1230"
    b_dir = "F:/RobotProgram/data/test_data_yolov5m/running/img1230_old"
    if not os.path.exists(b_dir): os.mkdir(b_dir)

    shutil_img_file(a_dir, b_dir)
