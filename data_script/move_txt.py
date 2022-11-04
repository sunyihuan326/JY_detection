# -*- coding: utf-8 -*-
# @Time    : 2022/3/4
# @Author  : sunyihuan
# @File    : move_txt.py
'''
文件夹的txt移动到单独文件夹
'''
import os
import shutil
from tqdm import tqdm


def move_txt(all_dir, save_dir):
    '''

    :param all_dir:
    :param save_dir:
    :return:
    '''
    all_list = os.listdir(all_dir)
    for ll in tqdm(all_list):
        if "txt" in ll:
            shutil.move(all_dir + "/" + ll, save_dir + "/" + ll)


if __name__ == "__main__":
    all_dir = "F:/扫地机项目/孙义环-交接资料202203/data/all_model_data/Annotations"
    save_dir = "F:/扫地机项目/孙义环-交接资料202203/data/all_model_data/annotations-txt"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    move_txt(all_dir, save_dir)
