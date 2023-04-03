# -*- coding: utf-8 -*-
# @Time    : 2022/12/19
# @Author  : sunyihuan
# @File    : copyfromU.py

'''
从U盘拷贝数据至本地
'''
import os
import shutil
from tqdm import tqdm
import time


class copy_data(object):
    def __init__(self, data_src, data_dst):
        self.data_src = data_src
        self.data_dst = data_dst

    def copy_img_from_datetime(self, start_time, end_time):
        '''
        按文件名中时间戳拷贝数据
        :param start_time:
        :param end_time:
        :return:
        '''
        all_len = len(os.listdir(self.data_src))
        print("all data nums:::::", all_len)

        c = 0
        for file_name in tqdm(os.listdir(self.data_src)):
            datetime = file_name.split("_")[0]
            if int(datetime) < end_time and int(datetime) >= start_time:
                shutil.copy(self.data_src + "/" + file_name, self.data_dst + "/" + file_name)
                c += 1
        print("拷贝图片数量：", c)


if __name__ == "__main__":
    data_src = "G:/img"
    data_dst = "F:/RobotProgram/data/test_data_yolov5m/running/img1222"
    if not os.path.exists(data_dst): os.mkdir(data_dst)
    start_time = 0
    end_time = time.time()
    cd = copy_data(data_src, data_dst)

    cd.copy_img_from_datetime(start_time, end_time)
