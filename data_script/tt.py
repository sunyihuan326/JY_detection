# -*- encoding: utf-8 -*-

"""
@File    : tt.py
@Time    : 2019/12/6 15:51
@Author  : sunyihuan
"""

from data_script.append_data_for_model.data_process_aug import *

txt_list = ["E:/DataSets/model_data/2phase_data2020/train42.txt"]
ap = append_txt(txt_list)
ap.append_txt2all("", 3)
