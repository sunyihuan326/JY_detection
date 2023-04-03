# -*- coding: utf-8 -*-
# @Time    : 2022/12/17
# @Author  : sunyihuan
# @File    : png2jpg.py

'''
png转jpg
'''
import os
from tqdm import tqdm

if __name__ == "__main__":
    src = ""
    for f in tqdm(os.listdir(src)):
        if f.endswith(".png"):
            os.rename(src + "/" + f, src + "/" + f.split(".")[0] + ".jpg")
