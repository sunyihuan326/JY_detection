# -*- coding: utf-8 -*-
# @Time    : 2021/8/27
# @Author  : sunyihuan
# @File    : split_train_test.py
import os
import random


def split_data(data_root, test_percent, val_percent):
    '''
    '''
    if not os.path.exists(data_root):
        print("cannot find such directory: " + data_root)
        exit()
    imgfilepath = data_root + "/JPGImages"
    txtsavepath = data_root + "/ImageSets/Main"
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

    total_xml = []
    for a in os.listdir(imgfilepath):
        if a.endswith(".jpg"):
            total_xml.append(a)

    random.shuffle(total_xml)  # 打乱total_xml
    num = len(total_xml)

    te = int(num * test_percent)  # test集数量
    test = total_xml[:te]  # test集列表数据内容
    val = int(num * val_percent)  # val集数量
    val_list = total_xml[:te + val]  # val集列表数据内容
    tr = num - val - te  # val集数量
    print("train size:", tr)
    print("test size:", te)
    print("val size:", val)
    ftest = open(txtsavepath + '/test_8.txt', 'w')
    ftrain = open(txtsavepath + '/train_8.txt', 'w')
    fval = open(txtsavepath + '/val_8.txt', 'w')

    for x in total_xml:
        if str(x).endswith("jpg"):
            name = x[:-4] + '\n'
            if x in test:
                ftest.write(name)
            elif x in val_list:
                fval.write(name)
            else:
                ftrain.write(name)

    ftrain.close()
    ftest.close()
    fval.close()


if __name__ == "__main__":
    data_root = "F:/model_data/XDSJ/2020_data_bai"
    test_percent = 0.2
    val_percent = 0
    split_data(data_root, test_percent, val_percent)
