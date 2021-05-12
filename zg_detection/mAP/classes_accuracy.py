#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 
# @Author  : sunyihuan
'''
从ground-truth文件、predicted文件中查看标签的准确率

并将标签判断错误的图片拷贝到error文件中

'''
import os
from sklearn.metrics import confusion_matrix
from detection.core import utils
from detection.core.config import cfg
import xlwt

gt_txt_root = "./ground-truth"
pre_txt_root = "./predicted"

CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
CLASSES = dict(zip(CLASSES.values(), CLASSES.keys()))


def get_accuracy(error_write=True):
    '''
    由txt文件，查看classes准确率
    :param error_write: 是否将错误图片数据写入到error文件中，True/False
    :return: error_,
            error_c,
            acc
    '''

    pre_txt_list = os.listdir(pre_txt_root)

    class_true = []
    class_pre = []
    no_result = {}

    error_c = 0  # 输出标签种类错误的nums
    error_noresults = 0  # 输出无结果的nums
    for pre in pre_txt_list:
        if pre.endswith("txt"):
            with open(os.path.join(pre_txt_root, pre), "r") as f:
                with open(os.path.join(gt_txt_root, pre), "r") as fg:  # 读取真实类别
                    for lin in fg.readlines():
                        true_cc = lin.split(" ")[0]  # 真实结果

                all_lines = f.readlines()
                if len(all_lines) > 0:
                    # 预测结果排序
                    pre_c = {}
                    score_list = []
                    for line in all_lines:
                        c = line.split(" ")[0]
                        if c not in pre_c.keys():
                            pre_c[c] = 1
                        else:
                            pre_c[c] += 1
                        score_list.append(line.split(" ")[1])
                    pre_cc = sorted(pre_c.items(), key=lambda x: x[1], reverse=True)
                    predict_c = list(pre_cc[0])[0]
                    if len(pre_cc) == 1:
                        if predict_c != true_cc:
                            error_c += 1
                    else:
                        if pre_cc[0][1] != pre_cc[1][1]:  # 如果输种类大于1个，最多的只有一类，取数量最多的一个
                            predict_c = pre_cc[0][0]
                            if predict_c != true_cc:
                                error_c += 1
                        else:  # 若最多种类不唯一，取得分最高的
                            predict_c = all_lines[score_list.index(max(score_list))].split(" ")[0]
                            if predict_c != true_cc:
                                error_c += 1
                    class_true.append(CLASSES[true_cc])
                    class_pre.append(CLASSES[predict_c])
                else:  # 无任何结果，error_noresults统计
                    error_noresults += 1
                    print(pre)

                    if true_cc not in no_result.keys():  # no result写入到字典no_result中
                        no_result[true_cc] = 1
                    else:
                        no_result[true_cc] += 1
    # labels = ["0", "1", "2", "3", "4",
    #           "5", "6", "7", "8", "9",
    #           "10", "11", "12", "13", "14",
    #           "15", "16", "17", "18", "19",
    #           "20", "21", "22", "23", "24",
    #           "25", "26", "27", "28", "29",
    #           "30", "31", "32", "33", "34",
    #           "35", "36", "37", "38", "39"]
    matrix = confusion_matrix(y_pred=class_pre, y_true=class_true)
    print(matrix)
    w = xlwt.Workbook()
    sheet = w.add_sheet("confusion_matrix")
    cls = list(CLASSES.keys())
    for i in range(len(cls)):
        sheet.write(i + 1, 0, cls[i])
        sheet.write(0, i + 1, cls[i])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sheet.write(i + 1, j + 1, str(matrix[i, j]))
    w.save("{}_confusion_matrix.xls".format("epoch_100"))

    print("no_result:", no_result)
    return error_c, error_noresults


if __name__ == "__main__":
    error_c, error_noresults = get_accuracy()
    print("标签错误数量：", error_c)
    print("无任何结果输出数量：", error_noresults)
