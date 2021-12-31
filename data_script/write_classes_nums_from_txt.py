# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : sunyihuan
# @File    : write_classes_nums_from_txt.py
'''
从生成的txt文件中，输出各类的数量
从生成的txt文件中，输出各类检测框的数量

'''
import xlwt
id_2_name = {0: "牛排", 1: "卡通饼干", 2: "鸡翅", 3: "戚风蛋糕", 4: "戚风蛋糕", 5: "曲奇饼干"
    , 6: "蔓越莓饼干", 7: "纸杯蛋糕", 8: "蛋挞", 9: "空", 10: "花生米"
    , 11: "披萨", 12: "披萨", 13: "披萨", 14: "排骨", 15: "土豆切"
    , 16: "大土豆", 17: "小土豆", 18: "红薯切", 19: "大红薯", 20: "小红薯"
    , 21: "烤鸡", 22: "吐司", 23: "板栗", 24: "玉米", 25: "玉米"
    , 26: "鸡腿", 27: "芋头", 28: "小馒头", 29: "整个茄子", 30: "切开茄子"
    , 31: "吐司面包", 32: "餐具", 33: "餐具", 34: "鱼", 35: "热狗"
    , 36: "虾", 37: "虾", 38: "烤肉串", 39: "锡纸", 101: "戚风蛋糕"
    , 40: "大土豆", 41: "大红薯"}

class_id = {"dishcloth": 0, "dustbin": 1, "line": 2, "shoes": 3, "socks": 4,
            "None": 5, "carpet": 6, "cup": 7, "station": 8}
class_id = dict(zip(class_id.values(), class_id.keys()))


def cls_bboxes_nums(txt_path):
    '''
    输出各类别检测框数量
    :param txt_path:
    :return:
    '''
    nums_dict = {}
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    for tt in txt_files:
        tt = tt.strip()
        bboxes_all = tt.split(" ")[1:]
        for kk in bboxes_all:
            cls = int(kk.split(",")[-1])
            cls_name = class_id[cls]
            if cls_name not in nums_dict.keys():
                nums_dict[cls_name] = 1
            else:
                nums_dict[cls_name] += 1
    return nums_dict


if __name__ == "__main__":
    txt_path = "F:/model_data/XDSJ/test9_1216.txt"
    w = xlwt.Workbook()
    cls_bboxes_nums_dict = cls_bboxes_nums(txt_path)
    print(cls_bboxes_nums_dict)
