# -*- coding: utf-8 -*-
# @Time    : 2021/5/25
# @Author  : sunyihuan
# @File    : get_zg_testset_brightness.py
'''
输出所有zg测试集brightness值
并保存至excel中
'''
from PIL import Image, ImageStat
import xlwt, xlrd
import os


def get_bright(img):
    '''
    获取该区域亮度
    :param img:
    :param crop_size:
    :return:
    '''
    crop_size = [160, 310, 900, 745]
    img = img.crop(crop_size)
    img = img.convert("YCbCr")
    start = ImageStat.Stat(img)
    # print(start.mean[0])
    return start.mean[0]


def writ_b_excel(img_list, excel_save):
    w = xlwt.Workbook()
    sheet = w.add_sheet("jpg_b")
    sheet.write(0, 0, "jpg_name")
    sheet.write(0, 1, "brightness")
    for i, img_p in enumerate(img_list):
        print(i, img_p)
        sheet.write(i + 1, 0, img_p)
        try:
            img = Image.open(img_p)
            b = get_bright(img)
            sheet.write(i + 1, 1, b)
        except:
            print("error::::::")
    w.save(excel_save)


if __name__ == "__main__":
    img_dir = "F:/Test_set/ZG/202104_test"
    excel_save = "F:/Test_set/ZG/202104_test/brightness.xls"
    img_list = []
    for c in os.listdir(img_dir):
        c_list = os.listdir(img_dir + "/" + c)
        c_list = [img_dir + "/" + c + "/" + i for i in c_list]
        img_list = list(set(img_list) | set(c_list))
    print(len(img_list))
    writ_b_excel(img_list, excel_save)
