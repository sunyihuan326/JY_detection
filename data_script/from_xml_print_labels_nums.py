# -*- coding: utf-8 -*-
# @Time    : 2021/8/25
# @Author  : sunyihuan
# @File    : from_xml_print_labels_nums.py

'''
从所有的xml文件中，统计各类标签数量
'''
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


def get_bboxes(xml_dir):
    '''
    输出一个文件的所有标注框
    :param xml_name:
    :param xml_root:
    :return:
    '''
    labels_nums = {}
    for xm in tqdm(os.listdir(xml_dir)):
        if xm.endswith(".xml"):
            file = os.path.join(xml_dir, xm)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                label_name = object1.find('name').text.lower().strip()
                if label_name not in labels_nums.keys():
                    labels_nums[label_name] = 1
                else:
                    labels_nums[label_name] += 1
    return labels_nums


if __name__ == "__main__":
    xml_dir = "F:/model_data/XDSJ/all_data/20221115/Annotations"
    labels_nums = get_bboxes(xml_dir)
    s = 0
    for k in labels_nums.keys():
        s += labels_nums[k]
    print(labels_nums)
    print("总数：：：：", s)
