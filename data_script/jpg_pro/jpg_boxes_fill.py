# -*- coding: utf-8 -*-
# @Time    : 2021/9/8
# @Author  : sunyihuan
# @File    : jpg_boxes_fill.py
'''
将图片中某块区域填充为其他的图片

'''
import cv2
from tqdm import tqdm
import numpy as np
import os
import xml.etree.ElementTree as ET
from random import choice


def get_bboxes(xml_name):
    '''
    输出一个标注框
    :param xml_name:
    :return:
    '''
    bb = []
    tree = ET.parse(xml_name)
    root = tree.getroot()
    for object1 in root.findall('object'):
        bbox = object1.find('bndbox')
        label_name = object1.find('name').text.lower().strip()
        xmin = bbox.find('xmin').text.strip()
        xmax = bbox.find('xmax').text.strip()
        ymin = bbox.find('ymin').text.strip()
        ymax = bbox.find('ymax').text.strip()
        bb.append([int(xmin), int(ymin), int(xmax), int(ymax),label_name])
    return bb


def jpg_bb_fill(image_file, bb, save_dir):
    samples_dir = "F:/robots_images_202107/samples"
    image = choice(os.listdir(samples_dir))
    image_path = samples_dir + "/" + image
    image0 = cv2.imread(image_path)

    image1 = cv2.imread(image_file)
    image1 = np.array(image1)

    save_name = image_file.split("/")[-1].split(".")[0] + "_" + image.split(".")[0] + ".jpg"

    image0 = cv2.resize(image0, (image1.shape[1], image1.shape[0]))

    image0 = np.array(image0)
    for b in bb:
        for j in range(int(b[3] - b[1])):
            for k in range(int(b[2] - b[0])):
                # print(k)
                image0[int(b[1]) + j, int(b[0]) + k, 0] = image1[int(b[1]) + j, int(b[0]) + k, 0]
                image0[int(b[1]) + j, int(b[0]) + k, 1] = image1[int(b[1]) + j, int(b[0]) + k, 1]
                image0[int(b[1]) + j, int(b[0]) + k, 2] = image1[int(b[1]) + j, int(b[0]) + k, 2]

    cv2.imwrite(save_dir + "/" + save_name, image0)


def dir_fill(image_dir, xml_dir, save_dir):
    for image in tqdm(os.listdir(image_dir)):
        image_file = image_dir + "/" + image
        xml_path = xml_dir + "/" + image.split(".")[0] + ".xml"
        if os.path.exists(xml_path):
            bb = get_bboxes(xml_path)
            jpg_bb_fill(image_file, bb, save_dir)


# if __name__ == "__main__":
#     image_dir = "F:/model_data/XDSJ/202107/JPGImages"
#     save_dir = "F:/model_data/XDSJ/202107/JPGImages_aug"
#     if not os.path.exists(save_dir): os.mkdir(save_dir)
#     xml_dir = "F:/model_data/XDSJ/202107/Annotations"
#     dir_fill(image_dir, xml_dir, save_dir)
