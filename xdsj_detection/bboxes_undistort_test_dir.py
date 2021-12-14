# -*- coding: utf-8 -*-
# @Time    : 2021/12/9
# @Author  : sunyihuan
# @File    : bboxes_undistort_test.py
'''
查看图片畸变矫正后，bbox矫正是否准确
'''

from xdsj_detection.distance_and_angle import *
import xdsj_detection.core.utils as utils
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from xdsj_detection.core.config import cfg


def bboxes_undistort_(bboxes_p):
    '''
    标签框畸变矫正处理
    :param bboxes_p:
    :return:
    '''
    bboxes = []
    if len(bboxes_p) > 0:
        for b in bboxes_p:
            # 找上边缘最大的点(即y_min)
            b_up_min = [b[0], 0]
            # print(b_up_min)
            for x_ in range(int(b[0]), int(b[2]), 3):
                b_up = np.array([float(x_), float(b[1])])
                # print("b_up:::::", b_up)
                b0 = bboxes_undistort(b_up)
                # print("b0*****", b0)
                if b0[1] > b_up_min[1]:  # 寻找最大y值
                    b_up_min = b0
            print(b_up_min[0],b_up_min[1])
            y_up = b_up_min[1]
            print("y_up,", y_up)

            # 找下边缘最小的点(即y_max)
            b_bottom_max = [b[2], 720]
            for x_ in range(int(b[0]), int(b[2]), 3):
                b_up = np.array([float(x_), float(b[3])])
                b1 = bboxes_undistort(b_up)
                if b1[1] < b_bottom_max[1]:  # 寻找最小y值
                    b_bottom_max = b1
            y_bottom = b_bottom_max[1]

            # 找左边缘最大的点(即x_min)
            b_left_min = [0, b[1]]
            for y_ in range(int(b[1]), int(b[3]), 3):
                b_up = np.array([float(b[0]), float(y_)])
                b2 = bboxes_undistort(b_up)
                if b2[0] > b_left_min[0]:  # 寻找最大x值
                    b_left_min = b2
            x_left = b_left_min[0]

            # 找右边缘最小的点(即x_max)
            b_right_max = [1280, b[3]]
            for y_ in range(int(b[1]), int(b[3]), 3):
                b_up = np.array([float(b[2]), float(y_)])
                b3 = bboxes_undistort(b_up)
                if b3[0] < b_right_max[0]:  # 寻找最小x值
                    b_right_max = b3
            x_rigt = b_right_max[0]

            b[0] = x_left
            b[1] = y_up
            b[2] = x_rigt
            b[3] = y_bottom
            bboxes.append(b)
    return bboxes


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = []
    with open(class_file_name, 'r') as data:
        for name in data:
            names.append(name.strip('\n'))
    return names


classes = read_class_names(cfg.YOLO.CLASSES)


def get_bboxes(xml_name):
    '''
    输出一个xml文件的标注框
    :param xml_name:
    :return:
    '''
    bb = []
    if xml_name.endswith(".xml"):
        tree = ET.parse(xml_name)
        root = tree.getroot()
        for object1 in root.findall('object'):
            b = []
            bbox = object1.find('bndbox')
            label_name = object1.find('name').text.lower()
            class_ind = classes.index(label_name.strip())

            xmin = bbox.find('xmin').text.strip()
            xmax = bbox.find('xmax').text.strip()
            ymin = bbox.find('ymin').text.strip()
            ymax = bbox.find('ymax').text.strip()
            b.append(int(xmin)), b.append(int(ymin)), b.append(int(xmax)), b.append(int(ymax)), b.append(
                1.0), b.append(class_ind)
            bb.append(b)
    return bb


if __name__ == "__main__":
    img_dir = "F:/model_data/XDSJ/20211126use/JPGImages0"
    anno_dir = "F:/model_data/XDSJ/20211126use/Annotations"

    img_ann_dir = "F:/model_data/XDSJ/20211126use/JPGImages_ann"
    if not os.path.exists(img_ann_dir): os.mkdir(img_ann_dir)
    undistort_img_ann_dir = "F:/model_data/XDSJ/20211126use/JPGImages_ann_un"
    if not os.path.exists(undistort_img_ann_dir): os.mkdir(undistort_img_ann_dir)

    for im in tqdm(os.listdir(img_dir)):
        image_path = img_dir + "/" + im
        if image_path.endswith(".jpg"):
            image = cv2.imread(image_path)  # 图片读取
            image_un = image_undistort(image)  # 图片畸变矫正

            xml_path = anno_dir + "/" + im.split(".jpg")[0] + ".xml"

            bboxes_p = get_bboxes(xml_path)
            print("bboxes_p****", bboxes_p)
            image = utils.draw_bbox(image, bboxes_p, show_label=True)
            cv2.imwrite(img_ann_dir + "/" + im, image)  # 保存原框图

            # bboxes_p=[[1, 436, 495, 484, 1.0, 6]]
            bboxes = bboxes_undistort_(bboxes_p)
            print("bboxes：：：", bboxes)
            image_un = utils.draw_bbox(image_un, bboxes, show_label=True)
            cv2.imwrite(undistort_img_ann_dir + "/" + im, image_un)  # 保存畸变矫正后图
