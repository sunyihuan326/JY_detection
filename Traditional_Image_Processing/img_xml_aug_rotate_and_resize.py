# -*- encoding: utf-8 -*-

"""
@File    : img_aug_rotate.py
@Time    : 2019/11/8 11:45
@Author  : sunyihuan
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import cv2
import math

def getRotatedImg(Pi_angle,img_path,img_write_path):
    img = cv2.imread(img_path)
    rows, cols = img.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D((a, b), Pi_angle, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))  # 旋转后的图像保持大小不变
    cv2.imwrite(img_write_path,rotated_img)
    return a,b

def getRotatedAnno(Pi_angle,a,b,anno_path,anno_write_path):
    tree = ET.parse(anno_path)
    root = tree.getroot()
    objects = root.findall("object")
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        x3=x1
        y3=y2
        x4=x2
        y4=y1

        X1 = (x1 - a) * math.cos(Pi_angle) - (y1 - b) * math.sin(Pi_angle) + a
        Y1 = (x1 - a) * math.sin(Pi_angle) + (y1 - b) * math.cos(Pi_angle) + b

        X2 = (x2 - a) * math.cos(Pi_angle) - (y2 - b) * math.sin(Pi_angle) + a
        Y2 = (x2 - a) * math.sin(Pi_angle) + (y2 - b) * math.cos(Pi_angle) + b

        X3 = (x3 - a) * math.cos(Pi_angle) - (y3 - b) * math.sin(Pi_angle) + a
        Y3 = (x3 - a) * math.sin(Pi_angle) + (y3 - b) * math.cos(Pi_angle) + b

        X4 = (x4 - a) * math.cos(Pi_angle) - (y4 - b) * math.sin(Pi_angle) + a
        Y4 = (x4 - a) * math.sin(Pi_angle) + (y4 - b) * math.cos(Pi_angle) + b

        X_MIN=min(X1,X2,X3,X4)
        X_MAX = max(X1, X2, X3, X4)
        Y_MIN = min(Y1, Y2, Y3, Y4)
        Y_MAX = max(Y1, Y2, Y3, Y4)

        bbox.find('xmin').text=str(int(X_MIN))
        bbox.find('ymin').text=str(int(Y_MIN))
        bbox.find('xmax').text=str(int(X_MAX))
        bbox.find('ymax').text=str(int(Y_MAX))

    tree.write(anno_write_path)  # 保存修改后的XML文件

def rotate(angle,img_dir,anno_dir,img_write_dir,anno_write_dir):
    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)

    if not os.path.exists(anno_write_dir):
        os.makedirs(anno_write_dir)

    Pi_angle = -angle * math.pi / 180.0  # 弧度制，后面旋转坐标需要用到，注意负号！！！
    img_names=os.listdir(img_dir)
    for img_name in img_names:
        img_path=os.path.join(img_dir,img_name)
        img_write_path=os.path.join(img_write_dir,img_name[:-4]+'R'+str(angle)+'.jpg')
        #
        anno_path=os.path.join(anno_dir,img_name[:-4]+'.xml')
        anno_write_path = os.path.join(anno_write_dir, img_name[:-4]+'R'+str(angle)+'.xml')
        #
        a,b=getRotatedImg(Pi_angle,img_path,img_write_path)
        getRotatedAnno(Pi_angle,a,b,anno_path,anno_write_path)

def pic_rotate(img_path):
    '''
    图片旋转1-5度
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    rota = random.randint(1, 5)
    img_new = img.rotate(rota, expand=1)
    return img_new


def pic_crop(img_path):
    '''
    图片区域裁剪
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    w, h = img.size
    img_new = img.crop((100, 0, w, h))
    return img_new


def pic_resize(img_path, xml_path):
    '''
    图片拉伸后裁剪
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    w, h = img.size  # 原图尺寸
    resize_wh = (800, 800)  # resize尺寸
    img_new = img.resize(resize_wh)  # 图片resize
    crop_d = (0, 0, 800, 800)  # 裁剪位置
    img_new = img_new.crop(crop_d)  # 图片裁剪

    # xml文件中对应坐标调整
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for object1 in root.findall('object'):
        for sku in object1.findall('bndbox'):
            ymin = sku.find("ymin")
            ymax = sku.find("ymax")
            xmin = sku.find("xmin")
            xmax = sku.find("xmax")
            # resize坐标调整
            xmin.text = str(int(int(xmin.text) * resize_wh[0] / w))
            xmax.text = str(int(int(xmax.text) * resize_wh[0] / w))
            ymin.text = str(int(int(ymin.text) * resize_wh[1] / h))
            ymax.text = str(int(int(ymax.text) * resize_wh[1] / h))

            # 裁剪坐标调整
            # xmin.text = str(max(int(xmin.text) - crop_d[0], 0))
            # xmax.text = str(min(int(xmax.text) - crop_d[0], 800))
            # ymin.text = str(max(int(ymin.text) - crop_d[1], 0))
            # ymax.text = str(min(int(ymax.text) - crop_d[1], 600))

    return np.array(img_new), tree


def img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir):
    '''
    文件夹padding和保存
    :param img_dir:图片文件夹地址
    :param xml_dir: xml文件夹地址
    :param img_save_dir:图片保存文件夹地址
    :param xml_save_dir: xml保存文件夹地址
    :return:
    '''
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.endswith("jpg"):
            img_path = img_dir + "/" + img_file

            img_name = str(img_file).split(".")[0] + "_cropy" + ".jpg"  # 图片名称
            xml_name = xml_dir + "/" + str(img_file).split(".")[0] + ".xml"  # xml文件名称
            xml_save_name = xml_save_dir + "/" + str(img_name).split(".")[0] + ".xml"  # xml文件保存名称
            img, tree = pic_resize(img_path, xml_name)

            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            tree.write(xml_save_name, encoding='utf-8')  # xml文件写入


if __name__ == "__main__":
    img_dir = "F:/model_data/FOOD/2023/03/JPGImages"
    xml_dir = "F:/model_data/FOOD/2023/03/Annotations"
    img_save_dir = "F:/model_data/FOOD/2023/03/aug_data/aug/JPGImages_resize"
    xml_save_dir = "F:/model_data/FOOD/2023/03/aug_data/aug/JPGImages_resize_annotations"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
    #
    # img_dir = 'several/JPEGImages'
    # anno_dir = 'several/Annotations'
    # img_write_dir = 'Rotated/rotated_JPEGImages'
    # anno_write_dir = 'Rotated/rotated_Annotations'

    # rotate(45, img_dir, xml_dir, img_save_dir, xml_save_dir)
