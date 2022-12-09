# -*- coding: utf-8 -*-
'''
一键加入数据
'''

import os
import argparse
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
from xdsj_detection.core.config import cfg
import shutil


def delete_xmljpg_diff(img_dir, xml_dir, cut_save_dir):
    '''
    删除多余的xml文件和jpg文件
    :param img_dir: 图片地址
    :param xml_dir: xml文件标注地址
    :return:
    '''
    xml_name_list = os.listdir(xml_dir)
    img_name_list = os.listdir(img_dir)

    xml_cut_save_dir = cut_save_dir + "/Annotations"
    jpg_cut_save_dir = cut_save_dir + "/JPGImages"
    if not os.path.exists(xml_cut_save_dir): os.mkdir(xml_cut_save_dir)
    if not os.path.exists(jpg_cut_save_dir): os.mkdir(jpg_cut_save_dir)

    # jpg中有,xml中没有
    print("图片总数：", len(img_name_list))
    print("未标注图片名称：")
    for i in img_name_list:
        try:
            if not i.endswith(".jpg"):
                os.remove(img_dir + "/" + i)
            if str(i.split(".jpg")[0] + ".xml") not in xml_name_list:
                print(img_dir + "/" + i)
                shutil.move(img_dir + "/" + i, jpg_cut_save_dir + "/" + i)
        except:
            print(img_dir + "/" + i)

    # xml中有，jpg中没有的
    print("已标注总数：", len(xml_name_list))
    print("已标注，但图片已删除名称：")
    for i in xml_name_list:
        # print(i)
        if not i.endswith(".xml"):
            os.remove(xml_dir + "/" + i)
        else:
            if str(i.split(".xml")[0] + ".jpg") not in img_name_list:
                print(xml_dir + "/" + i)
                shutil.move(xml_dir + "/" + i, xml_cut_save_dir + "/" + i)

def delete_xml_jpg(jpg_dir, xml_dir, cut_save_dir):  # step1：删除无框数据
    '''
    删除无标签框图片和xml文件
    :param inputpath: xml文件夹地址
    :return:
    '''
    xml_cut_save_dir = cut_save_dir + "/Annotations"
    jpg_cut_save_dir = cut_save_dir + "/JPGImages"
    if not os.path.exists(xml_cut_save_dir): os.mkdir(xml_cut_save_dir)
    if not os.path.exists(jpg_cut_save_dir): os.mkdir(jpg_cut_save_dir)

    listdir = os.listdir(xml_dir)
    for file in tqdm(listdir):
        if file.endswith('xml'):
            file_path = os.path.join(xml_dir, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            bboxes_nums = len(root.findall('object'))
            if bboxes_nums < 1:
                print(file)
                shutil.move(file_path, xml_cut_save_dir + "/" + file)
                if os.path.exists(jpg_dir + "/" + file.split(".")[0] + ".jpg"):
                    shutil.move(jpg_dir + "/" + file.split(".")[0] + ".jpg",
                                jpg_cut_save_dir + "/" + file.split(".")[0] + ".jpg")


def split_data(data_root, test_percent, val_percent):  # step2：将所有图片分为train、test
    '''
    数据分为train、test
    （所有jpg在一个文件中，不分小类别）
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
    ftest = open(txtsavepath + '/test_21.txt', 'w')
    ftrain = open(txtsavepath + '/train_21.txt', 'w')
    fval = open(txtsavepath + '/val_21.txt', 'w')

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


def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):  # step3：生成txt文件
    def read_class_names(class_file_name):
        '''loads class name from a file'''
        names = []
        with open(class_file_name, 'r') as data:
            for name in data:
                names.append(name.strip('\n'))
        return names

    classes = read_class_names(cfg.YOLO.CLASSES)
    print(classes)
    print(len(classes))
    img_inds_file = data_path + '/ImageSets' + '/Main/' + '{}.txt'.format(data_type)
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    random.shuffle(image_inds)

    with open(anno_path, 'a') as f:
        for image_ind in tqdm(image_inds):
            st = "/"
            image_path = (data_path, 'JPGImages', image_ind + '.jpg')  # 原图片
            image_path = st.join(image_path)

            annotation = image_path
            try:
                label_path = (data_path, 'Annotations', image_ind + '.xml')  # 原数据
                label_path = st.join(label_path)
                root = ET.parse(label_path).getroot()
                objects = root.findall('object')
                for obj in objects:
                    bbox = obj.find('bndbox')
                    label_name = obj.find('name').text.lower()

                    class_ind = classes.index(label_name.strip())

                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                f.write(annotation + "\n")
            except:
                print(image_ind)
    return len(image_inds)


if __name__ == '__main__':
    data_root = "F:/model_data/XDSJ/20221111use"

    # step1:去除无框数据
    # img_dir = data_root + "/JPGImages"
    # xml_dir = data_root + "/Annotations"
    # cut_save_dir = data_root + "/use_cut"
    # if not os.path.exists(cut_save_dir): os.mkdir(cut_save_dir)
    # delete_xmljpg_diff(img_dir, xml_dir, cut_save_dir)
    # delete_xml_jpg(img_dir, xml_dir, cut_save_dir)
    #
    # # step2:分train、test
    # test_percent = 0.1
    # val_percent = 0
    # split_data(data_root, test_percent, val_percent)

    # step3：生成train.txt、test.txt
    train_annotation = data_root + "/train21.txt"
    test_annotation = data_root + "/test21.txt"
    num1 = convert_voc_annotation(data_root, 'train_21', train_annotation, False)
    num2 = convert_voc_annotation(data_root, 'test_21', test_annotation, False)
    print("train nums::{},test nums:::{}".format(num1, num2))
