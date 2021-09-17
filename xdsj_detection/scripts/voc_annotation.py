# -*- coding: utf-8 -*-
import os
import argparse
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
from xdsj_detection.core.config import cfg

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):
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
                    print(class_ind)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        default="F:/model_data/XDSJ/2020_data_bai")
    parser.add_argument("--train_annotation",
                        default="F:/model_data/XDSJ/2020_data_bai/train14.txt")
    parser.add_argument("--test_annotation",
                        default="F:/model_data/XDSJ/2020_data_bai/test14.txt")
    # parser.add_argument("--val_annotation",
    #                     default="E:/DataSets/2020_two_phase_KXData/only2phase_data/val18.txt")
    flags = parser.parse_args()
    #
    if os.path.exists(flags.train_annotation): os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation): os.remove(flags.test_annotation)
    # if os.path.exists(flags.val_annotation): os.remove(flags.val_annotation)
    # # #
    num1 = convert_voc_annotation(flags.data_path, 'train_8',
                                  flags.train_annotation, False)
    num2 = convert_voc_annotation(flags.data_path, 'test_8',
                                  flags.test_annotation, False)
    # num3 = convert_voc_annotation(flags.data_path, 'val',
    #                               flags.val_annotation, False)
    # print(
    #     '=> The number of image for train is: %d\tThe number of image for test is:%d\tThe number of image for val is:%d' % (
    #         num1, num2, num3))
