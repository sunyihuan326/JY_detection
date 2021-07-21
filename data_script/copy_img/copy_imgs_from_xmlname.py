# coding:utf-8 
'''

按xml文件中的文件名称，从所有图片中拷贝部分至另一个文件夹

created on 2019/7/17

@author:sunyihuan
'''

import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm


class copy_img(object):
    '''
    按xml文件中的文件名称，从所有图片中拷贝部分至另一个文件夹
    '''

    def __init__(self, xml_dir, img_orginal_dir, img_copy_dir):
        '''
        :param xml_dir: xml文件地址（全路径）
        :param img_orginal_dir: 原jpg图片地址，含即所有图片的文件夹
        :param img_copy_dir: xml文件对应jpg图片要保存的文件夹（全路径）
        '''
        self.xml_dir = xml_dir
        self.img_orginal_dir = img_orginal_dir
        self.img_copy_dir = img_copy_dir

    def copy_imgs(self):
        for file in tqdm(os.listdir(self.xml_dir)):
            if str(file).endswith("xml"):
                xml_file = self.xml_dir + "/" + file
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for object1 in root.findall('object'):
                    for sku in object1.findall('name'):
                        label = sku.text
                img_cop_dir = self.img_copy_dir + "/" + label
                if not os.path.exists(img_cop_dir): os.mkdir(img_cop_dir)
                try:
                    img_orginal_file = os.path.join(self.img_orginal_dir, file.split(".")[0] + ".jpg")
                    img_copy_file = os.path.join(img_cop_dir, file.split(".")[0] + ".jpg")
                    shutil.copy(img_orginal_file, img_copy_file)
                except:
                    print(file)


if __name__ == "__main__":
    xml_dir = "F:/model_data/XDSJ/2020_data_bai/train/JoyRobot_1/Annotations_new"
    img_orginal_dir = "F:/model_data/XDSJ/2020_data_bai/train/JoyRobot_1/JPEGImages"
    img_copy_dir = "F:/model_data/XDSJ/2020_data_bai/train/JoyRobot_1/JPEGImages_classes"
    ci = copy_img(xml_dir, img_orginal_dir, img_copy_dir)
    ci.copy_imgs()
