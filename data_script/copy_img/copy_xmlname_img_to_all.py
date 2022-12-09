# coding:utf-8 
'''
拷贝xml文件中，某一类别的数据
含xml文件和image文件
'''

import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm


class copy_img_and_xml(object):
    '''
    按xml文件中的标签名称，从所有图片、xml中拷贝部分至另一个文件夹
    '''

    def __init__(self, xml_dir, target_labelname, img_orginal_dir, img_copy_dir, xml_copy_dir):
        '''
        :param xml_dir: xml文件地址（全路径）
        :param img_orginal_dir: 原jpg图片地址，含即所有图片的文件夹
        :param img_copy_dir: xml文件对应jpg图片要保存的文件夹（全路径）
        '''
        self.xml_dir = xml_dir
        self.img_orginal_dir = img_orginal_dir
        self.img_copy_dir = img_copy_dir
        self.xml_copy_dir = xml_copy_dir
        self.target_labelname = target_labelname

        if not os.path.exists(self.img_copy_dir): os.mkdir(self.img_copy_dir)
        if not os.path.exists(self.xml_copy_dir): os.mkdir(self.xml_copy_dir)

    def copy_imgs_xmls(self):
        c = 0
        for file in tqdm(os.listdir(self.xml_dir)):
            if str(file).endswith("xml"):
                xml_file = self.xml_dir + "/" + file
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for object1 in root.findall('object'):
                    for sku in object1.findall('name'):
                        label = sku.text
                        if self.target_labelname == label and c < 3000:
                            img_file = file.split(".")[0] + ".jpg"

                            img_save_file = file.split(".")[0] + "_cp.jpg"
                            xml_save_file = file.split(".")[0] + "_cp.xml"
                            try:
                                shutil.copy(self.img_orginal_dir + "/" + img_file,
                                            self.img_copy_dir + "/" + img_save_file)
                                shutil.copy(xml_file, self.xml_copy_dir + "/" + xml_save_file)
                                c += 1
                            except:
                                print(img_file, img_save_file)
                                print(xml_file, self.xml_copy_dir + "/" + xml_save_file)


if __name__ == "__main__":
    target_labelname = "carpet"
    xml_dir = "F:/model_data/XDSJ/all_data/20221115/Annotations"
    img_orginal_dir = "F:/model_data/XDSJ/all_data/20221115/JPGImages"
    img_copy_dir = "F:/model_data/XDSJ/all_data/20221115/JPGImages_{}".format(target_labelname)
    xml_copy_dir = "F:/model_data/XDSJ/all_data/20221115/Annotations_{}".format(target_labelname)

    ci = copy_img_and_xml(xml_dir, target_labelname, img_orginal_dir, img_copy_dir, xml_copy_dir)
    ci.copy_imgs_xmls()
