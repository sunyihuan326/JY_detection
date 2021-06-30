# _*_ coding:utf-8 _*_

import xml.etree.ElementTree as ET
import os
import math


# 批量修改整个文件夹所有的xml文件
def change_all_xml(xml_path):
    filelist = os.listdir(xml_path)
    print(filelist)
    # 打开xml文档
    for xmlfile in filelist:
        print(xmlfile)
        print("##########")
        doc = ET.parse(xml_path + xmlfile)
        root = doc.getroot()
        sub0 = root.find('size')  # 找到size标签
        sub1 = root.findall('object')  # 找到object标签
        sub2 = root.find('filename')

        for sub_object in sub1:
            # print(sub_object.findall('bndbox'))
            # print(sub_object.findall('name'))
            sub_object.find("name").text = str('05')

            for i in sub_object.findall('bndbox'):
                print(i.find("xmin").text)
                i.find("xmin").text = str(math.ceil(int(i.find("xmin").text) / int(sub0.find("width").text) * 800))  # 修改xmin标签内容
                i.find("ymin").text = str(math.ceil(int(i.find("ymin").text) / int(sub0.find("height").text) * 600))  # 修改ymin标签内容
                i.find("xmax").text = str(math.ceil(int(i.find("xmax").text) / int(sub0.find("width").text) * 800))  # 修改xmax标签内容
                i.find("ymax").text = str(math.ceil(int(i.find("ymax").text) / int(sub0.find("height").text) * 600))  # 修改ymax标签内容

        sub0.find("width").text = str('800')
        sub0.find("height").text = str('600')

        sub2.text = str(xmlfile[:3] + '3' + xmlfile[4:-4] + '.jpg')
        # print("############", str(xmlfile[:3] + '3' + xmlfile[4:-4] + '.jpg'))

        doc.write(xml_path + xmlfile)  # 保存修改


# 修改某个特定的xml文件
def change_one_xml(xml_path):  # 输入的是这个xml文件的全路径
    # 打开xml文档
    doc = ET.parse(xml_path)
    root = doc.getroot()
    sub1 = root.find('filename')  # 找到filename标签，
    sub1.text = '07_205.jpg'  # 修改标签内容
    doc.write(xml_path)  # 保存修改
    print('----------done--------')

if __name__ == '__main__':
    # change_all_xml(r'Z:\pycharm_projects\ssd\VOC2007\Annotations')     # xml文件总路径
    xml_path = r'C:\\Users\\bai\\Desktop\\SDJ\\Annotation\\05\\'
    # change_one_xml(xml_path)
    change_all_xml(xml_path)
