# -*- coding: utf-8 -*-
# @Time    : 2021/8/24
# @Author  : sunyihuan
# @File    : concat_bboxes.py
'''
将各类别标签数据，统一至一个xml文件中
数据格式为：jpgdir
           xmldir
               xxx_dir(类别1的标注文件夹)
               xxx_dir(类别2的标注文件夹)
               xxx_dir(类别3的标注文件夹)
输出为：jpgdir
       xmldir
          xxxxxxxxxx.xml（某张图片标注）
'''
import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import *


def get_bboxes(xml_name, xml_root):
    '''
    输出一个文件的所有标注框
    :param xml_name:
    :param xml_root:
    :return:
    '''
    bb = []
    for xm in os.listdir(xml_root):
        if not xm.endswith(".xml"):
            xm_l = os.listdir(xml_root + "/" + xm)
            if xml_name in xm_l:
                file = os.path.join(xml_root + "/" + xm, xml_name)
                tree = ET.parse(file)
                root = tree.getroot()
                for object1 in root.findall('object'):
                    bbox = object1.find('bndbox')
                    label_name = object1.find('name').text.lower().strip()
                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    bb.append([xmin, ymin, xmax, ymax, label_name])
    return bb


def write2xml(xml_path, bboxes):
    '''
    将所有标注框数据写入到xml文件中
    :param xml_path:
    :param bboxes:
    :return:
    '''
    (img_width, img_height, img_channel) = (1280, 720, 3)
    # 创建一个文档对象
    doc = Document()

    # 创建一个根节点
    root = doc.createElement('annotation')

    # 根节点加入到tree
    doc.appendChild(root)

    # 创建二级节点
    fodler = doc.createElement('fodler')
    fodler.appendChild(doc.createTextNode('1'))  # 添加文本节点

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode('xxxx.jpg'))  # 添加文本节点

    path = doc.createElement('path')
    path.appendChild(doc.createTextNode('./xxxx.jpg'))  # 添加文本节点

    source = doc.createElement('source')
    name = doc.createElement('database')
    name.appendChild(doc.createTextNode('Unknown'))  # 添加文本节点
    source.appendChild(name)  # 添加文本节点

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(img_width)))  # 添加图片width
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(img_height)))  # 添加图片height
    channel = doc.createElement('depth')
    channel.appendChild(doc.createTextNode(str(img_channel)))  # 添加图片channel
    size.appendChild(height)
    size.appendChild(width)
    size.appendChild(channel)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    root.appendChild(fodler)  # fodler加入到根节点
    root.appendChild(filename)  # filename加入到根节点
    root.appendChild(path)  # path加入到根节点
    root.appendChild(source)  # source加入到根节点
    root.appendChild(size)  # source加入到根节点
    root.appendChild(segmented)  # segmented加入到根节点

    for i in range(len(bboxes)):
        object = doc.createElement('object')
        name = doc.createElement('name')
        name.appendChild(doc.createTextNode(str(bboxes[i][-1])))  # 标签名写入
        object.appendChild(name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("0"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(int(bboxes[i][0]))))
        bndbox.appendChild(xmin)
        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(int(bboxes[i][1]))))
        bndbox.appendChild(ymin)
        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(int(bboxes[i][2]))))
        bndbox.appendChild(xmax)
        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(int(bboxes[i][3]))))
        bndbox.appendChild(ymax)
        # difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(bndbox)

        root.appendChild(object)  # object加入到根节点

    # 存成xml文件
    fp = open(xml_path, 'w', encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')
    fp.close()


def concat_all_bboxes(img_dir, xml_root):
    '''
    将文件夹img_dir所有标注数据汇总
    :param img_dir:
    :param xml_root: 标注文件根目录，格式为：xxx_dir(类别1的标注文件夹)
                                           xxx_dir(类别2的标注文件夹)
                                           xxx_dir(类别3的标注文件夹)
    :return:
    '''
    listdir = os.listdir(img_dir)
    for file0 in listdir:
        xml_name = file0.split(".")[0] + ".xml"
        bboxes = get_bboxes(xml_name, xml_root)
        xml_path = xml_root + "/" + xml_name
        if len(bboxes)>0:
            write2xml(xml_path, bboxes)


if __name__ == "__main__":
    img_dir = "F:/robots_images_202107/20210719/10/use"
    xml_root = "F:/robots_images_202107/20210719/10/use_annotations"
    concat_all_bboxes(img_dir, xml_root)
