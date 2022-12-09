# coding:utf-8 
'''
统一修改xml文件中label标签

created on 2019/7/18

@author:sunyihuan
'''
import os
import xml.etree.ElementTree as ET


classes_display = {'01': 'shoes', '02': 'dustbin', '03': 'dishcloth',
                   '04': 'dustbin', '05': 'shoes', '06': 'dishcloth',
                   '07': 'shoes', '08': 'socks', '09': 'dishcloth',
                   '10': 'socks', '11': 'line', '12': 'line',
                   '13': 'line', '14': 'None'}

def changesku(inputpath, label_name='Potatom'):
    '''
    更改标签名称
    :param inputpath: xml文件夹地址
    :param label_name: 标签名称
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):
                    # if sku.text=="corn_others":
                    #     print(file)
                    print(sku.text)
                    sku.text = classes_display[sku.text]
                    print(sku.text)
                    tree.write(file, encoding='utf-8')


def check_labelname(inputpath, label_name='Potatom'):
    '''
    检查标签是否有更改完成，标签名不正确的输出
    :param inputpath: xml文件路径
    :param label_name: 标签名
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):
                    # if sku.text=="corn_others":
                    #     print(file)
                    if sku.text == label_name:
                        print(file)
                        print(sku.text)
                        sku.text ="chairleg"
                        print(sku.text)
            tree.write(file, encoding='utf-8')


if __name__ == '__main__':
    inputpath = "F:/RobotProgram/data/chair_leg/annotations"  # 这是xml文件的文件夹的绝对地址
    check_labelname(inputpath, "chair_leg")
    # cls_list = ["Pizzasix", "Pizzafour"]  # Pizza合并
    # cls_list = ["duck"]
    # for c in cls_list:
    #     input_dir = inputpath + "/" + c
    #     changesku(input_dir, "roastedchicken")
