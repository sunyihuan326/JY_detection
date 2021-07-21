"""preprocess pascal_voc data
"""
import os
import xml.etree.ElementTree as ET
import struct
import numpy as np

classes_name = ["flip-flops(black)", "dustbin(plastic)", "dishcloth(green)", "dustbin(stainless)", "slipper(grey)",
                "dishcloth(blue)", "gym-shoes(black)",
                "cotton-socks(black)", "stockings(flesh)", "socks(khaki)", "power-cord(white)", "power-cord(black)",
                "network-cable(grey)", "None"]

classes_num = {'flip-flops(black)': 1, 'dustbin(plastic)': 2, 'dishcloth(green)': 3, 'dustbin(stainless)': 4,
               'slipper(grey)': 5, 'dishcloth(blue)': 6,
               'gym-shoes(black)': 7, 'cotton-socks(black)': 8, 'stockings(flesh)': 9, 'socks(khaki)': 10,
               'power-cord(white)': 11, 'power-cord(black)': 12,
               'network-cable(grey)': 13, 'None': 14}

YOLO_ROOT = os.path.abspath('./')
DATA_PATH = os.path.join(YOLO_ROOT, 'F:\\model_data\\XDSJ\\2020_data_bai')
OUTPUT_PATH = os.path.join(YOLO_ROOT, 'E:\\JY_detection\\xdsj_detection0\\data\\JoyRobot_test14.txt')


def parse_xml(xml_file):
    """parse xml_file

    Args:
      xml_file: the input xml file path

    Returns:
      image_path: string
      labels: list of [xmin, ymin, xmax, ymax, class]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_path = ''
    labels = []

    for item in root:
        if item.tag == 'filename':
            if item.text[-4:] == '.png':
                item.text = item.text[:-4] + '.jpg'
            image_path = os.path.join(DATA_PATH, 'test\\JoyRobot_1\\JPEGImages', item.text)
        elif item.tag == 'object':
            obj_name = item[0].text
            # obj_num = classes_num[obj_name]
            xmin = int(item[4][0].text)
            ymin = int(item[4][1].text)
            xmax = int(item[4][2].text)
            ymax = int(item[4][3].text)
            labels.append([xmin, ymin, xmax, ymax, obj_name])

        # print("$$$$$$$$$$$$$$$$", image_path, labels)
    return image_path, labels


def convert_to_string(image_path, labels):
    """convert image_path, lables to string
    Returns:
      string
    """
    out_string = ''
    out_string += image_path
    for label in labels:
        for i in label:
            out_string += ' ' + str(i)
    out_string += '\n'
    return out_string


def main():
    out_file = open(OUTPUT_PATH, 'w')

    xml_dir = DATA_PATH + '\\test\\JoyRobot_1\\Annotations\\'

    xml_list = os.listdir(xml_dir)
    xml_list = [xml_dir + temp for temp in xml_list]

    i = 0
    for xml in xml_list:
        try:
            i += 1
            image_path, labels = parse_xml(xml)
            record = convert_to_string(image_path, labels)
            out_file.write(record)
        except Exception:
            print("***ExceptionExceptionException***")
            pass

    out_file.close()


if __name__ == '__main__':
    main()
