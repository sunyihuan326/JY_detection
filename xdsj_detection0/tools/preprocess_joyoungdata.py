"""preprocess pascal_voc data
"""
import os
import xml.etree.ElementTree as ET 
import struct
import numpy as np


classes_name =  ["flip-flops(black)", "dustbin(plastic)", "dishcloth(green)", "dustbin(stainless)", "slipper(grey)", "dishcloth(blue)", "gym-shoes(black)",
                 "cotton-socks(black)", "stockings(flesh)", "socks(khaki)", "power-cord(white)", "power-cord(black)", "network-cable(grey)", "None"]


classes_num = {'flip-flops(black)': 1, 'dustbin(plastic)': 2, 'dishcloth(green)': 3, 'dustbin(stainless)': 4, 'slipper(grey)': 5, 'dishcloth(blue)': 6,
    'gym-shoes(black)': 7, 'cotton-socks(black)': 8, 'stockings(flesh)': 9, 'socks(khaki)': 10, 'power-cord(white)': 11, 'power-cord(black)': 12,
    'network-cable(grey)': 13,  'None': 14}

YOLO_ROOT = os.path.abspath('./')
DATA_PATH = os.path.join(YOLO_ROOT, 'data\\JoyRobot_aug\\')
OUTPUT_PATH = os.path.join(YOLO_ROOT, 'data\\JoyRobot_train14.txt')

def is_float(i):
    if i.count('.') == 1:
            new_i = i.split('.')
            right = new_i[-1]
            left = new_i[0]
            if right.isdigit():
                if left.isdigit():
                    return True
                elif left.count('-')== 1 and left.startswith('-'):
                    new_left = left.split('-')[-1]
                    if new_left.isdigit():
                        return True
    else:
        return False

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
  # print("###########", xml_file)

  for item in root:
    if item.tag == 'filename':
      if item.text[-4:] == '.png':
        item.text = item.text[:-4] + '.jpg'
      image_path = os.path.join(DATA_PATH, 'train\\JoyRobot_1\\JPEGImages', item.text)
      # print("^^^^^^^^", image_path)
    elif item.tag == 'object':
      obj_name = item[0].text
      m = 4
      if(item[2].tag == 'bndbox'):
        m = 2

      # print(item[2].tag)

      # obj_num = classes_num[obj_name]

      if(is_float(item[m][0].text)):
        xmin = item[m][0].text.split('.')[0]
      else:
        xmin = int(item[m][0].text)

      if(is_float(item[m][1].text)):
        ymin = item[m][1].text.split('.')[0]
      else:
        ymin = int(item[m][1].text)

      if(is_float(item[m][2].text)):
        xmax = item[m][2].text.split('.')[0]
      else:
        xmax = int(item[m][2].text)

      if(is_float(item[m][3].text)):
        ymax = item[m][3].text.split('.')[0]
      else:
        ymax = int(item[m][3].text)

      # xmin = int(item[m][0].text)
      # ymin = int(item[m][1].text)
      # xmax = int(item[m][2].text)
      # ymax = int(item[m][3].text)
      # print("&&&&&&&&&&", xmin, ymin, xmax, ymax, obj_name)
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

  xml_dir = DATA_PATH + '\\train\\JoyRobot_1\\Annotations\\'

  xml_list = os.listdir(xml_dir)
  xml_list = [xml_dir + temp for temp in xml_list]

  i = 0
  for xml in xml_list:
    try:
      i += 1
      # print("&&&&&&&&&&", i)
      image_path, labels = parse_xml(xml)
      # print(image_path, labels)
      record = convert_to_string(image_path, labels)
      out_file.write(record)
    except Exception:
      print("***ExceptionExceptionException***")
      pass

  out_file.close()

if __name__ == '__main__':
  main()