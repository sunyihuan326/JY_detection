# -*- coding: utf-8 -*-
# @Time    : 2021/6/30
# @Author  : sunyihuan
# @File    : txt_to_txt.py
'''
讲bai之前版本数据集，更改为本模型数据加载使用格式
'''


def txt_file_change(src_txt_path, dst_txt_path):
    txt_file = open(src_txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    train_all_list = []
    for txt_file_one in txt_files:
        txt_file_one = txt_file_one.strip().split(" ")
        image_path = txt_file_one[0]  # 处理图片地址
        t_new_file = image_path
        if len(txt_file_one[1:]) / 5 < 1:
            continue
        else:
            for bb in range(int(len(txt_file_one[1:]) / 5)):
                xmin = txt_file_one[bb * 5 + 1]
                ymin = txt_file_one[bb * 5 + 2]
                xmax = txt_file_one[bb * 5 + 3]
                ymax = txt_file_one[bb * 5 + 4]
                class_ind = int(txt_file_one[bb * 5 + 5])-1
                t_new_file += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
        t_new_file += '\n'
        train_all_list.append(t_new_file)
    file = open(dst_txt_path, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == "__main__":
    src_txt_path = 'E:\\JY_detection\\xdsj_detection(bai)\\data\\JoyRobot_train14.txt'
    dst_txt_path = "E:/JY_detection/xdsj_detection/data/dataset/JoyRobot_train14.txt"
    txt_file_change(src_txt_path, dst_txt_path)
