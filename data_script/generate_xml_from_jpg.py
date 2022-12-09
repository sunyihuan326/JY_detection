# -*- coding: utf-8 -*-
# @Time    : 2022/11/15
# @Author  : sunyihuan
# @File    : generate_xml_from_jpg.py
import os
import shutil

if __name__ == "__main__":

    JPG_dir = "F:/model_data/XDSJ/all_data/20221115/JPGImages"
    xml_dir = "F:/model_data/XDSJ/all_data/20221115/Annotations"
    #
    # shutil.copytree(xml_dir,"F:/model_data/XDSJ/all_data/20221115/Annotations_0")

    xml_list = [o.split(".")[0] for o in os.listdir(xml_dir)]
    for ff in os.listdir(JPG_dir):
        if "_" in ff and "undistort" not in ff and ff not in xml_list:
            xml = ff.split("_")[0] + ".xml"
            print(xml_dir + "/" + xml, xml_dir + "/" + ff.split(".")[0] + ".xml")
            shutil.copy(xml_dir + "/" + xml, xml_dir + "/" + ff.split(".")[0] + ".xml")
