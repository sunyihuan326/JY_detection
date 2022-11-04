# -*- encoding: utf-8 -*-

"""
从txt文件中读取图片地址，并将图片保存至统一文件夹，含layer数据

@File    : copy_img_from_txt.py
@Time    : 2019/12/5 16:52
@Author  : sunyihuan
"""
import shutil
from tqdm import tqdm
import os


def from_txt_copy_data2all(txt_path, save_dir, jpg_typ, layer_tpy=False):
    '''
    2020年3月20日修改
    :param txt_path:txt文件路径，全路径
    :param save_dir:保存地址
    :param jpg_typ:jpg或者xml，str格式
    :param layer_tpy:是否保存layer数据，与jpg_typ=jpg同时使用
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    if layer_tpy:
        layer_dir = save_dir + "/layer_data"
        # if not os.path.exists(layer_dir):os.mkdir(layer_dir)
        if os.path.exists(layer_dir): shutil.rmtree(layer_dir)
        os.mkdir(layer_dir)
        # 创建各层文件夹
        os.mkdir(layer_dir + "/bottom")
        os.mkdir(layer_dir + "/middle")
        os.mkdir(layer_dir + "/top")
        os.mkdir(layer_dir + "/others")
    if jpg_typ == "jpg":  # 拷贝jpg数据
        for file in tqdm(txt_files):
            img_name = file.strip().split(" ")[0]
            jpg_name = str(img_name).split("/")[-1]

            if "_hot.jpg" not in file and "_zi.jpg" not in file and "_lv.jpg" not in file and "_huang.jpg" not in file:
                shutil.copy(img_name, save_dir + "/" + jpg_name)
                if layer_tpy:
                    if file.split(" ")[1] == "0":
                        # print(layer_dir + "/bottom" + "/" + jpg_name)
                        shutil.copy(img_name, layer_dir + "/bottom" + "/" + jpg_name)
                    elif file.split(" ")[1] == "1":
                        shutil.copy(img_name, layer_dir + "/middle" + "/" + jpg_name)
                    elif file.split(" ")[1] == "2":
                        shutil.copy(img_name, layer_dir + "/top" + "/" + jpg_name)
                    elif file.split(" ")[1] == "3":
                        shutil.copy(img_name, layer_dir + "/others" + "/" + jpg_name)
                    else:
                        print(file)
            # if cls==37:#近拷贝37
            #     shutil.copy(img_name, save_dir + "/" + jpg_name)
            #     if layer_tpy:
            #         if file.split(" ")[1] == "0":
            #             # print(layer_dir + "/bottom" + "/" + jpg_name)
            #             shutil.copy(img_name, layer_dir + "/bottom" + "/" + jpg_name)
            #         elif file.split(" ")[1] == "1":
            #             shutil.copy(img_name, layer_dir + "/middle" + "/" + jpg_name)
            #         elif file.split(" ")[1] == "2":
            #             shutil.copy(img_name, layer_dir + "/top" + "/" + jpg_name)
            #         elif file.split(" ")[1] == "3":
            #             shutil.copy(img_name, layer_dir + "/others" + "/" + jpg_name)
            #         else:
            #             print(file)
    elif jpg_typ == "xml":  # 拷贝xml数据
        for file in tqdm(txt_files):
            img_name = file.split(" ")[0]
            jpg_name = str(img_name).split("/")[-1]
            na = str(jpg_name.split(".jpg")[0]) + ".xml"
            xml_path = img_name.split("JPGImages")[0] + "Annotations/" + na
            shutil.copy(xml_path, save_dir + "/" + na)
            # cls = int(file.strip().split(" ")[-1][-1])
            # if "_hot.jpg" not in file and "_zi.jpg" not in file and "_lv.jpg" not in file and "_huang.jpg" not in file:
            #     # shutil.copy(xml_path, save_dir + "/" + na)
            #     try:
            #         shutil.copy(xml_path, save_dir + "/" + na)
            #     except:
            #         print(xml_path)
            # if cls == 37:
            #     try:
            #         shutil.copy(xml_path, save_dir + "/" + na)
            #     except:
            #         print(na)
    else:
        for file in tqdm(txt_files):
            img_name = file.strip().split(" ")[0]
            jpg_name = str(img_name).split("/")[-1]
            na = str(jpg_name.split(".jpg")[0]) + ".xml"
            xml_path = img_name.split("JPGImages")[0] + "Annotations/" + na

            if not os.path.exists(save_dir + "/JPGImages/"): os.mkdir(save_dir + "/JPGImages/")
            if not os.path.exists(save_dir + "/Annotations/"): os.mkdir(save_dir + "/Annotations/")

            shutil.copy(img_name, save_dir + "/JPGImages/" + jpg_name)
            try:
                shutil.copy(xml_path, save_dir + "/Annotations/" + na)
            except:
                print(save_dir + "/Annotations/" + na)


def from_model_txt_copy_jpg(txt_path, jpg_save_dir):
    '''
    从训练使用txt文件中，拷贝对应的图片至jpg_save_dir中
    :param txt_path:
    :param jpg_save_dir:
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()

    for f in tqdm(txt_files):
        f = f.strip()
        img_path = f.split(" ")[0]
        if "aug" in img_path:  # 拷贝图片条件
            shutil.copy(img_path, jpg_save_dir + img_path.split("/")[-1])


if __name__ == "__main__":
    txt_path = "E:/JY_detection/xdsj_detection/data/dataset/test19_0930.txt"
    save_dir = "F:/model_data/XDSJ/all_data/20221018"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    # if not os.path.exists(save_dir): os.mkdir(save_dir)
    # from_txt_copy_data2all(txt_path, save_dir, "jpg", True)
    #
    # save_dir_anno = "E:/DataSets/X_3660_data/all_data/Annotations_test"
    # if not os.path.exists(save_dir_anno): os.mkdir(save_dir_anno)
    # from_txt_copy_data2all(txt_path, save_dir_anno, "xml", False)

    from_txt_copy_data2all(txt_path, save_dir,"all")
