# -*- coding: utf-8 -*-
# @Time    : 2020/11/10
# @Author  : sunyihuan
# @File    : copy_to_all.py
'''
拷贝所有判断错误、无输出结果图片至统一文件夹

'''
import os
import shutil
from tqdm import tqdm


class copyfile(object):
    def copy2all(self, file_dir, save_dir):
        '''
        拷贝file_dir下所有文件至save_dir
        file_dir下格式为：xxx.jpg
        :param file_dir:
        :param save_dir:
        :return:
        '''
        for j in os.listdir(file_dir):
            shutil.copy(file_dir + "/" + j, save_dir + "/" + j)

    def copydir2all(self, jpg_root, save_root):
        '''
        拷贝文件夹下所有文件至统一文件夹
        jpg_root格式为：   cls1
                             xxx.jpg
        :return:
        '''
        for c in os.listdir(jpg_root):
            self.copy2all(jpg_root + "/" + c, save_root)

    def copydir2savedir(self, jpg_root, save_root):
        '''
        拷贝文件夹下所有文件至对应类别文件件
        :return:
        '''
        for c in os.listdir(jpg_root):
            if not os.path.exists(save_root + "/" + c): os.mkdir(save_root + "/" + c)
            self.copy2all(jpg_root + "/" + c, save_root + "/" + c)

    def copyspecialfile2dir(self, src, dst, target_str):
        '''
        将文件夹src中，带有target_str的图片拷贝至dst文件夹
        :param src:
        :param dst:
        :param target_str:
        :return:
        '''
        for f in tqdm(os.listdir(src)):
            if target_str in f:
                shutil.copy(src + "/" + f, dst + "/" + f)

    def copyvaljpg_fromtxt(self, jpg_root, txt_dir, typ, jpg_save_root):
        '''
        根据txt_dir中val.txt中数据，拷贝val数据至对应文件夹
        输出文件格式为     val
                            xxxx
                               yyy.jpg
                            ……
        :param src:
        :param dst:
        :param target_str:
        :return:
        '''
        for txt in tqdm(os.listdir(txt_dir)):
            if "_{}".format(typ) in txt:
                save_name = jpg_save_root + "/" + txt.split("_{}".format(typ))[0]
                if not os.path.exists(save_name): os.mkdir(save_name)
                val_jpg_names = open(txt_dir + "/" + txt).readlines()
                for jpg_name in val_jpg_names:
                    jpg_name = jpg_name.strip("\n")
                    jpg_name = jpg_name + ".jpg"
                    print(jpg_root + "/" + jpg_name)
                    print(save_name + "/" + jpg_name)
                    shutil.copy(jpg_root + "/" + jpg_name, save_name + "/" + jpg_name)

    def copyfilefromdir(self, src_dir, target_dir, save_dir):
        '''
        将src_dir中图片，若其在target_dir中，则拷贝至save_dir
        src_dir中文件格式：xxx.jpg
        target_dir中文件格式：   cls0
                                   xxx.jpg
        save_dir中文件格式：xxx.jpg
        :param src_dir:
        :param target_dir:
        :param save_dir:
        :return:
        '''
        # 所有图片列表
        file_list = {}
        for c in os.listdir(target_dir):
            file_dir = target_dir + "/" + c
            if c.endswith(".xls"): continue
            for jpg in os.listdir(file_dir):
                if not jpg.endswith(".jpg"): continue
                file_path = file_dir + "/" + jpg
                file_list[jpg] = file_path
        for k in os.listdir(src_dir):
            if not k.endswith(".jpg"): continue
            if k in file_list.keys():
                shutil.copy(file_list[k], save_dir + "/" + k)


if __name__ == "__main__":
    c = copyfile()

    img_root = "F:/Test_set/ZG/testset"
    save_root = "F:/Test_set/ZG/testset_results/no_result_multi_0517_75_score80_di_wan(终版)_all"
    dst_root = "F:/Test_set/ZG/testset_results/no_result_multi_0517_75_score80_di_wan(终版)_all_original"
    if not os.path.exists(dst_root): os.mkdir(dst_root)
    # c.copydir2all(img_root, save_root)
    c.copyfilefromdir(save_root, img_root, dst_root)
