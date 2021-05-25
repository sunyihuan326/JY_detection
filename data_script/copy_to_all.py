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


# img_root = "F:/Test_set/ZG/testset_results"
# detect_root = img_root + "/detect_multi_0517_75_score80_gai"
# food_error_root = img_root + "/food_error_multi_0517_75_score80_gai"
# noresults_root = img_root + "/no_result_multi_0517_75_score80_gai"
#
# save_root = img_root + "/incorrect_imgs"
# save_noresults_detect = save_root + "/noresults_detect"
# save_error_detect = save_root + "/error_detect"
# if not os.path.exists(save_root): os.mkdir(save_root)
# if not os.path.exists(save_noresults_detect): os.mkdir(save_noresults_detect)
# if not os.path.exists(save_error_detect): os.mkdir(save_error_detect)
#
# for c in tqdm((os.listdir(noresults_root))):
#     for img in os.listdir(noresults_root + "/" + c):
#         detect_path = detect_root + "/" + c + "/" + img.split(".jpg")[0] + "_.jpg"
#         if os.path.exists(detect_path):
#             shutil.copy(detect_path, save_noresults_detect + "/" + img)
#         else:
#             print(detect_path)
#
# for c in tqdm((os.listdir(food_error_root))):
#     for img in os.listdir(food_error_root + "/" + c):
#         detect_path = food_error_root + "/" + c + "/" + img
#         if os.path.exists(detect_path):
#             shutil.copy(detect_path, save_error_detect + "/" + img)
#         else:
#             print(detect_path)

def move2all(img_dir, save_dir):
    for c in os.listdir(img_dir):
        if c.endswith(".jpg"):continue
        for img in os.listdir(img_dir + "/" + c):
            if img.endswith(".jpg"):
                img_path = img_dir + "/" + c + "/" + img
                if os.path.exists(img_path):
                    shutil.move(img_path, save_dir + "/" + img)
                else:
                    print(img_path)
            else:
                for img_ in os.listdir(img_dir + "/" + c + "/" + img):
                    img__path = img_dir + "/" + c + "/" + img + "/" + img_
                    if os.path.exists(img__path):
                        shutil.move(img__path, save_dir + "/" + img_)
                    else:
                        print(img__path)


if __name__ == "__main__":
    img_root = "F:/model_data/ZG/202011"
    for kk in tqdm(os.listdir(img_root)):
        img_dir = img_root + "/" + kk
        move2all(img_dir, img_dir)
