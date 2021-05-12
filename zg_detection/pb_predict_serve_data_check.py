# -*- encoding: utf-8 -*-

"""
@File    : ckpt_all_TXZG.py
@Time    : 2019/12/18 14:39
@Author  : sunyihuan
@Modify  : FreeBird
@Mod-Time: 2020/8/31 09:24
"""

import cv2
import numpy as np
import tensorflow as tf
from detection.core import utils
import os
import shutil
from tqdm import tqdm
import xlwt
import time
from sklearn.metrics import confusion_matrix


def he_foods(pre, c):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    if pre in [29, 30, 31, 32, 33, 34] and classes_id[c] in [29, 30, 31, 32, 33, 34]:  #
        rigth_label = True
    elif pre in [12, 24] and classes_id[c] in [12, 24]:  #
        rigth_label = True
    else:
        rigth_label = False
    return rigth_label


class YoloTest(object):
    def __init__(self):
        self.input_size = 416
        self.num_classes = 38
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.pb_file = "E:/project/zg_detection/detection/model/food38_0129.pb"
        self.write_image = True
        self.show_label = True

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            self.input = self.sess.graph.get_tensor_by_name("input/input_data:0")
            # self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = self.sess.graph.get_tensor_by_name("pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("pred_lbbox/concat_2:0")

    def predict(self, image):
        '''
        预测结果
        :param image: 图片数据，shape为[800,600,3]
        :return:
            bboxes：食材检测预测框结果，格式为：[x_min, y_min, x_max, y_max, probability, cls_id],
        '''
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input: image_data
                # self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def result(self, image_path):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr = self.predict(image)  # 预测结果
        #
        # if self.write_image:
        #     image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
        #     drawed_img_save_to_path = str(image_path).split("/")[-1]
        #     drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + ".jpg"  # 图片保存地址
        #     cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片

        # 预测结果,bboxes_pr输出格式为[x_min, y_min, x_max, y_max, probability, cls_id]

        num_label = len(bboxes_pr)
        # 未检测食材
        if num_label == 0:
            return bboxes_pr

        # 检测到一个食材    
        elif num_label == 1:
            if bboxes_pr[0][4] < 0.45:
                # if bboxes_pr[0][5] == 19:  # 低分花生米
                #     bboxes_pr[0][4] = 0.75
                # elif bboxes_pr[0][5] == 24:  # 低分整鸡
                #     bboxes_pr[0][4] = 0.75
                if bboxes_pr[0][5] == 37:  # 低分nofood
                    bboxes_pr[0][4] = 0.85
                else:
                    del bboxes_pr[0]

            return bboxes_pr

        # 检测到多个食材
        else:
            new_bboxes_pr = []
            for i in range(len(bboxes_pr)):
                if bboxes_pr[i][4] >= 0.3:
                    new_bboxes_pr.append(bboxes_pr[i])

            new_num_label = len(new_bboxes_pr)
            # print(new_num_label)
            # print(new_bboxes_pr)
            same_label = True
            for i in range(new_num_label):
                if i == (new_num_label - 1):
                    break
                if new_bboxes_pr[i][5] == new_bboxes_pr[i + 1][5]:
                    continue
                else:
                    same_label = False

            sumProb = 0.
            # 多个食材，同一标签        
            if same_label:
                new_bboxes_pr[0][4] = 0.98
                return new_bboxes_pr
            # 多个食材，非同一标签
            else:
                problist = list(map(lambda x: x[4], new_bboxes_pr))
                labellist = list(map(lambda x: x[5], new_bboxes_pr))

                labeldict = {}
                for key in labellist:
                    labeldict[key] = labeldict.get(key, 0) + 1
                    # 按同种食材label数量降序排列
                s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

                n_name = len(s_labeldict)
                name1 = s_labeldict[0][0]
                num_name1 = s_labeldict[0][1]
                name2 = s_labeldict[1][0]
                num_name2 = s_labeldict[1][1]

                # 优先处理食材特例
                if n_name == 2:
                    # 如果鸡翅中检测到了排骨，默认单一食材为鸡翅
                    if (name1 == 5 and name2 == 23) or (name1 == 23 and name2 == 5):
                        for i in range(new_num_label):
                            new_bboxes_pr[i][5] = 5
                        return new_bboxes_pr

                    # 如果菜心中检测到遮挡黑暗，默认为异常场景-遮挡黑暗
                    if (name1 == 2 and name2 == 36) or (name1 == 36 and name2 == 2):
                        for i in range(new_num_label):
                            new_bboxes_pr[i][5] = 36
                        return new_bboxes_pr

                    # 如果切开红薯中检测到了红薯，默认单一食材为切开红薯
                    if (name1 == 29 and name2 == 30) or (name1 == 30 and name2 == 29):
                        for i in range(new_num_label):
                            new_bboxes_pr[i][5] = 30
                        return new_bboxes_pr

                # 数量最多label对应的食材占比0.7以上
                if num_name1 / new_num_label > 0.7:
                    name1_bboxes_pr = []
                    for i in range(new_num_label):
                        if name1 == new_bboxes_pr[i][5]:
                            name1_bboxes_pr.append(new_bboxes_pr[i])

                    name1_bboxes_pr[0][4] = 0.95
                    return name1_bboxes_pr

                # 按各个label的probability降序排序
                else:
                    new_bboxes_pr = sorted(new_bboxes_pr, key=lambda x: x[4], reverse=True)
                    for i in range(len(new_bboxes_pr)):
                        new_bboxes_pr[i][4] = new_bboxes_pr[i][4] * 0.9
                    return new_bboxes_pr


if __name__ == '__main__':

    classes_label38 = ["babycabbage",
                       "beefsteak",
                       "cabbage_heart",
                       "chestnut",
                       "chickenwing_root", "chickenwing_middle", "chickenwing_tip",
                       "chips",
                       "cookies",
                       "corn",
                       "crab",
                       "cranberrycookies",
                       "duck",
                       "eggplant",
                       "eggtart",
                       "fish",
                       "lambchops",
                       "mushrooms",
                       "oysters",
                       "peanuts",
                       "pizzacut", "pizzaone",
                       "popcorn_chicken",
                       "porkchops",
                       "roastedchicken",
                       "scallop",
                       "shrimp",
                       "strand",
                       "sweetpotato",
                       "wan",
                       "danye",
                       "rice",
                       "fenzhengrou",
                       "jikuai",
                       "milk",
                       "shuiwu",
                       "zhedang-heian",
                       "nofood"]

    classes_id38 = {"babycabbage": 0,
                    "beefsteak": 1,
                    "cabbage_heart": 2,
                    "chestnut": 3,
                    "chickenwing_root": 4,
                    "chickenwing_middle": 5,
                    "chickenwing_tip": 6,
                    "chips": 7,
                    "cookies": 8,
                    "corn": 9,
                    "crab": 10,
                    "cranberrycookies": 11,
                    "duck": 12,
                    "eggplant": 13,
                    "eggtart": 14,
                    "fish": 15,
                    "lambchops": 16,
                    "mushrooms": 17,
                    "oysters": 18,
                    "peanuts": 19,
                    "pizzacut": 20,
                    "pizzaone": 21,
                    "popcorn_chicken": 22,
                    "porkchops": 23,
                    "roastedchicken": 24,
                    "scallop": 25,
                    "shrimp": 26,
                    "strand": 27,
                    "sweetpotato": 28,
                    "wan": 29,
                    "danye": 30,
                    "rice": 31,
                    "fenzhengrou": 32,
                    "jikuai": 33,
                    "milk": 34,
                    "shuiwu": 35,
                    "zhedang-heian": 36,
                    "nofood": 37}

    # 需要修改
    classes_id = classes_id38  #######
    classes = classes_label38  #######

    img_dir = "F:/serve_data/ZG_data/20210129/2021"  # 文件夹地址

    img_nofood = "F:/serve_data/ZG_data/20210129/2021_nofood"  # nofood文件夹地址
    if not os.path.exists(img_nofood): os.mkdir(img_nofood)

    img_noresults = "F:/serve_data/ZG_data/20210129/2021_noresults"  # noresults文件夹地址
    if not os.path.exists(img_noresults): os.mkdir(img_noresults)

    img_gaofen_correct = "F:/serve_data/ZG_data/20210129/2021_gaofen_correct"  # 高分正确文件夹地址
    if not os.path.exists(img_gaofen_correct): os.mkdir(img_gaofen_correct)

    img_gaofen_error = "F:/serve_data/ZG_data/20210129/2021_gaofen_error"  # 高分错误文件夹地址
    if not os.path.exists(img_gaofen_error): os.mkdir(img_gaofen_error)

    img_difen_correct = "F:/serve_data/ZG_data/20210129/2021_difen_correct"  # 低分正确文件夹地址
    if not os.path.exists(img_difen_correct): os.mkdir(img_difen_correct)

    img_difen_error = "F:/serve_data/ZG_data/20210129/2021_difen_error"  # 低分错误文件夹地址
    if not os.path.exists(img_difen_error): os.mkdir(img_difen_error)

    start_time = time.time()
    Y = YoloTest()  # 加载模型

    for cc in tqdm(os.listdir(img_dir)):
        if cc in classes:
            img___list = os.listdir(img_dir + "/" + cc)
            for img in img___list:
                if img.endswith(".jpg"):
                    img_path = img_dir + "/" + cc + "/" + img
                    bb = Y.result(img_path)
                    if len(bb) == 0:  # 无任何结果，直接移动至img_noresults
                        shutil.move(img_path, img_noresults + "/" + img)
                    else:
                        pre = int(bb[0][-1])
                        score = bb[0][-2]
                        if pre == 37:  # 空直接移动至nofood文件夹地址
                            shutil.move(img_path, img_nofood + "/" + img)
                        else:
                            if score >= 0.8:  # 高分
                                if pre == classes_id[cc]:  # 高分且正确
                                    img_c_dir = img_gaofen_correct + "/" + cc
                                    if not os.path.exists(img_c_dir): os.mkdir(img_c_dir)
                                    shutil.move(img_path, img_c_dir + "/" + img)
                                else:
                                    he = he_foods(pre, cc)
                                    if he:  # 高分且正确
                                        img_c_dir = img_gaofen_correct + "/" + cc
                                        if not os.path.exists(img_c_dir): os.mkdir(img_c_dir)
                                        shutil.move(img_path, img_c_dir + "/" + img)
                                    else:  # 高分错误
                                        img_e_dir = img_gaofen_error + "/" + cc
                                        if not os.path.exists(img_e_dir): os.mkdir(img_e_dir)
                                        shutil.move(img_path, img_e_dir + "/" + img)
                            elif score < 0.8 and score >= 0.6:
                                if pre == classes_id[cc]:  # 低分且正确
                                    img_dc_dir = img_difen_correct + "/" + cc
                                    if not os.path.exists(img_dc_dir): os.mkdir(img_dc_dir)
                                    shutil.move(img_path, img_dc_dir + "/" + img)
                                else:
                                    he = he_foods(pre, cc)
                                    if he:  # 低分且正确
                                        img_dc_dir = img_difen_correct + "/" + cc
                                        if not os.path.exists(img_dc_dir): os.mkdir(img_dc_dir)
                                        shutil.move(img_path, img_dc_dir + "/" + img)
                                    else:  # 低分错误
                                        img_de_dir = img_difen_error + "/" + cc
                                        if not os.path.exists(img_de_dir): os.mkdir(img_de_dir)
                                        shutil.move(img_path, img_de_dir + "/" + img)
                            else:
                                shutil.move(img_path, img_noresults + "/" + img)



        else:
            print("img dir name not in classes list:", cc)
