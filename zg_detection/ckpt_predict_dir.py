# -*- encoding: utf-8 -*-

"""
预测一个文件夹图片结果
@File    : ckpt_predict.py
@Time    : 2019/12/16 15:45
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import detection.core.utils as utils
import os
import time
import shutil
from tqdm import tqdm
from detection.core.config import cfg

# gpu限制
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


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
    classes_id = {"babycabbage": 0,
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

    img_root = "F:/serve_data/ZG_data/20210129/2021_gaofen_error"  # 图片文件地址

    Y = YoloTest()
    end_time0 = time.time()
    cls = os.listdir(img_root)
    classes = {value: key for key, value in classes_id.items()}
    for cc in cls:
        for img in tqdm(os.listdir(img_root + "/" + cc)):
            if img.endswith("jpg"):
                img_path = img_root + "/" + cc + "/" + img
                end_time1 = time.time()
                bboxes_p = Y.result(img_path)
                # 食材分到对应文件夹
                if len(bboxes_p) == 0:
                    if not os.path.exists(img_root + "/" + cc + "/noresult"): os.mkdir(img_root + "/noresult")
                    shutil.move(img_path, img_root + "/" + cc + "/noresult" + "/" + img)
                else:
                    pre = int(bboxes_p[0][-1])
                    if not os.path.exists(img_root + "/" + cc + "/" + classes[pre]): os.mkdir(
                        img_root + "/" + cc + "/" + classes[pre])
                    shutil.move(img_path, img_root + "/" + cc + "/" + classes[pre] + "/" + img)
