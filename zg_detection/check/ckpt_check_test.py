# -*- encoding: utf-8 -*-

"""
@File    : ckpt_check_test.py
@Time    : 2019/12/19 18:16
@Author  : sunyihuan
"""

'''
ckpt文件预测某一文件夹下各类所有图片食材结果
并输出各准确率至excel表格中
'''

import cv2
import numpy as np
import tensorflow as tf
import detection.core.utils as utils
import os
import shutil
from tqdm import tqdm
import xlwt
from sklearn.metrics import confusion_matrix


def correct_bboxes(bboxes_pr):
    '''
    bboxes_pr结果矫正
    :param bboxes_pr: 模型预测结果，格式为[x_min, y_min, x_max, y_max, probability, cls_id]
    :param layer_n:
    :return:
    '''
    num_label = len(bboxes_pr)
    # 未检测食材
    if num_label < 2:  # 标签数少于2个直接输出
        return bboxes_pr
    # 检测到多个食材
    else:
        new_bboxes_pr = []
        for i in range(len(bboxes_pr)):
            if bboxes_pr[i][4] >= 0.45:
                new_bboxes_pr.append(bboxes_pr[i])

        new_num_label = len(new_bboxes_pr)
        if new_num_label == 0:
            return new_bboxes_pr
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


class YoloTest(object):
    def __init__(self):
        self.input_size = 320  # 输入图片尺寸（默认正方形）
        self.num_classes = 20  # 种类数
        self.score_threshold = 0.1
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/zg_project/detection0/20200601/yolov3_train_loss=62.7481.ckpt-350"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            # 模型加载
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            # 输入
            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            # 输出检测结果
            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

    def predict(self, image):
        '''
        预测结果
        :param image: 图片数据，shape为[800,600,3]
        :return:
            bboxes：食材检测预测框结果，格式为：[x_min, y_min, x_max, y_max, probability, cls_id],
            layer_n[0]：烤层检测结果，0：最下层、1：中间层、2：最上层、3：其他
        '''
        org_image = np.copy(image)

        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def result(self, image_path, save_dir):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr = self.predict(image)  # 预测结果

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + ".jpg"  # 图片保存地址，烤层结果在命名中
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return bboxes_pr


if __name__ == '__main__':
    img_dir = "E:/zg_data/check_data/JPGImages20_test"  # 文件夹地址
    save_dir = "E:/zg_data/check_data/JPGImages20_testdetection"  # 图片保存地址
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    fooderror_dir = "E:/zg_data/check_data/JPGImages20_test_fooderror"  # 食材预测结果错误保存地址
    if not os.path.exists(fooderror_dir): os.mkdir(fooderror_dir)

    no_result_dir = "E:/zg_data/check_data/JPGImages20_test_noresult"  # 无任何输出结果保存地址
    if not os.path.exists(no_result_dir): os.mkdir(no_result_dir)

    Y = YoloTest()  # 加载模型

    classes = ["babycabbage", "cabbage_heart", "chestnut", "chickenwing_middle", "chickenwing_root",
               "chickenwing_tip", "corncut", "cornone", "eggplant",
               "eggtart", "frenchfries", "mushrooms", "nofood",
               "peanuts", "popcorn_chicken", "redshrimp", "scallop", "shrimp",
               "sweetpotato", "sweetpotatocut"]

    classes_id = {"babycabbage": 0, "cabbage_heart": 1, "chestnut": 2, "chickenwing_middle": 3, "chickenwing_root": 4,
                  "chickenwing_tip": 5, "corncut": 6, "cornone": 7, "eggplant": 8,
                  "eggtart": 9, "frenchfries": 10, "mushrooms": 11, "nofood": 12,
                  "peanuts": 13, "popcorn_chicken": 14, "redshrimp": 15, "scallop": 16, "shrimp": 17,
                  "sweetpotato": 18, "sweetpotatocut": 19}
    jpgs_count_all = 0
    food_jpgs_acc = 0

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("zg_food")
    sheet1.write(0, 0, "classes")
    sheet1.write(1, 0, "classes")

    sheet1.write(1, 1, "jpgs_all")
    sheet1.write(1, 2, "food_acc")
    sheet1.write(1, 3, "no_result_nums")
    sheet1.write(1, 4, "食材准确率")

    food_img_true = []
    food_img_pre = []
    for i in range(len(classes)):
        c = classes[i].lower()

        error_noresults = 0  # 食材无任何输出结果统计
        food_acc = 0  # 食材正确数量统计
        all_jpgs = 0  # 图片总数统计

        img_dirs = img_dir + "/" + c

        fooderror_dirs = fooderror_dir + "/" + c
        if os.path.exists(fooderror_dirs): shutil.rmtree(fooderror_dirs)
        os.mkdir(fooderror_dirs)

        noresult_dir = no_result_dir + "/" + c
        if os.path.exists(noresult_dir): shutil.rmtree(noresult_dir)
        os.mkdir(noresult_dir)

        save_c_dir = save_dir + "/" + c
        if os.path.exists(save_c_dir): shutil.rmtree(save_c_dir)
        os.mkdir(save_c_dir)

        for file in tqdm(os.listdir(img_dirs)):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/" + file
                bboxes_pr = Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                bboxes_pr = correct_bboxes(bboxes_pr)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    pre = bboxes_pr[0][-1]
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                        food_acc += 1
                    else:
                        drawed_img_save_to_path = str(image_path).split("/")[-1]
                        drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + ".jpg"  # 图片保存地址，烤层结果在命名中
                        shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                    fooderror_dirs + "/" + file.split(".jpg")[0] + ".jpg")
        jpgs_count_all += all_jpgs
        food_jpgs_acc += food_acc
        sheet1.write(i + 2, 1, all_jpgs)  # 写入正确总数
        sheet1.write(i + 2, 2, food_acc)  # 写入食材正确数
        sheet1.write(i + 2, 3, error_noresults)  # 写入食材正确数
        sheet1.write(i + 2, 4, round((food_acc / all_jpgs) * 100, 2))  # 写入食材准确率

    print("all food accuracy:", round((food_jpgs_acc / jpgs_count_all) * 100, 2))  # 输出食材正确数

    food_conf = confusion_matrix(y_pred=food_img_pre, y_true=food_img_true)

    print(food_conf)
    print(sum(sum(food_conf)))

    sheet1.write(30, 1, jpgs_count_all)
    sheet1.write(30, 2, food_jpgs_acc)
    sheet1.write(30, 4, round((food_jpgs_acc / jpgs_count_all) * 100, 2))

    workbook.save("E:/zg_data/check_data/test.xls")
