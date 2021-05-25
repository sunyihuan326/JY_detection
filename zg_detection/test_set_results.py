# -*- coding: utf-8 -*-
# @Time    : 2021/5/14
# @Author  : sunyihuan
# @File    : test_set_results.py
'''
test集查看预测结果
数据集格式：root
              beefsteak
                 xxx.jpg
              ……
输出excel表格统计准确率和混淆矩阵

'''
import cv2
import numpy as np
import tensorflow as tf
from zg_detection.core import utils
import os
import shutil
from tqdm import tqdm
import xlwt
import time
from sklearn.metrics import confusion_matrix
from zg_detection.food_correct_utils import correct_bboxes

# gpu限制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(gpu_options=gpu_options)


def he_foods(pre):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    if pre in [29, 30, 31, 32, 33, 34] and classes_id[classes[i]] in [29, 30, 31, 32, 33, 34]:  #
        rigth_label = True
    elif pre in [12, 24] and classes_id[classes[i]] in [12, 24]:  #
        rigth_label = True
    # elif pre in [32, 33] and classes_id[classes[i]] in [32, 33]:  #
    #    rigth_label = True
    else:
        rigth_label = False
    return rigth_label


class YoloTest(object):
    def __init__(self):
        self.input_size = 416
        self.num_classes = 38
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        # self.weight_file = "E:/ckpt_dirs/zg_project/20210517/yolov3.pb"
        self.pb_file = "E:/ckpt_dirs/zg_project/20210517/yolov3.pb"
        self.write_image = True
        self.show_label = True

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

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

    def result(self, image_path, save_dir, s_thre):
        '''

        :param image_path: 图片地址
        :param save_dir: 保存地址
        :param s_thre: 输出置信度限制
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr = self.predict(image)  # 预测结果

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + ".jpg"  # 图片保存地址
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片

        # 预测结果,bboxes_pr输出格式为[x_min, y_min, x_max, y_max, probability, cls_id]
        if len(bboxes_pr) > 0:
            bboxes_pr = correct_bboxes(bboxes_pr)
            if bboxes_pr[0][-2] > s_thre:
                return bboxes_pr
            else:
                return []
        else:
            return []


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
    mode = "multi_0517"  #######
    tag = "_75_score80_gai"
    img_dir = "F:/Test_set/ZG/testset"  # 文件夹地址
    save_root = "F:/Test_set/ZG/testset_results"
    save_dir = "{0}/detect_{1}{2}".format(save_root, mode, tag)  # 图片保存地址
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    fooderror_dir = "{0}/food_error_{1}{2}".format(save_root, mode, tag)  # 食材预测结果错误保存地址
    if not os.path.exists(fooderror_dir): os.mkdir(fooderror_dir)

    no_result_dir = "{0}/no_result_{1}{2}".format(save_root, mode, tag)  # 无任何输出结果保存地址
    if not os.path.exists(no_result_dir): os.mkdir(no_result_dir)

    start_time = time.time()
    Y = YoloTest()  # 加载模型
    end0_time = time.time()
    print("model loading time:", end0_time - start_time)
    new_classes = {v: k for k, v in classes_id.items()}

    jpgs_count_all = 0
    layer_jpgs_acc = 0
    food_jpgs_acc = 0

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("multi_food")
    sheet1.write(0, 0, "classes")
    sheet1.write(1, 0, "classes")

    sheet1.write(1, 3, "food_acc")
    sheet1.write(1, 1, "jpgs_all")
    sheet1.write(1, 2, "food_right_nums")
    sheet1.write(1, 4, "food_acc")
    sheet1.write(1, 5, "no_result_nums")

    layer_img_true = []
    layer_img_pre = []

    food_img_true = []
    food_img_pre = []
    # for i in range(len(classes)):
    for i in range(38):
        if i == 12 or i == 29 or i == 30 or i == 31 or i == 32 or i == 33 or i == 34 or i == 37:
            continue
        else:
            c = classes[i].lower()
            error_noresults = 0  # 食材无任何输出结果统计
            food_acc = 0  # 食材正确数量统计
            all_jpgs = 0  # 图片总数统计

            img_dirs = img_dir + "/" + c
            if os.path.exists(img_dirs):
                fooderror_dirs = fooderror_dir + "/" + c
                if os.path.exists(fooderror_dirs): shutil.rmtree(fooderror_dirs)
                os.mkdir(fooderror_dirs)

                noresult_dir = no_result_dir + "/" + c
                if os.path.exists(noresult_dir): shutil.rmtree(noresult_dir)
                os.mkdir(noresult_dir)

                save_c_dir = save_dir + "/" + c
                if os.path.exists(save_c_dir): shutil.rmtree(save_c_dir)
                os.mkdir(save_c_dir)

                c_food_right_list = []

                # 结果查看
                for file in tqdm(os.listdir(img_dirs)):
                    if file.endswith("jpg"):
                        all_jpgs += 1  # 统计总jpg图片数量
                        image_path = img_dirs + "/" + file
                        bboxes_pr = Y.result(image_path, save_c_dir, 0.8)  # 预测每一张结果并保存

                        if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                            error_noresults += 1
                            shutil.copy(image_path, noresult_dir + "/" + file)
                        else:
                            pre = int(bboxes_pr[0][-1])
                            food_img_pre.append(pre)
                            food_img_true.append(classes_id[classes[i]])

                            if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                                food_acc += 1
                                c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                            else:
                                right_label = he_foods(pre)
                                if right_label:  # 合并后结果正确
                                    food_acc += 1
                                    c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                                else:

                                    drawed_img_save_to_path = str(image_path).split("/")[-1]
                                    drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[
                                                                  0] + "_" + ".jpg"  # 图片保存地址
                                    shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                                fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(pre) + ".jpg")

            sheet1.write(i + 2, 3, food_acc)  # 食材准确率写入
            sheet1.write(i + 2, 0, c)
            sheet1.write(i + 2, 1, all_jpgs)  # 写入检测总数
            sheet1.write(i + 2, 2, food_acc)  # 写入食材正确数
            if all_jpgs == 0:
                sheet1.write(i + 2, 4, 0)
            else:
                sheet1.write(i + 2, 4, round((food_acc / all_jpgs) * 100, 2))
            sheet1.write(i + 2, 5, error_noresults)

            print("food name:", c)
            jpgs_count_all += all_jpgs
            food_jpgs_acc += food_acc
    print("all food accuracy:", round((food_jpgs_acc / jpgs_count_all) * 100, 2))  # 输出食材正确数

    food_conf = confusion_matrix(y_pred=food_img_pre, y_true=food_img_true)

    sheet2 = workbook.add_sheet("food_confusion_matrix")
    for i in range(len(classes)):
        sheet2.write(i + 1, 0, classes[i])
        sheet2.write(0, i + 1, classes[i])
    for i in range(food_conf.shape[0]):
        for j in range(food_conf.shape[1]):
            sheet2.write(i + 1, j + 1, str(food_conf[i, j]))

    print(food_conf)
    print(sum(sum(food_conf)))

    sheet1.write(55, 1, jpgs_count_all)
    sheet1.write(55, 2, food_jpgs_acc)
    sheet1.write(55, 4, round((food_jpgs_acc / jpgs_count_all) * 100, 2))

    workbook.save("{0}/all_result_{1}{2}.xls".format(save_root, mode, tag))

    end_time = time.time()
    print("all jpgs time:", end_time - end0_time)
