# -*- encoding: utf-8 -*-

"""
预测一个文件夹图片结果
@File    : ckpt_predict_camera.py
@Time    : 2019/12/16 15:45
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import xdsj_detection.core.utils as utils
import os
import time
import shutil
from tqdm import tqdm

# gpu限制
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 19  # 种类数

        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/JY_detection/xdsj_detection/checkpoint/yolov3_test_loss=2.2842.ckpt-75" # ckpt文件地址
        # self.weight_file = "./checkpoint/yolov3_train_loss=4.7681.ckpt-80"
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            # 模型加载
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=config)
            self.saver.restore(self.sess, self.weight_file)

            # 输入
            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            # 输出检测结果
            # self.conv_sbbox = graph.get_tensor_by_name("define_loss/conv_sbbox/BiasAdd:0")
            # self.conv_mbbox = graph.get_tensor_by_name("define_loss/conv_mbbox/BiasAdd:0")
            # self.conv_lbbox = graph.get_tensor_by_name("define_loss/conv_lbbox/BiasAdd:0")

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
        # image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr = self.predict(image)  # 预测结果

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + ".jpg"
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return bboxes_pr


if __name__ == '__main__':
    start_time = time.time()

    # img_dir = "F:/robot_test_from_YangYalin/saved_pictures_1280"  # 图片文件地址
    #
    # save_dir = "F:/robot_test_from_YangYalin/saved_pictures_1280_detetction1012"

    img_dir = "F:/RobotProgram/data/other_camera/100_A03"  # 图片文件地址

    save_dir = "F:/RobotProgram/data/other_camera/100_A03_detect"

    if not os.path.exists(save_dir): os.mkdir(save_dir)
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)

    for img in tqdm(os.listdir(img_dir)):
        if img.endswith("jpg"):
            img_path = img_dir + "/" + img
            end_time1 = time.time()
            bboxes_p = Y.result(img_path, save_dir)
            print(bboxes_p)

    end_time1 = time.time()
    print("all data time:", end_time1 - end_time0)
