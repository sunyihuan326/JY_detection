# -*- coding: utf-8 -*-
# @Time    : 2021/1/5
# @Author  : sunyihuan
# @File    : pb_time_check.py

import cv2
import numpy as np
import tensorflow as tf
import detection.core.utils as utils
import os
import time


class YoloPredic(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 38  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.pb_file = "E:/ckpt_dirs/Food_detection/zg_project/detection0/20210104/food38_0104.pb"  # pb文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            # 输入
            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            # 输出
            self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        p_t = time.time()
        print("process time:", p_t - start_t)
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )
        pre_t = time.time()
        print("predit time:", pre_t - start_t)
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, 0.1)
        # print(bboxes)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        p_p_t = time.time()
        print("predict process time:", p_p_t - start_t)

        return bboxes

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        a_t = time.time()
        print("loading time:", a_t - start_t)
        bboxes_pr = self.predict(image)  # 预测结果
        return bboxes_pr
        # print(bboxes_pr)
        # if self.write_image:
        #     image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)  # 画图
        #     drawed_img_save_to_path = str(image_path).split("/")[-1]
        #     cv2.imwrite(drawed_img_save_to_path, image)


if __name__ == '__main__':
    img_root = "C:/Users/sunyihuan/Desktop/t"  # 图片地址
    Y = YoloPredic()
    for img in os.listdir(img_root):
        start_t = time.time()
        print(img)
        bb = Y.result(img_root + "/" + img)
        end_t = time.time()
        print("one predict time:", end_t - start_t)
