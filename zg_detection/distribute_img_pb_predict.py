# -*- coding: utf-8 -*-
# @Time    : 2021/6/5
# @Author  : sunyihuan
# @File    : distribute_img_pb_predict.py
'''
将图片拆分为小图片检测并输出结果

'''

import cv2
import numpy as np
import tensorflow as tf
import zg_detection.core.utils as utils
import os
import matplotlib.pyplot as plt


class YoloPredic(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 38  # 种类数
        self.score_threshold = 0.6
        self.iou_threshold = 0.5
        self.pb_file = "E:/ckpt_dirs/zg_project/20210517/yolov3.pb"  # pb文件地址
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
        org_h, org_w, _ = image.shape
        org_image = np.copy(image)
        image_data = utils.image_preporcess(org_image, [self.input_size, self.input_size])
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
        print(bboxes)
        return bboxes

    # def result(self, image_path):
    #
    #     bboxes_pr = self.predict(image)  # 预测结果
    #     print(bboxes_pr)
    #     if self.write_image:
    #         image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)  # 画图
    #         drawed_img_save_to_path = str(image_path).split("/")[-1]
    #         cv2.imwrite(save_root + "/" + drawed_img_save_to_path, image)


if __name__ == '__main__':
    img_path = "C:/Users/sunyihuan/Desktop/t/4_20210114__eggtart_tj_1000.jpg"  # 图片地址
    # save_root = "F:/Test_set/ZG/testset_results/no_result_multi_0517_75_score80_di_wan_all_original_adjust_detect"
    # if not os.path.exists(save_root): os.mkdir(save_root)
    Y = YoloPredic()
    image = cv2.imread(img_path)  # 图片读取
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    org_h, org_w, _ = image.shape

    image0 = image[0:int(org_h / 2), 0:int(org_w / 2)]
    image1 = image[int(org_h / 2):org_h, 0:int(org_w / 2)]
    image2 = image[0:int(org_h / 2), int(org_w / 2):org_w]
    image3 = image[int(org_h / 2):org_h, int(org_w / 2):org_w]

    bboxes_pr0 = Y.predict(image0)
    bboxes_pr1 = Y.predict(image1)
    bboxes_pr2 = Y.predict(image2)
    bboxes_pr3 = Y.predict(image3)

    # boxes合并
    bboxes_pr = bboxes_pr0
    for b in bboxes_pr1:
        print("b::::", b)
        b[1] = b[1] + int(org_h / 2)
        b[3] = b[3] + int(org_h / 2)
        bboxes_pr.append(b)
    for b in bboxes_pr2:
        b[0] = b[0] + int(org_w / 2)
        b[2] = b[2] + int(org_w / 2)
        bboxes_pr.append(b)
    for b in bboxes_pr3:
        b[0] = b[0] + int(org_w / 2)
        b[2] = b[2] + int(org_w / 2)
        b[1] = b[1] + int(org_h / 2)
        b[3] = b[3] + int(org_h / 2)
        bboxes_pr.append(b)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)
    image3 = utils.draw_bbox(image, bboxes_pr,
                             show_label=True)  # 画图

    cv2.imwrite(img_path.split(".jpg")[0] + "_0.jpg", image3)
