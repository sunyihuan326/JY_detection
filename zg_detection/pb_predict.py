# -*- encoding: utf-8 -*-

"""
pb文件预测输出结果

@File    : pb_and_savedmodel_predict.py
@Time    : 2019/8/20 10:29
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import zg_detection.core.utils as utils
import os


class YoloPredic(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 38  # 种类数
        self.score_threshold = 0.6
        self.iou_threshold = 0.5
        self.pb_file = "./yolov3.pb"  # pb文件地址
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
            # self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            # 输出
            self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input: image_data,
                # self.trainable: False
            }
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        print(bboxes)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr = self.predict(image)  # 预测结果
        print(bboxes_pr)
        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)  # 画图
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            cv2.imwrite(save_root+"/"+drawed_img_save_to_path, image)


if __name__ == '__main__':

    img_path = "F:/Test_set/ZG/testset_results/no_result_multi_0517_75_score80_di_wan_all_original_adjust"  # 图片地址
    save_root = "F:/Test_set/ZG/testset_results/no_result_multi_0517_75_score80_di_wan_all_original_adjust_detect"
    if not os.path.exists(save_root):os.mkdir(save_root)
    Y = YoloPredic()
    for jpg in os.listdir(img_path):
        Y.result(img_path+"/"+jpg)
