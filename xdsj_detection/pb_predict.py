# -*- coding: utf-8 -*-
# @Time    : 2021/7/2
# @Author  : sunyihuan
# @File    : pb_predict.py

import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import xdsj_detection.core.utils as utils
from xdsj_detection.core.config import cfg


class YPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path = cfg.TEST.ANNOT_PATH
        self.pb_file = "E:/JY_detection/xdsj_detection/model/yolov3_tiny.pb"
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL
        self.typ = "tiny"

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")

            # 输出
            if self.typ=="tiny":
                self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
                self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")
            else:
                self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
                self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
                self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        print(org_h, org_w)

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        if self.typ=="tiny":
            pred_mbbox, pred_lbbox = self.sess.run(
                [self.pred_mbbox, self.pred_lbbox],
                feed_dict={
                    self.input: image_data,
                }
            )

            pred_bbox = np.concatenate([np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        else:
            pred_sbbox,pred_mbbox, pred_lbbox = self.sess.run(
                [self.pred_sbbox,self.pred_mbbox, self.pred_lbbox],
                feed_dict={
                    self.input: image_data,
                }
            )

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr = self.predict(image)  # 预测结果
        print(bboxes_pr)
        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            cv2.imwrite(drawed_img_save_to_path, image)


if __name__ == '__main__':
    import time

    start_time = time.time()
    img_path = "F:/model_data/XDSJ/2020_data_bai/JPGImages/07302050.jpg"  # 图片地址
    Y = YPredict()
    end_time0 = time.time()

    print("model loading time:", end_time0 - start_time)
    Y.result(img_path)
    end_time1 = time.time()
    print("predict time:", end_time1 - end_time0)
