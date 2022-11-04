# -*- coding: utf-8 -*-
# @Time    : 2021/7/2
# @Author  : sunyihuan
# @File    : pb_and_savedmodel_predict.py

import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np
import tensorflow.compat.v1 as tf
import xdsj_detection.core.utils as utils
from xdsj_detection.core.config import cfg
from xdsj_detection.distance_and_angle import *


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
        self.pb_file = "E:/JY_detection/xdsj_detection/model/yolo_60.pb"
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL
        self.typ = "yolov3"

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            # 输入
            self.input = graph.get_tensor_by_name("define_input/input_data:0")

            # 输出检测结果
            # self.conv_sbbox = graph.get_tensor_by_name("define_loss/conv_sbbox/BiasAdd:0")
            # self.conv_mbbox = graph.get_tensor_by_name("define_loss/conv_mbbox/BiasAdd:0")
            # self.conv_lbbox = graph.get_tensor_by_name("define_loss/conv_lbbox/BiasAdd:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
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

        # image = image_undistort(image)  # 图片畸变矫正

        bboxes_pr = self.predict(image)  # 预测结果

        return bboxes_pr


if __name__ == '__main__':
    import time

    start_time = time.time()

    img_dir = "C:/Users/sunyihuan/Desktop/t"  # 图片文件地址

    save_dir = "C:/Users/sunyihuan/Desktop/t_"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    Y = YPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)

    for img in tqdm(os.listdir(img_dir)):
        img_path = img_dir + "/" + img
        # print(img)
        end_time1 = time.time()
        bboxes_p = Y.result(img_path)
        bboxes = []
        if len(bboxes_p) > 0:
            for b in bboxes_p:
                b = list(b)
                # b0 = bboxes_undistort(b[:2])
                # b1 = bboxes_undistort(b[2:4])
                # b[:2] = b0
                # b[2:4] = b1
                b.append(distance_to_camera(b[3]))
                # b0[6] = compute_angle(b[0])
                # b0[7] = compute_angle(b[2])
                b.append(compute_angle(b[0]))
                b.append(compute_angle(b[2]))
                bboxes.append(b)

        print(img, bboxes)
        image = cv2.imread(img_path)  # 图片读取
        # image = image_undistort(image)  # 图片畸变矫正

        image = utils.draw_bbox(image, bboxes, show_label=True)
        drawed_img_save_to_path = str(img_path).split("/")[-1]
        cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)
        # if img.endswith("jpg") :
        #     img_path = img_dir + "/" + img
        #     # print(img)
        #     end_time1 = time.time()
        #     bboxes_p = Y.result(img_path)
        #     bboxes = []
        #     if len(bboxes_p) > 0:
        #         for b in bboxes_p:
        #             b = list(b)
        #             # b0 = bboxes_undistort(b[:2])
        #             # b1 = bboxes_undistort(b[2:4])
        #             # b[:2] = b0
        #             # b[2:4] = b1
        #             b.append(distance_to_camera(b[3]))
        #             # b0[6] = compute_angle(b[0])
        #             # b0[7] = compute_angle(b[2])
        #             b.append(compute_angle(b[0]))
        #             b.append(compute_angle(b[2]))
        #             bboxes.append(b)
        #
        #     print(img,bboxes)
        #     image = cv2.imread(img_path)  # 图片读取
        #     # image = image_undistort(image)  # 图片畸变矫正
        #
        #     image = utils.draw_bbox(image, bboxes, show_label=True)
        #     drawed_img_save_to_path = str(img_path).split("/")[-1]
        #     cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)

    end_time1 = time.time()
    print("all data time:", end_time1 - end_time0)
    #
    #
    # start_time = time.time()
    # img_path = "F:/model_data/XDSJ/2020_data_bai/JPGImages/07302050.jpg"  # 图片地址
    # Y = YPredict()
    # end_time0 = time.time()
    #
    # print("model loading time:", end_time0 - start_time)
    # Y.result(img_path)
    # end_time1 = time.time()
    # print("predict time:", end_time1 - end_time0)
