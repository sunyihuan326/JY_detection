# -*- encoding: utf-8 -*-

"""
预测一张图片结果
@File    : ckpt_predict.py
@Time    : 2020/12/08 15:45
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import xdsj_detection.core.utils as utils
from xdsj_detection.core.config import cfg


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        # self.num_classes = 13
        self.score_threshold = 0.5
        self.iou_threshold = 0.4
        self.weight_file = "E:\\JY_detection\\xdsj_detection\\checkpoint\\yolov3_test_loss=0.8410.ckpt-60"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        print(org_h, org_w)

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
    img_path = "F:/model_data/XDSJ/2020_data_bai/JPGImages/13502050.jpg"  # 图片地址
    Y = YoloPredict()
    end_time0 = time.time()

    print("model loading time:", end_time0 - start_time)
    Y.result(img_path)
    end_time1 = time.time()
    print("predict time:", end_time1 - end_time0)
