# -*- coding: utf-8 -*-
# @Time    : 2021/9/26
# @Author  : sunyihuan
# @File    : predict_and_seg.py
'''
预测后，再分割
'''
import cv2
import numpy as np
import tensorflow as tf
import xdsj_detection.core.utils as utils
import threading
import copy
import time


def watershred(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    return img


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 8  # 种类数
        # 类别对应id
        # 0：dishcloth，1：dustbin，2：line，3：shoes，4：socks，5：None，6：carpet，7：cup
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/JY_detection/xdsj_detection/checkpoint/yolov3_test_loss=2.7978.ckpt-60"  # ckpt文件地址
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
        # bboxes格式：[[xmin,ymin,xmax,ymax,score,cls_id]]

        return bboxes


if __name__ == "__main__":
    img_path = "F:/robots_images_202107/saved_pictures_1280/19_-10_45--TueAug240907112021--0.16_-10.06_43.84.jpg"
    end_time1 = time.time()
    Y = YoloPredict()
    image = cv2.imread(img_path)  # 图片读取
    bboxes_p = Y.predict(image)
    if len(bboxes_p) > 0:
        for i in range(len(bboxes_p)):
            image_crop = image[int(bboxes_p[i][1]):int(bboxes_p[i][3]), int(bboxes_p[i][0]):int(bboxes_p[i][2])]
            img = watershred(image_crop)

            cv2.namedWindow('first', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('first', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()