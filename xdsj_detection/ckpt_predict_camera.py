# -*- encoding: utf-8 -*-

"""
预测一张图片结果
@File    : ckpt_predict_camera.py
@Time    : 2020/12/08 15:45
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import xdsj_detection.core.utils as utils
import threading
import copy


class UsbCamCapture:
    def __init__(self, url):
        self.Frame = []
        self.status = False
        self.is_stop = False
        self.capture = cv2.VideoCapture(url)
        if self.capture.isOpened():
            self.capture.set(3, 1280)
            self.capture.set(4, 720)
            self.capture.read()

    def start(self):
        # 把程序放进子线程，daemon=True 表示该线程会随着主线程关闭而关闭。
        print('usb_cam started!')
        threading.Thread(target=self.query_frame, daemon=True, args=()).start()

    def stop(self):
        # 记得要设计停止无限循环的开关。
        self.is_stop = True
        print('usb_cam stopped!')

    def get_frame(self):
        # 当有需要影像时，再回传最新的影像。
        return self.Frame

    def query_frame(self):
        while not self.is_stop:
            self.status, self.Frame = self.capture.read()
        self.capture.release()


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


if __name__ == '__main__':
    import time

    Y = YoloPredict()  # 加载模型

    usb_cam = UsbCamCapture(0)
    # 启动子线程
    usb_cam.start()
    # 暂停1秒，确保影像已经填充
    time.sleep(1)

    while True:
        np_img = usb_cam.get_frame()
        org_img = copy.deepcopy(np_img)

        start_time = time.time()
        end_time0 = time.time()

        bboxes_pr = Y.predict(np_img)  # 预测结果
        print(bboxes_pr)

        end_time1 = time.time()
        print("predict time:", end_time1 - end_time0)
        image = utils.draw_bbox(np_img, bboxes_pr, show_label=True)

        t = time.localtime()
        str_time = str(time.asctime(t)).replace(' ', '').replace(':', '')
        dir = "F:/robots_images_202107/save_p/detect_dir/" + str_time + ".jpg"
        dir_deal = "F:/robots_images_202107/save_p/orignal_dir/" + str_time + "_deal.jpg"
        cv2.imwrite(dir, image)
        cv2.imwrite(dir_deal, org_img)

        cv2.namedWindow('camera_output', 0)
        cv2.imshow('camera_output', image)
        cv2.waitKey(30)
