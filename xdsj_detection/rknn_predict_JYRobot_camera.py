# -*- coding: utf-8 -*-
# @Time    : 2021/11/18
# @Author  : sunyihuan
# @File    : rknn_predict_JYRobot.py
print('start import')
# from rknn.api import RKNN
import cv2
import os
import numpy as np
import utils
from rknnlite.api import RKNNLite
import time
import threading
import copy

print('import over')


class UsbCamCapture:
    def __init__(self, url):
        self.Frame = []
        self.status = False
        self.is_stop = False
        self.capture = cv2.VideoCapture(url)
        if self.capture.isOpened():
            self.capture.set(3, 640)
            self.capture.set(4, 360)
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


INPUT_SIZE = 416

class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = INPUT_SIZE  # 输入图片尺寸（默认正方形）
        self.num_classes = 8  # 种类数
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.RKNN_MODEL_PATH = "./yolov3_JYRobot_94_quantization_i16.rknn"  # pb文件地址
        self.rknn = RKNNLite()
        # Direct load rknn model
        print('Loading RKNN model')
        time0 = time.time()
        ret = self.rknn.load_rknn(self.RKNN_MODEL_PATH)
        time1 = time.time()
        print("load rknn time:", time1 - time0)
        ret = self.rknn.init_runtime(target='rv1126', device_id='c3d9b8674f4b94f6')
        time2 = time.time()
        print("init run env time:", time2 - time1)
        if ret != 0:
            print('load rknn model failed.')
            exit(ret)

    def predict(self, image):
        img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        print(type(image_data))

        outputs = self.rknn.inference(inputs=[img])
        pred_sbbox = outputs[0]
        pred_mbbox = outputs[1]
        pred_lbbox = outputs[2]

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def result(self, image_path):
        #image = cv2.imread(image_path)  # 图片读取
        bboxes_pr = self.predict(image_path)
        print("预测结果：", bboxes_pr)
        # memory_detail = self.rknn.eval_memory()

        return bboxes_pr


if __name__ == '__main__':
    import time

    Y = YoloPredict()  # 加载模型

    usb_cam = UsbCamCapture(20)
    # 启动子线程
    usb_cam.start()
    # 暂停1秒，确保影像已经填充
    time.sleep(1)

    while True:
        np_img = usb_cam.get_frame()
        org_img = copy.deepcopy(np_img)

        start_time = time.time()
        end_time0 = time.time()

        bboxes_pr = Y.result(np_img)  # 预测结果
        print(bboxes_pr)

        end_time1 = time.time()
        print("predict time:", end_time1 - end_time0)
        image = utils.draw_bbox(np_img, bboxes_pr, show_label=True)

        # t = time.localtime()
        # str_time = str(time.asctime(t)).replace(' ', '').replace(':', '')
        # dir = "F:/robots_images_202107/save_p/detect_dir/" + str_time + ".jpg"
        # dir_deal = "F:/robots_images_202107/save_p/orignal_dir/" + str_time + "_deal.jpg"
        # cv2.imwrite(dir, image)
        # cv2.imwrite(dir_deal, org_img)
        #k = cv2.waitKey(0)
        #if k==ord("s"):
            #cv2.imwrite("./test_data/"+str(start_time)+".jpg", np_img)

        cv2.namedWindow('camera_output', 0)
        cv2.imshow('camera_output',image)
        cv2.waitKey(30)
