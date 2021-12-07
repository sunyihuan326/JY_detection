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

print('import over')
INPUT_SIZE =416
class YoloPredic(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = INPUT_SIZE  # 输入图片尺寸（默认正方形）
        self.num_classes = 8  # 种类数
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.RKNN_MODEL_PATH = "./yolov3_JYRobot_94_quantization_i16.rknn"  # rknn文件地址
        self.rknn = RKNNLite()
        # Direct load rknn model
        print('Loading RKNN model')
        time0 = time.time()
        ret = self.rknn.load_rknn(self.RKNN_MODEL_PATH)
        time1 = time.time()
        print("load rknn time:",time1 - time0)
        ret = self.rknn.init_runtime(target='rv1126', device_id='c3d9b8674f4b94f6')
        time2 = time.time()
        print("init run env time:",time2 - time1)
        if ret != 0:
            print('load rknn model failed.')
            exit(ret)

    def predict(self, image):
        #img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        print(image_data.shape)

        outputs = self.rknn.inference(inputs=[image_data])
        #print(outputs)
        pred_sbbox = outputs[0]
        pred_mbbox = outputs[1]
        pred_lbbox = outputs[2]
        print(pred_sbbox.sum())
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr= self.predict(image)
        print("预测结果：", bboxes_pr)
        # memory_detail = self.rknn.eval_memory()

def scan_files(directory,prefix=None,postfix=None):
    files_list=[]
    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root,special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root,special_file))
            else:
                files_list.append(os.path.join(root,special_file))

    return files_list


if __name__ == '__main__':
    #img_path = "./1.jpg"  # 图片地址
    print('program started')
    start_time = time.time()
    file_list = scan_files("./test_data")
    scan_over = time.time()
    print('scan time:',scan_over - start_time)
    Y = YoloPredic()

    #Y.result('./1.jpg')

    for i, val in enumerate(file_list):
          print(val)
          time0 = time.time()
          Y.result(val)
          time1 = time.time()
          print("单张预测时间：", time1 - time0)
