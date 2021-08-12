# -*- coding: utf-8 -*-
# @Time    : 2021/8/4
# @Author  : sunyihuan
# @File    : rknn_predict.py

from rknn.api import RKNN
import cv2
import numpy as np
from kx_detection.core import utils
from kx_detection.food_correct_utils import correct_bboxes, get_potatoml


class YoloPredic(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.top_n = 5
        self.RKNN_MODEL_PATH = "./yolov3_151_0.rknn"  # pb文件地址
        typ = "pb"  # or pb
        self.rknn = RKNN()

        # Direct load rknn model
        print('Loading RKNN model')
        ret = self.rknn.load_rknn(self.RKNN_MODEL_PATH)
        if ret != 0:
            print('load rknn model failed.')
            exit(ret)

    def get_top_cls(self, pred_bbox, org_h, org_w, top_n):
        '''
        获取top_n，类别和得分
        :param pred_bbox:所有框
        :param org_h:高
        :param org_w:宽
        :param top_n:top数
        :return:按置信度前top_n个，输出类别、置信度，
        例如
        [(18, 0.9916), (19, 0.0105), (15, 0.0038), (1, 0.0018), (5, 0.0016), (13, 0.0011)]
        '''
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_cls_threshold)
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = {}
        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            best_score = 0
            for i in range(len(cls_bboxes)):
                if cls_bboxes[i][-2] > best_score:
                    best_score = cls_bboxes[i][-2]
            if int(cls) not in best_bboxes.keys():
                best_bboxes[int(cls)] = round(best_score, 4)
        best_bboxes = sorted(best_bboxes.items(), key=lambda best_bboxes: best_bboxes[1], reverse=True)
        return best_bboxes[:top_n]

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        outputs = self.rknn.inference(image_data)
        pred_sbbox = outputs[0]
        pred_mbbox = outputs[1]
        pred_lbbox = outputs[2]
        layer_ = outputs[3]
        print(layer_)
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        best_bboxes = self.get_top_cls(pred_bbox, org_h, org_w, self.top_n)  # 获取top_n类别和置信度
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        layer_n = layer_[0]  # 烤层结果

        return bboxes, layer_n, best_bboxes

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr, layer_n, best_bboxes = self.predict(image)
        print("top_n类被及置信度：", best_bboxes)
        print("食材结果：", bboxes_pr)
        bboxes_pr, layer_n, best_bboxes = correct_bboxes(bboxes_pr, layer_n, best_bboxes)  # 矫正输出结果
        print(best_bboxes)
        print("top_n类被及置信度：", best_bboxes)
        print("食材结果：", bboxes_pr)
        bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出
        cls = best_bboxes[0][0]
        if int(cls) == 10:
            best_bboxes[0][1] = best_bboxes[0][1] * 1.2
        # best_bboxes为top_n类别和置信度，格式为：[(classes,score)]
        #                               例如： [(27, 0.9858), (14, 0.0046), (39, 0.004), (20, 0.004), (1, 0.0039)]
        # 预测结果,bboxes_pr输出格式为[x_min, y_min, x_max, y_max, probability, cls_id]
        # cls_id对应标签[0:beefsteak,1:cartooncookies,2:chickenwings,3:chiffoncake6,4:chiffoncake8
        #                 5:cookies,6:cranberrycookies,7:cupcake,8:eggtart,9:nofood,
        #                 10:peanuts,11:pizzacut,12:pizzaone,13:pizzatwo,14:porkchops,
        #                 15:potatocut,16:potatol,17:potatos,18:sweetpotatocut,19:sweetpotatol,
        #                 20:sweetpotatos,21:roastedchicken,22:toast,23:chestnut,24:cornone,
        #                 25:corntwo,26:drumsticks,27:taro,28:steamedbread,29:eggplant,
        #                 30:eggplant_cut_sauce,31:bread,32:container_nonhigh,33:container,34:fish,
        #                 35:hotdog,36:redshrimp,37:shrimp,38:strand,39:xizhi,
        #                 40:potatom,41:sweetpotatom,101:chiffon_size4]
        #                 预测结果,layer_输出结果为0：最下层、1：中间层、2：最上层、3：其他
        print("top_n类被及置信度：", best_bboxes)
        print("食材结果：", bboxes_pr)
        print("烤层结果：", layer_n)


if __name__ == '__main__':
    img_path = "./data/37_191021X3_kaojia_Potatom.jpg"  # 图片地址
    import time

    start_time = time.time()
    Y = YoloPredic()
    end_time0 = time.time()

    print("加载时间：", end_time0 - start_time)
    Y.result(img_path)
    end_time1 = time.time()
    print("单张预测时间：", end_time1 - end_time0)

if __name__ == "__main__":
    pass
