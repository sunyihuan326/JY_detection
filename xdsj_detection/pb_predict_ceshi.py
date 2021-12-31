# -*- coding: utf-8 -*-
# @Time    : 2021/12/28
# @Author  : sunyihuan
# @File    : pb_predict_ceshi.py
'''
扫地机测试部采集数据，输出准确率、距离、角度等数据
'''
import os
from tqdm import tqdm
import xlwt
import tensorflow as tf
import xdsj_detection.core.utils as utils
from xdsj_detection.core.config import cfg
from xdsj_detection.distance_and_angle import *
import time
from sklearn.metrics import confusion_matrix


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
        self.pb_file = "E:/JY_detection/xdsj_detection/model/yolov3_1226.pb"
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
            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")

            # 输出
            if self.typ == "tiny":
                self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
                self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")
            else:
                self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
                self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
                self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        if self.typ == "tiny":
            pred_mbbox, pred_lbbox = self.sess.run(
                [self.pred_mbbox, self.pred_lbbox],
                feed_dict={
                    self.input: image_data,
                }
            )

            pred_bbox = np.concatenate([np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        else:
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

    def result(self, image):
        bboxes_pr = self.predict(image)  # 预测结果

        return bboxes_pr


def max_score_bbox(bb):
    '''
    输出置信度最大的框
    【前提：一张图片中仅含有一个类别】
    :param bb:
    :return:
    '''
    if len(bb) < 1:
        return bb
    else:
        best_b = []
        max_score = 0
        for b in bb:
            score = b[4]
            if score > max_score:
                max_score = score
                best_b = b
    return best_b


if __name__ == '__main__':
    class_id = {"dishcloth": 0, "dustbin": 1, "line": 2, "shoes": 3, "socks": 4,
                "None": 5, "carpet": 6, "cup": 7, "station": 8}
    img_root = "F:/Test_set/STSJ/saved_pictures_202112_undistorted"  # 图片文件地址
    save_root = "F:/Test_set/STSJ/saved_pictures_202112_undistorted_detect1230"  # 图片预测后保存地址
    if not os.path.exists(save_root): os.mkdir(save_root)

    start_time = time.time()
    Y = YPredict()

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("STSJ_all_test_data")
    sheet1.write(0, 0, "image_name")
    sheet1.write(0, 1, "true_id")
    sheet1.write(0, 2, "pre_id")
    sheet1.write(0, 3, "true_distance")
    sheet1.write(0, 4, "pre_distance")
    sheet1.write(0, 5, "true_left_angle")
    sheet1.write(0, 6, "pre_left_angle")
    sheet1.write(0, 7, "true_right_angle")
    sheet1.write(0, 8, "pre_right_angle")
    sheet1.write(0, 10, "class_name")
    sheet1.write(0, 11, "class_acc")

    all_nums = 0
    all_right_nums = 0  # 所有识别正确数量

    true_l = []
    pre_l = []

    for k, c in enumerate(os.listdir(img_root)):
        c_nums = 0  # 某一类别数量
        c_right_nums = 0  # 某一类别识别正确数量

        img_dir = img_root + "/" + c  # 类别文件夹
        save_dir = save_root + "/" + c  # 预测后类别文件夹
        if not os.path.exists(save_dir): os.mkdir(save_dir)

        for img in tqdm(os.listdir(img_dir)):
            img_path = img_dir + "/" + img  # 图片路径
            image = cv2.imread(img_path)  # 图片读取
            bboxes_p = Y.result(image)
            bboxes = []
            if len(bboxes_p) > 0:
                for b in bboxes_p:
                    b = list(b)
                    distance = 100 * distance_to_camera(b[3])  # 距离,单位：cm
                    if distance < 0:
                        distance = 200
                    b.append(distance)
                    left_angle = compute_angle(b[0])  # 左偏角
                    b.append(left_angle)
                    right_angle = compute_angle(b[2])  # 右偏角
                    b.append(right_angle)

                    bboxes.append(b)
            image = utils.draw_bbox(image, bboxes, show_label=True)
            drawed_img_save_to_path = str(img_path).split("/")[-1]
            # cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)

            c_nums += 1  # 该类别数+1
            all_nums += 1  # 总数+1

            bbox = max_score_bbox(bboxes)  # 输出置信度最大的框
            sheet1.write(all_nums + 1, 0, img)
            if len(bbox) > 0:
                true_id = class_id[c]  # 真实类别
                pre_id = int(bbox[5])  # 预测类别

                true_l.append(true_id)
                pre_l.append(pre_id)

                true_distance = float(img.split("_")[0])  # 真实距离
                pre_distance = bbox[6]  # 预测距离
                true_left_angle = float(img.split("_")[1])  # 真实左偏角
                pre_left_angle = bbox[7]  # 预测左偏角
                true_right_angle = img.split("_")[2].split("--")[0]  # 真实右偏角
                pre_right_angle = bbox[8]  # 预测右偏角

                if pre_id == true_id:
                    c_right_nums += 1
                    all_right_nums += 1
                else:
                    print(img, true_id, pre_id)
                    print(bboxes)
                    print(bbox)
                sheet1.write(all_nums + 1, 1, true_id)
                sheet1.write(all_nums + 1, 2, pre_id)
                sheet1.write(all_nums + 1, 3, true_distance)
                sheet1.write(all_nums + 1, 4, pre_distance)
                sheet1.write(all_nums + 1, 5, true_left_angle)
                sheet1.write(all_nums + 1, 6, pre_left_angle)
                sheet1.write(all_nums + 1, 7, true_right_angle)
                sheet1.write(all_nums + 1, 8, pre_right_angle)

        # 每一类识别率
        if c_nums > 0:
            c_acc = str(round(c_right_nums * 100 / c_nums, 2)) + "%"
        else:
            c_acc = "/"
        sheet1.write(k + 1, 10, c)
        sheet1.write(k + 1, 11, c_acc)
    print(confusion_matrix(true_l, pre_l))
    sheet1.write(all_nums + 5, 0, "总体准确率")
    sheet1.write(all_nums + 5, 1, str(round(all_right_nums * 100 / all_nums, 2)) + "%")
    # workbook.save("F:/Test_set/STSJ/saved_pictures_202112_undistorted_all_data1230.xls")
