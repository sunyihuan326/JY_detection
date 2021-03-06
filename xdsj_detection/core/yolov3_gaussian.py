# -*- encoding: utf-8 -*-

"""
@File    : yolov3_gaussian.py
@Time    : 202003/1/15 10:53
@Author  : sunyihuan
"""

import numpy as np
import tensorflow as tf
import xdsj_detection.core.utils as utils
import xdsj_detection.core.common as common
import xdsj_detection.core.backbone as backbone
from xdsj_detection.core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, input_data, trainable):

        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD
        self.layer_nums = cfg.YOLO.LAYER_NUMS

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, self.out = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")
        print(self.out)
        self.predict_op = tf.argmax(input=self.out, axis=1, name='layer_classes')
        print("layer_nums::", self.predict_op)

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data):

        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)
        out = tf.layers.flatten(input_data)
        # out = tf.layers.dense(out, 400, activation=tf.nn.relu)
        out = tf.layers.dense(out, self.layer_nums)  # layer 输出

        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 9)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 9)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 9)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
        return conv_lbbox, conv_mbbox, conv_sbbox, out

    def nll_loss(self, x, mu, sigma, sigma_const=0.3):
        '''
        计算nll_loss
        :param x: y_true
        :param mu:y_pred_mu
        :param sigma:y_pred_sigma
        :param sigma_const:sigma常量
        :return:nll_loss值
        '''

        pi = tf.constant(np.pi)
        Z = (2 * pi * (sigma + sigma_const) ** 2) ** 0.5
        probability_density = tf.exp(-0.5 * (x - mu) ** 2 / ((sigma + sigma_const) ** 2)) / Z
        nll = -tf.log(probability_density + 1e-7)
        return nll

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 9 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_sigma = conv_output[:, :, :, :, 4:8]  # sigma数据读取
        conv_raw_conf = conv_output[:, :, :, :, 8:9]
        conv_raw_prob = conv_output[:, :, :, :, 9:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)  # 坐标数据

        pred_sigma = tf.sigmoid(conv_raw_sigma)  # sigma数据
        pred_conf = tf.sigmoid(conv_raw_conf)  # 置信度
        pred_prob = tf.sigmoid(conv_raw_prob)  # 类别

        return tf.concat([pred_xywh, pred_sigma, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = tf.maximum(boxes1_area + boxes2_area - inter_area, 1e-5)
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 9 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 8:9]
        conv_raw_prob = conv[:, :, :, :, 9:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_sigma = pred[:, :, :, :, 4:8]
        pred_conf = pred[:, :, :, :, 8:9]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        input_size = tf.cast(input_size, tf.float32)

        self.bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

        x_loss = self.nll_loss(label_xywh[..., 0:1], pred_xywh[..., 0:1], pred_sigma[..., 0:1])
        x_loss = respond_bbox * self.bbox_loss_scale * x_loss
        y_loss = self.nll_loss(label_xywh[..., 1:2], pred_xywh[..., 1:2], pred_sigma[..., 1:2])
        y_loss = respond_bbox * self.bbox_loss_scale * y_loss
        w_loss = self.nll_loss(label_xywh[..., 2:3], pred_xywh[..., 2:3], pred_sigma[..., 2:3])
        w_loss = respond_bbox * self.bbox_loss_scale * w_loss
        h_loss = self.nll_loss(label_xywh[..., 3:4], pred_xywh[..., 3:4], pred_sigma[..., 3:4])
        h_loss = respond_bbox * self.bbox_loss_scale * h_loss

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        xywh_loss = tf.reduce_mean(tf.reduce_sum(x_loss + y_loss + w_loss + h_loss, axis=[1, 2, 3, 4]))
        # giou_loss = tf.reduce_mean(tf.reduce_sum(bbox_loss_scale, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return xywh_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss

    def layer_loss(self, layer_label):
        '''
        烤层识别loss
        :param layer_label:
        :return:
        '''
        layer_label = tf.cast(layer_label, tf.int32)
        # Y = tf.one_hot(layer_label, depth=self.layer_nums, axis=1, dtype=tf.float32)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=layer_label, logits=self.out),
                              name='layer_loss')
        # cost=tf.losses.mean_squared_error(labels=layer_label,predictions=self.predict_op)
        return cost
