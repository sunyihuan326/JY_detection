#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import xdsj_detection.core.utils as utils
import xdsj_detection.core.common as common
import xdsj_detection.core.backbone as backbone
from xdsj_detection.core.config import cfg
from keras import backend as K


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

        try:
            # self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data)
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_network_csp53(input_data)
            print(self.conv_lbbox)
            print(self.conv_mbbox)
            print(self.conv_sbbox)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
            print(self.pred_sbbox)

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data):

        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

        # input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        # input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        # input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        # input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def __build_network_csp53(self, input_data):
        '''Build yolov5 body, including SPP, PAN, Yolov3/v4 Head/Neck.
           param input_data: Input tensor, return: Three stage outputs'''

        init_width_size = int(64 * 0.5)
        init_depth_size = 4
        # if self.depth_scale == 0.33:
        #     init_depth_size = 1
        # elif self.depth_scale == 0.67:
        #     init_depth_size = 2
        # elif self.depth_scale == 1.33:
        #     init_depth_size = 4
        # else:
        #     init_depth_size = 3

        route_1, route_2, route_3, input_data, last_layer_num = backbone.cspdarknet53(input_data, self.trainable,
                                                                                      init_width_size, init_depth_size)

        layer_num = last_layer_num
        y19, layer_num = common.cspstage(input_data, self.trainable, 16 * init_width_size, init_depth_size, layer_num,
                                         4,
                                         1 + 7 * init_depth_size)
        y19_upsample = common.upsample(y19, name='upsample0', method=self.upsample_method)

        # 1024x38x38 -> 512x38x38
        y19_1 = common.convolutional(y19_upsample, (1, 1, 16 * init_width_size, 8 * init_width_size), self.trainable,
                                     name='conv%d' % (layer_num + 1))

        # 512x38x38 -> 512x76x76
        y19_upsample = common.upsample(y19_1, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_0'):
            y38 = common.convolutional(route_2, (1, 1, 8 * init_width_size, 8 * init_width_size), self.trainable,
                                       'conv_route_0')
            y38 = tf.concat([y38, y19_upsample], axis=-1)

            # 1024x76x76 -> 512x76x76
        y38 = common.convolutional(y38, (1, 1, 16 * init_width_size, 8 * init_width_size), self.trainable,
                                   name='conv%d' % (layer_num + 2))

        # 76x76 head/neck
        layer_num = layer_num + 3
        y38, layer_num = common.cspstage(y38, self.trainable, 8 * init_width_size, init_depth_size, layer_num, 5,
                                         1 + 8 * init_depth_size)

        # 512x76x76 -> 256x76x76
        y38_1 = common.convolutional(y38, (1, 1, 8 * init_width_size, 4 * init_width_size), self.trainable,
                                     name='conv%d' % (layer_num + 1))

        # 256x76x76 -> 256x152x152
        y38_upsample = common.upsample(y38_1, name='upsample2', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            y76 = common.convolutional(route_1, (1, 1, 4 * init_width_size, 4 * init_width_size), self.trainable,
                                       'conv_route_1')
            y76 = tf.concat([y76, y38_upsample], axis=-1)

        # 512x152x152 -> 256x152x152
        y76 = common.convolutional(y76, (1, 1, 8 * init_width_size, 4 * init_width_size), self.trainable,
                                   name='conv%d' % (layer_num + 2))

        layer_num = layer_num + 3
        y76, layer_num = common.cspstage(y76, self.trainable, 4 * init_width_size, init_depth_size, layer_num, 6,
                                         1 + 9 * init_depth_size)

        # 256x152x152 -> 256x76x76
        y76_downsample = common.convolutional(y76, (1, 1, 4 * init_width_size, 4 * init_width_size),
                                              trainable=self.trainable,
                                              name='downsample0', downsample=True)
        y76_output = common.convolutional(y76_downsample, (1, 1, 4 * init_width_size, 3 * (self.num_class + 5)),
                                          trainable=self.trainable,
                                          name='conv_sbbox', activate=False, bn=False)

        # 38x38 head/neck
        # 256x152x152 -> 256x76x76
        y38_1 = common.convolutional(y76, (3, 3, 4 * init_width_size, 4 * init_width_size), self.trainable,
                                     name='conv%d' % (layer_num + 1), downsample=True)
        with tf.variable_scope('route_2'):
            y38 = common.convolutional(route_2, (1, 1, 8 * init_width_size, 8 * init_width_size), self.trainable,
                                       'conv_route_2')
            y38 = tf.concat([y38, y38_1], axis=-1)

        # 768x76x76 -> 512x76x76
        y38 = common.convolutional(y38, (1, 1, 12 * init_width_size, 8 * init_width_size), self.trainable,
                                   name='conv%d' % (layer_num + 2))

        layer_num = layer_num + 3
        y38, layer_num = common.cspstage(y38, self.trainable, 8 * init_width_size, init_depth_size, layer_num, 7,
                                         1 + 10 * init_depth_size)

        # 512x76x76 -> 512x38x38
        y38_downsample = common.convolutional(y38, (1, 1, 8 * init_width_size, 8 * init_width_size),
                                              trainable=self.trainable,
                                              name='downsample1', downsample=True)
        y38_output = common.convolutional(y38_downsample, (1, 1, 8 * init_width_size, 3 * (self.num_class + 5)),
                                          trainable=self.trainable,
                                          name='conv_mbbox', activate=False, bn=False)

        # 19x19 head/neck
        # 512x76x76 -> 512x38x38
        y19_1 = common.convolutional(y38, (3, 3, 8 * init_width_size, 8 * init_width_size), self.trainable,
                                     name='conv%d' % (layer_num + 1), downsample=True)
        with tf.variable_scope('route_3'):
            y19 = common.convolutional(route_3, (1, 1, 16 * init_width_size, 16 * init_width_size), self.trainable,
                                       'conv_route_3')
            y19 = tf.concat([y19, y19_1], axis=-1)

        # 1536x38x38 -> 1024x38x38
        y19 = common.convolutional(y19, (1, 1, 24 * init_width_size, 16 * init_width_size), self.trainable,
                                   name='conv%d' % (layer_num + 2))

        layer_num = layer_num + 3
        y19, layer_num = common.cspstage(y19, self.trainable, 16 * init_width_size, init_depth_size, layer_num, 8,
                                         1 + 11 * init_depth_size)

        # 1024x38x38 -> 1024x19x19
        y19_downsample = common.convolutional(y19, (1, 1, 16 * init_width_size, 16 * init_width_size),
                                              trainable=self.trainable,
                                              name='downsample2', downsample=True)
        y19_output = common.convolutional(y19_downsample, (1, 1, 16 * init_width_size, 3 * (self.num_class + 5)),
                                          trainable=self.trainable,
                                          name='conv_lbbox', activate=False, bn=False)

        return y19_output, y38_output, y76_output

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
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_diou(self, boxes1, boxes2):
        '''
        计算diou = iou - p2/c2
        :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
        :param boxes2: (8, 13, 13, 3, 4)   label_xywh
        :return:
        '''

        # 变成左上角坐标、右下角坐标
        boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                     boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                     boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        '''
        逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
        比如留下了[x0, y0]
        这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
        '''
        boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                     tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
        boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                     tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

        # 两个矩形的面积
        boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
        boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

        # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
        left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
        right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

        # 相交矩形的面积inter_area。iou
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / tf.maximum(union_area, 1e-6)

        # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
        enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
        enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

        # 包围矩形的对角线的平方
        enclose_wh = enclose_right_down - enclose_left_up
        enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)

        # 两矩形中心点距离的平方
        p2 = K.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)

        diou = iou - 1.0 * p2 / tf.maximum(enclose_c2, 1e-6)
        return diou

    def bbox_ciou(self, boxes1, boxes2):
        '''
        计算diou = iou - p2/c2
        :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
        :param boxes2: (8, 13, 13, 3, 4)   label_xywh
        :return:
        '''

        boxes1_1, boxes2_1 = boxes1, boxes2
        boxes1_center, boxes2_center = boxes1[..., :2], boxes2[..., :2]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]), tf.maximum(boxes1[..., :2], boxes1[..., 2:])],
                           axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]), tf.maximum(boxes2[..., :2], boxes2[..., 2:])],
                           axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / (union_area + 1e-7)

        center_distance = tf.reduce_sum(tf.square(boxes1_center - boxes2_center), axis=-1)
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
        diou = iou - 1.0 * center_distance / (enclose_diagonal + 1e-7)

        v = 4 / (np.pi * np.pi) * (tf.square(tf.math.atan2(boxes1_1[..., 2], boxes1_1[..., 3]) -
                                             tf.math.atan2(boxes2_1[..., 2], boxes2_1[..., 3])))
        alp = v / (1.0 - iou + v + 1e-7)
        ciou = diou - alp * v
        return ciou

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-6)

        return giou

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
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        # giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        # diou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        ciou = tf.expand_dims(self.bbox_ciou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        # giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        # diou_loss = respond_bbox * bbox_loss_scale * (1 - diou)
        ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)

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

        # giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        # diou_loss = tf.reduce_mean(tf.reduce_sum(diou_loss, axis=[1, 2, 3, 4]))
        ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return ciou_loss, conf_loss, prob_loss

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
