#! /usr/bin/env python
# coding=utf-8

import xdsj_detection.core.common as common
import tensorflow as tf


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(3):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(3):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data


def cspdarknet53(input_data, trainable, init_width_size, init_depth_size):
    '''CSPDarknet53 body; source: https://arxiv.org/pdf/1911.11929.pdf
        param input_data: Input tensor
        param trainable: A bool parameter, True ==> training, False ==> not train.
    return: Three stage tensors'''
    # for debug to print net layers' shape, need to remark while train/val/test phase
    # input_data = tf.reshape(input_data, [-1, 608, 608, 3])

    # 3x608x608 -> 64x608x608
    input_data = common.convolutional(input_data, (3, 3, 3, init_width_size), trainable=trainable, name='conv0', activate=True)

    # 64x608x608 -> 128x304x304
    input_data = common.convolutional(input_data, (1, 1, init_width_size, 2 * init_width_size), trainable=trainable, name='conv1',
                      downsample=True, activate=True)

    layer_num = 1
    input_data, layer_num = common.cspstage(input_data, trainable, 2 * init_width_size, init_depth_size, layer_num, 1, 1)

    # 128x304x304 -> 256x152x152
    input_data = common.convolutional(input_data, (3, 3, 2 * init_width_size, 4 * init_width_size), trainable=trainable,
                      name='conv%d' % (layer_num + 1), downsample=True, activate=True)
    route_1 = input_data

    layer_num = layer_num + 1
    input_data, layer_num = common.cspstage(input_data, trainable, 4 * init_width_size, 3 * init_depth_size, layer_num, 2,
                                     1 + init_depth_size)

    # 256x152x152 -> 512x76x76
    input_data = common.convolutional(input_data, (3, 3, 4 * init_width_size, 8 * init_width_size), trainable=trainable,
                      name='conv%d' % (layer_num + 1), downsample=True, activate=True)
    route_2 = input_data

    layer_num = layer_num + 1
    input_data, layer_num = common.cspstage(input_data, trainable, 8 * init_width_size, 3 * init_depth_size, layer_num, 3,
                                     1 + 4 * init_depth_size)

    # 512x76x76 -> 1024x38x38
    input_data = common.convolutional(input_data, (3, 3, 8 * init_width_size, 16 * init_width_size), trainable=trainable,
                      name='conv%d' % (layer_num + 1), downsample=True, activate=True)
    route_3 = input_data

    # SPP
    maxpool1 = tf.nn.max_pool(input_data, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
    maxpool2 = tf.nn.max_pool(input_data, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
    maxpool3 = tf.nn.max_pool(input_data, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
    input_data = tf.concat([maxpool1, maxpool2, maxpool3, input_data], axis=-1)

    # 4096x38x38 -> 1024x38x38
    input_data = common.convolutional(input_data, (1, 1, 64 * init_width_size, 16 * init_width_size), trainable=trainable,
                      name='conv%d' % (layer_num + 2), downsample=True,  activate=True)
    last_layer_num = layer_num + 2
    return route_1, route_2, route_3, input_data, last_layer_num