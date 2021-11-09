#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.relu(conv)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output

def cspstage(input_data, trainable, filters, loop, layer_nums, route_nums, res_nums):
        '''CSPNets stage
            param input_data: The input tensor
            param trainable: A bool parameter, True ==> training, False ==> not train.
            param filters: Filter nums
            param loop: ResBlock loop nums
            param layer_nums: Counter of Conv layers
            param route_nums: Counter of route nums
            param res_nums: Counter of ResBlock nums
        return: Output tensors and the last Conv layer counter of this stage'''
        c = filters
        out_layer = layer_nums + 1 + loop + 1
        route = input_data
        route = convolutional(route, (1, 1, c, int(c / 2)), trainable=trainable, name='conv_route%d' % route_nums,
                     activate=True)
        input_data = convolutional(input_data, (1, 1, c, int(c / 2)), trainable=trainable, name='conv%d' % (layer_nums + 1),
                          activate=True)

        for i in range(loop):
            input_data = residual_block(input_data, int(c / 2), int(c / 2), int(c / 2), trainable=trainable,
                                   name='residual%d' % (i + res_nums))

        input_data = convolutional(input_data, (1, 1, int(c / 2), int(c / 2)), trainable=trainable, name='conv%d' % out_layer,
                          activate=True)
        input_data = tf.concat([input_data, route], axis=-1)
        return input_data, out_layer
