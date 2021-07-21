# -*- coding: utf-8 -*-
# @Time    : 2021/7/19
# @Author  : sunyihuan
# @File    : covert_lite.py
import tensorflow as tf

input_data=["define_input/input_data:0"]
output = ["define_loss/pred_sbbox/concat_2:0", "define_loss/pred_mbbox/concat_2:0",
          "define_loss/pred_lbbox/concat_2:0"]
pb_file = "./yolov3.pb"
input_shapes = {"define_input/input_data": [1, 416, 416, 3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(
    pb_file, input_data, output, input_shapes)

tflite_model = converter.convert()
open("./converted_model.tflite", "wb").write(tflite_model)
