# -*- coding: utf-8 -*-
# @Time    : 2021/7/19
# @Author  : sunyihuan
# @File    : covert_lite.py
import tensorflow as tf

input_data=["define_input/input_data"]
output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
          "define_loss/pred_lbbox/concat_2"]
pb_file = "./yolov3.pb"

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    pb_file, input_data, output)

tflite_model = converter.convert()
open("./converted_model.tflite", "wb").write(tflite_model)
