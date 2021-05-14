# -*- encoding: utf-8 -*-

"""
量化压缩，未完成！！！！！！！

@File    : tflite_quantization.py
@Time    : 202003/1/9 9:32
@Author  : sunyihuan
"""

from absl import app, flags, logging
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

graph_def_file = 'model/yolo_model.pb'

# 查看pb节点
# with tf.Session() as sess:
#     with open(graph_def_file, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#         tf.import_graph_def(graph_def, name='')
#         tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
#         for tensor_name in tensor_name_list:
#             print(tensor_name,'\n')
#
# input_names = ["define_input/input_data"]
# output_names = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
#                 "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]
#
# input_tensor = {input_names[0]: [1, 320, 320, 3]}

# uint8 quant
# converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops = True
#
# converter.inference_type = tf.uint8  # tf.lite.constants.QUANTIZED_UINT8
# input_arrays = converter.get_input_arrays()
# converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)}  # mean, std_dev
# converter.default_ranges_stats = (0, 255)
#
# tflite_uint8_model = converter.convert()
# open("uint8.tflite", "wb").write(tflite_uint8_model)

save_model_path="pb_model"

converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

tflite_model = converter.convert()
open("yolo_lite_int8.tflite", 'wb').write(tflite_model)

logging.info("model saved to: {}".format("yolo_lite_int8.tflite"))