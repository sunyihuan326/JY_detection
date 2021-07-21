# -*- encoding: utf-8 -*-

"""
ckpt文件转成pb文件

@File    : freeze_graph.py
@Time    : 2021/07/19 11:43
@Author  : sunyihuan
"""

import tensorflow as tf

typ = ""

pb_file = "./yolov3.pb"
ckpt_file = "E:/JY_detection/xdsj_detection/checkpoint/yolov3_test_loss=1.7009.ckpt-25"
#
# if typ == "tiny":
#     from xdsj_detection.core.yolov3_tiny import YOLOV3
#     output = ["define_loss/pred_mbbox/concat_2:0", "define_loss/pred_lbbox/concat_2:0"]
# else:
#     from xdsj_detection.core.yolov3 import YOLOV3
#     output = ["define_loss/pred_sbbox/concat_2:0", "define_loss/pred_mbbox/concat_2:0",
#               "define_loss/pred_lbbox/concat_2:0"]

with tf.name_scope('define_input'):  # 输出
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3), name='input_data')
    # trainable = tf.placeholder(dtype=tf.bool, name='training')

with tf.name_scope("define_loss"):  # 输出

    if typ == "tiny":
        from xdsj_detection.core.yolov3_tiny import YOLOV3

        model = YOLOV3(input_data, trainable=False)
        output = [model.pred_mbbox, model.pred_lbbox]
    else:
        from xdsj_detection.core.yolov3 import YOLOV3

        model = YOLOV3(input_data, trainable=False)
        output = [model.pred_sbbox, model.pred_mbbox, model.pred_lbbox]

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)
# 生成pb文件
# converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
#                                                                    input_graph_def=sess.graph.as_graph_def(),
#                                                                    output_node_names=output)
#
# with tf.gfile.GFile(pb_file, "wb") as f:
#     f.write(converted_graph_def.SerializeToString())
#     print("%d ops in the final graph." % len(converted_graph_def.node))  # 得到当前图有几个操作节点

converter = tf.lite.TFLiteConverter.from_session(sess, [input_data], output)
tflite_model = converter.convert()
open("./converted_model.tflite", "wb").write(tflite_model)
