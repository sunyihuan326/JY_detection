# -*- encoding: utf-8 -*-

"""
将ckpt文件转为pb和saved_model文件

@File    : freeze_graph.py
@Time    : 2019/12/21 11:43
@Author  : sunyihuan
"""

import tensorflow.compat.v1 as tf
from kx_detection.core.yolov3 import YOLOV3

pb_file = "./yolov3_151_0.pb"
savemodel_file_path = "./savemodel/yolov3/3"

ckpt_file = "./checkpoint/yolov3_train_loss=5.9575.ckpt-151"
output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
          "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]

with tf.name_scope('define_input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3), name='input_data')
    # trainable = tf.placeholder(dtype=tf.bool, name='training')

with tf.name_scope("define_loss"):
    model = YOLOV3(input_data, trainable=False)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(converted_graph_def.node))  # 得到当前图有几个操作节点
print("pb saved::::::::::::::::::::..................")
builder = tf.saved_model.builder.SavedModelBuilder(savemodel_file_path)
builder.add_meta_graph_and_variables(sess, ["serve"])

print("save...")
builder.save()
