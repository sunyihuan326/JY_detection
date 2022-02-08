# -*- coding: utf-8 -*-
# @Time    : 2021/11/17
# @Author  : sunyihuan
# @File    : pb2rknn.py

from rknn.api import RKNN

INPUT_SIZE =416

if __name__ == '__main__':
    # 创建RKNN执行对象
    rknn = RKNN()
    # 配置模型输入，用于NPU对数据输入的预处理
    # channel_mean_value='0 0 0 255'，那么模型推理时，将会对RGB数据做如下转换
    # (R - 0)/255, (G - 0)/255, (B - 0)/255。推理时，RKNN模型会自动做均值和归一化处理
    # reorder_channel=’0 1 2’用于指定是否调整图像通道顺序，设置成0 1 2即按输入的图像通道顺序不做调整
    # reorder_channel=’2 1 0’表示交换0和2通道，如果输入是RGB，将会被调整为BGR。如果是BGR将会被调整为RGB
    # 图像通道顺序不做调整
    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2',target_platform=['rv1126'])

    # 加载TensorFlow模型
    # tf_pb='digital_gesture.pb'指定待转换的TensorFlow模型
    # inputs指定模型中的输入节点
    # outputs指定模型中输出节点
    # input_size_list指定模型输入的大小
    print('--> Loading model')
    output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
              "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]
    pb_path="E:/JY_detection/xdsj_detection/checkpoint/padding_resize/mAP94/yolov3_test_loss=1.5903.ckpt-40"
    rknn.load_tensorflow(tf_pb=pb_path,
                         inputs=['define_input/input_data'],
                         outputs=output,
                         input_size_list=[[INPUT_SIZE, INPUT_SIZE, 3]])
    print('done')

    # 创建解析pb模型
    # do_quantization=False指定不进行量化
    # 量化会减小模型的体积和提升运算速度，但是会有精度的丢失
    print('--> Building model')
    rknn.build(do_quantization=True,dataset='./dataset.txt')
    print('done')


    # 导出保存rknn模型文件
    rknn.export_rknn('./yolov3_ap94.rknn')

    # Release RKNN Context
    rknn.release()
