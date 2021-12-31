# -*- coding: utf-8 -*-
# @Time    : 2021/12/23
# @Author  : sunyihuan
# @File    : hybrid_quantization_step1.py

from rknn.api import RKNN

if __name__ == '__main__':

    INPUT_SIZE = 416
    tf_pb = "./checkpoint/yolov3_1220.pb"
    inputs = ['define_input/input_data']
    output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
              "define_loss/pred_lbbox/concat_2"]

    # Create RKNN object
    rknn = RKNN()

    # Set model config
    print('--> Config model')
    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2',target_platform=['rv1126'])
    print('done')

    # Load tensorflow model
    print('--> Loading model')

    ret = rknn.load_tensorflow(tf_pb=tf_pb,
                               inputs=inputs,
                               outputs=output,
                               input_size_list=[[INPUT_SIZE, INPUT_SIZE, 3]],
                               predef_file=None)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Hybrid quantization step1
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    # Tips
    print('Please modify ssd_mobilenet_v2.quantization.cfg!')
    print('==================================================================================================')
    print('Modify method:')
    print('Add {layer_name}: {quantized_dtype} to dict of customized_quantize_layers')
    print('If no layer changed, please set {} as empty directory for customized_quantize_layers')
    print('==================================================================================================')
    print('Notes:')
    print('1. The layer_name comes from quantize_parameters, please strip \'@\' and \':xxx\';')
    print('   If layer_name contains special characters, please quote the layer name.')
    print('2. Support quantized_type: asymmetric_affine-u8, dynamic_fixed_point-i8, dynamic_fixed_point-i16, float32.')
    print('3. Please fill in according to the grammatical rules of yaml.')
    print(
        '4. For this model, RKNN Toolkit has provided the corresponding configuration, please directly proceed to step2.')
    print('==================================================================================================')

    rknn.release()
