# -*- coding: utf-8 -*-
# @Time    : 2021/12/23
# @Author  : sunyihuan
# @File    : hybrid_quantization_step2.py

from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Set model config
    print('--> Config model')
    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2',target_platform=['rv1126'])
    print('done')

    # Hybrid quantization step2
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./yolov3_1220.json',
                                         data_input='./yolov3_1220.data',
                                         model_quantization_cfg='./yolov3_1220.quantization.cfg',
                                         dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./yolov3_1220.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    rknn.release()