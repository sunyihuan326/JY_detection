# -*- coding: utf-8 -*-
# @Time    : 2021/10/20
# @Author  : sunyihuan
# @File    : tt.py

# from cnstd import CnStd
# from cnocr import CnOcr
#
# std = CnStd()
# cn_ocr = CnOcr()
#
# box_info_list = std.detect('1111.jpg')
#
# for box_info in box_info_list:
#     cropped_img = box_info['cropped_img']  # 检测出的文本框
#     ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
#     print('ocr result: %s' % ''.join(ocr_res))
#
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr_for_single_line('1111.jpg')
print("Predicted Chars:", res)