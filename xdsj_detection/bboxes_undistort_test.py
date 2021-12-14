# -*- coding: utf-8 -*-
# @Time    : 2021/12/9
# @Author  : sunyihuan
# @File    : bboxes_undistort_test.py

from xdsj_detection.distance_and_angle import *
import xdsj_detection.core.utils as utils

if __name__ == "__main__":
    img_path = "20210915023920544113.jpg"
    image = cv2.imread(img_path)  # 图片读取
    # image = cv2.resize(image, (640, 360))
    image_un = image_undistort(image)  # 图片畸变矫正

    bboxes_p = [[1, 436, 495, 484, 1.0, 6]]
    print("bboxes_p****", bboxes_p)
    image = utils.draw_bbox(image, bboxes_p, show_label=True)

    # 矫正bboxes坐标
    bboxes = []
    # if len(bboxes_p) > 0:
    #     for b in bboxes_p:
    #
    #         # 找上边缘最大的点(即y_min)
    #         b_up_min = [b[0], 0]
    #         for x_ in range(int(b[0]), int(b[2]), 3):
    #             b_up = np.array([float(x_), float(b[1])])
    #             # print("b_up:::::", b_up)
    #             b0 = bboxes_undistort(b_up)
    #             # print("b0*****", b0)
    #             if b0[1] > b_up_min[1]:  # 寻找最大y值
    #                 b_up_min = b0
    #         y_up = b_up_min[1]
    #         # print("y_up::::", y_up)
    #
    #         # 找下边缘最小的点(即y_max)
    #         b_bottom_max = [b[2], 720]
    #         for x_ in range(int(b[0]), int(b[2]), 3):
    #             b_up = np.array([float(x_), float(b[3])])
    #             b0 = bboxes_undistort(b_up)
    #             if b0[1] < b_bottom_max[1]:  # 寻找最小y值
    #                 b_bottom_max = b0
    #         y_bottom = b_bottom_max[1]
    #
    #         # 找左边缘最大的点(即x_min)
    #         b_left_min = [0, b[1]]
    #         for y_ in range(int(b[1]), int(b[3]), 3):
    #             b_up = np.array([float(b[0]), float(y_)])
    #             b0 = bboxes_undistort(b_up)
    #             if b0[0] > b_left_min[0]:  # 寻找最大x值
    #                 b_left_min = b0
    #         x_left = b_left_min[0]
    #
    #         # 找右边缘最小的点(即x_max)
    #         b_right_max = [1280, b[3]]
    #         for y_ in range(int(b[1]), int(b[3]), 3):
    #             b_up = np.array([float(b[2]), float(y_)])
    #             b0 = bboxes_undistort(b_up)
    #             if b0[0] < b_right_max[0]:  # 寻找最小x值
    #                 b_right_max = b0
    #         x_rigt = b_right_max[0]
    #
    #         b[0] = x_left
    #         b[1] = y_up
    #         b[2] = x_rigt
    #         b[3] = y_bottom
    #         bboxes.append(b)


    cv2.imwrite("20210915023920544113_bboxes.jpg", image)
    image_bboxes = cv2.imread("20210915023920544113_bboxes.jpg")  # 图片读取
    image_bboxes_un = image_undistort(image_bboxes)  # 图片畸变矫正
    cv2.imwrite("20210915023920544113_bboxes_un.jpg", image_bboxes_un)

    # image_un = utils.draw_bbox(image_un, bboxes, show_label=True)
    # cv2.imshow("image_un", image_un)
    # cv2.imshow("image", image)
    #
    # cv2.waitKey(0)
