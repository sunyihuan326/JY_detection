# -*-coding:utf-8_*_

#和摄像头垂直角度，形变系数有关

import cv2
import numpy

if __name__ == '__main__':
    img = cv2.imread(r'C:\JoyRobot\Project\modelcompression\YOLOv3-ModelCompression-MultidatasetTraining-Multibackbone-master\01500015.jpg')

    height = img.shape[0]
    width = img.shape[1]

    #画圆
    #cv2.circle(img, (int(width/2), int(height-1)), 200, (0, 255, 0), 3)

    #画椭圆
    ptCenter = (int(width/2), int(height-1)) # 中心点位置
    rotateAngle = 0 # 旋转角度为 90
    startAngle = 180
    endAngle = 360

    point_color = (0, 0, 255) # BGR
    thickness = 2
    lineType = 3

    # 绘制一个红色上半椭圆10cm
    axesSize1 = (500, 120) # 长轴半径为 1200，短轴半径为 300
    cv2.ellipse(img, ptCenter, axesSize1, rotateAngle, startAngle, endAngle, point_color, thickness, lineType)
    text1 = "10cm"
    cv2.putText(img, text1, (int(width/2 - 25), int(height - 105)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)

    # 绘制一个红色上半椭圆30cm
    axesSize2 = (500, 218) # 长轴半径为 1200，短轴半径为 555
    cv2.ellipse(img, ptCenter, axesSize2, rotateAngle, startAngle, endAngle, point_color, thickness, lineType)
    text2 = "30cm"
    cv2.putText(img, text2, (int(width/2 - 25), int(height-200)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)

    # 绘制一个红色上半椭圆50cm
    axesSize3 = (500, 240) # 长轴半径为 1200，短轴半径为 60
    cv2.ellipse(img, ptCenter, axesSize3, rotateAngle, startAngle, endAngle, point_color, thickness, lineType)
    text3 = "50cm"
    cv2.putText(img, text3, (int(width/2 - 25), int(height-225)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)

    #画矩形框，x和y为深度学习的参数
    colors = (0, 255, 0)
    x = 338
    y = 320
    cv2.rectangle(img, (x, y), (x + 120, y + 36), colors, 2)

    #判断矩形和椭圆是否相交
    #1.用CreateRectRgn建立矩形区域、用CreateEllipticRgn建立椭圆区域。
    #2.用CreateRectRgn建立结果区域。
    #3.用CombineRgn对矩形、椭圆区域进行and操作，并判断返回值，如果是0表示操作失败，如果是1则无交集，如果是其他值则有交集

    #输出类别和距离文字信息
    W = 50
    text_distance = "Slippers, Distance<=" + str(W) + "cm"
    cv2.putText(img, text_distance, (30, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 1)

    #绘制线段
    cv2.line(img, (200, height), (330, 365), (0, 255, 255), 2, cv2.LINE_4)
    cv2.line(img, (600, height), (470, 365), (0, 255, 255), 2, cv2.LINE_4)

    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)