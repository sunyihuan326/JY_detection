# -*- coding: utf-8 -*-
# @Time    : 2021/7/27
# @Author  : sunyihuan
# @File    : blur_detection.py
import os
import cv2
import shutil
import numpy as np
from skimage import filters
from tqdm import tqdm

class BlurDetection:
    def __init__(self, strDir):
        print("图片检测对象已经创建...")
        self.strDir = strDir

    def _getAllImg(self):
        """
        根据目录读取所有的图片
        :param strType: 图片的类型
        :return:  图片列表
        """
        names = []
        for file in os.listdir(self.strDir):  # 此处有bug  如果调试的数据还放在这里，将会递归的遍历所有文件
            if file.endswith(".jpg"):
                    names.append(str(file))
        return names

    def _imageToMatrix(self, image):
        """
        根据名称读取图片对象转化矩阵
        :param strName:
        :return: 返回矩阵
        """
        imgMat = np.matrix(image)
        return imgMat

    def _blurDetection(self, imgName):

        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        imgMat = self._imageToMatrix(img2gray) / 255.0
        x, y = imgMat.shape
        score = 0
        for i in range(x - 2):
            for j in range(y - 2):
                score += (imgMat[i + 2, j] - imgMat[i, j]) ** 2
        score = score / 10
        return score

    def TestBrener(self, blur_save_dir):
        if not os.path.exists(blur_save_dir): os.mkdir(blur_save_dir)
        imgList = self._getAllImg()

        for i in tqdm(range(len(imgList))):
            strPath = self.strDir + "/" + imgList[i]
            score = self._blurDetection(strPath)
            if score > 50:
                shutil.move(strPath, blur_save_dir + "/" + imgList[i])

        return

    def preImgOps(self, strPath):
        """
        图像的预处理操作
        :param imgName: 图像的而明朝
        :return: 灰度化和resize之后的图片对象
        """

        img = cv2.imread(strPath)  # 读取图片
        cv2.moveWindow("", 1000, 100)
        # cv2.imshow("原始图", img)
        # 预处理操作
        reImg = cv2.resize(img, (800, 600))  #
        img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
        return img2gray, reImg


if __name__ == "__main__":
    BlurDetection = BlurDetection(strDir="F:/robots_images_202107/20210719/12")
    BlurDetection.TestBrener("F:/robots_images_202107/20210719/12/bbbb")
