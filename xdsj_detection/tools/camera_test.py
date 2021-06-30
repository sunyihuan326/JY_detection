# -*- coding:utf-8 -*-

import cv2
import sys
import time

start = time.time()

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret_flag, img_camera = cap.read()
    cv2.imshow("camera", img_camera)
    # cv2.imwrite("camera.jpg", img_camera)

    # break
    print(img_camera.shape)
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
totaltime = (time.time() - start)*1000
print("totaltime = ", totaltime, "ms")


cap.release()

cv2.destroyAllWindows()



