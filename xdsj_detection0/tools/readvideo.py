# import cv2
#
# vid = cv2.VideoCapture('.\\image\\output_cloth.avi')
#
# # vid = cv2.VideoCapture(1)
# height, width = None, None
#
# if vid.isOpened():
#     success, np_img1 = vid.read()
# else:
#     success = False
#
# if not vid.isOpened():
#     print("Cannot open camera")
#     exit()
# else:
#     print("[INFO] warming up camera...")
#
# while True:
#     success, np_img1 = vid.read()
#     if not success:
#         print("Cannot receive frame. Exiting ...")
#         break
#
#     if width is None or height is None:
#         height, width = np_img.shape[:2]
#
#     resized_img = cv2.resize(np_img1, (448, 448))
#     np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
#
#     cv2.imshow("np_img", np_img)
#     # cv2.waitKey(10)
#
#     if cv2.waitKey(1) == ord("q"):
#         break
#
# print("[INFO] cleaning up...")
# cv2.destroyAllWindows()
# vid.release()


import numpy as np
import cv2 as cv
cap = cv.VideoCapture('.\\image\\output_cloth.avi')
while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
