import os
import cv2

for filename in os.listdir(r"C:\Users\bai\Desktop\image\JPEGImages\2\dishcloth"):
    filename_final = os.path.join("C:", "\\Users", "bai", "Desktop", "image","JPEGImages","2","dishcloth", str(filename))
    print (filename_final)

    image=cv2.imread(filename_final)
    # cv2.namedWindow('image', 0)
    # cv2.imshow('image', image)

    res=cv2.resize(image,(800,600),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(filename_final, res)