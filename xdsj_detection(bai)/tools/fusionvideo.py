import cv2
import numpy as np

INPUT_FILE1 = r'C:\\Users\\bai\\Desktop\\image\\testvideo\\1608629025.avi'
INPUT_FILE2 = r'C:\\Users\\bai\\Desktop\\image\\testvideo\\1608629045.avi'
INPUT_FILE3 = r'C:\\Users\\bai\\Desktop\\image\\testvideo\\1608629040.avi'
INPUT_FILE4 = r'C:\\Users\\bai\\Desktop\\image\\testvideo\\1608629032.avi'

OUTPUT_FILE = r'C:\\Users\\bai\\Desktop\\image\\testvideo\\merge\\merge.avi'

reader1 = cv2.VideoCapture(INPUT_FILE1)
reader2 = cv2.VideoCapture(INPUT_FILE2)
reader3 = cv2.VideoCapture(INPUT_FILE3)
reader4 = cv2.VideoCapture(INPUT_FILE4)

width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(OUTPUT_FILE,
                         cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                         2,  # fps
                         (width*2, height*2))  # resolution

print(reader1.isOpened())
print(reader2.isOpened())
print(reader3.isOpened())
print(reader4.isOpened())

have_more_frame = True
c = 0
while have_more_frame:
    have_more_frame, frame1 = reader1.read()
    _, frame2 = reader2.read()
    _, frame3 = reader3.read()
    _, frame4 = reader4.read()

    # frame1 = cv2.resize(frame1, (width // 2, height // 2))
    # frame2 = cv2.resize(frame2, (width // 2, height // 2))
    # frame3 = cv2.resize(frame3, (width // 2, height // 2))
    # frame4 = cv2.resize(frame4, (width // 2, height // 2))
    # img = np.hstack((frame1, frame2))

    image = np.concatenate([frame1, frame2], axis=1)  # axis=0时为垂直拼接；axis=1时为水平拼接
    image1 = np.concatenate([frame3, frame4], axis=1)

    image = np.concatenate([image, image1], axis=0)
    # img1 = np.hstack((frame1, frame2))
    # finalimg = np.vstack((img, img1))
    # img = np.vstack([frame1, frame2])
    cv2.waitKey(1)
    writer.write(image)
    c += 1
    print(str(c) + ' is ok')

writer.release()
reader1.release()
reader2.release()
cv2.destroyAllWindows()