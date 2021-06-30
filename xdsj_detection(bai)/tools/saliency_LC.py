import numpy as np
import time
import cv2
import heapq

def diag_sym_matrix(k=256):
    base_matrix = np.zeros((k,k))
    base_line = np.array(range(k))
    base_matrix[0] = base_line
    for i in range(1,k):
        base_matrix[i] = np.roll(base_line,i)
    base_matrix_triu = np.triu(base_matrix)
    return base_matrix_triu + base_matrix_triu.T

def cal_dist(hist):
    Diag_sym = diag_sym_matrix(k=256)
    hist_reshape = hist.reshape(1,-1)
    hist_reshape = np.tile(hist_reshape, (256, 1))
    return np.sum(Diag_sym*hist_reshape,axis=1)

def LC(image_gray):
    image_height,image_width = image_gray.shape[:2]
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    gray_dist = cal_dist(hist_array)

    image_gray_value = image_gray.reshape(1,-1)[0]
    image_gray_copy = [(lambda x: gray_dist[x]) (x)  for x in image_gray_value]
    image_gray_copy = np.array(image_gray_copy).reshape(image_height,image_width)
    image_gray_copy = (image_gray_copy-np.min(image_gray_copy))/(np.max(image_gray_copy)-np.min(image_gray_copy))
    return image_gray_copy

def saliency_function(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saliency_image = LC(image_gray)
    # saliency_image = LC(image_gray[300:799, 0:599])
    threshMap = cv2.threshold((saliency_image*255).astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(threshMap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))

    a = np.array(area)
    l = heapq.nlargest(1, range(len(a)), a.take)
    # print("&&&&&&&&&&&", l)
    if (l > [300]):
        for i in l:
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (153, 153, 0), 3)

        cv2.namedWindow("Thresh", 0)
        cv2.imshow("Thresh", threshMap)
        cv2.namedWindow("image_output", 0)
        cv2.imshow("image_output", image)
    else:
        cv2.namedWindow("image_output", 0)
        cv2.imshow("image_output", image)

if __name__ == '__main__':
    file = r"C:\Users\bai\Desktop\tensorflow-yolo-python\image\bottle1.jpg"

    start = time.time()
    image = cv2.imread(file)
    saliency_function(image)

    end = time.time()
    print("Duration: %.2f seconds." % (end - start))

    cv2.waitKey(0)
    cv2.destroyAllWindows()