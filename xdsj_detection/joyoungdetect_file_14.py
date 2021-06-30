# __*__coding:utf-8__*__

import sys
import os

sys.path.append('./')

from xdsj_detection.yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np
import time
import argparse
import heapq
import copy
import math

# python joyoungdetect_file.py -f=".\\image\\folader"
# python joyoungdetect_file.py -v=".\\image\\output_cloth.avi"

classes_name = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]

# classes_display = {'01': 'flip-flops(black)', '02': 'dustbin(plastic)', '03': 'dishcloth(green)',
#                    '04': 'dustbin(stainless)','05': 'slipper(grey)', '06': 'dishcloth(blue)',
#                    '07': 'gym-shoes(black)', '08': 'cotton-socks(black)','09': 'stockings(flesh)',
#                    '10': 'socks(khaki)', '11': 'power-cord(white)', '12': 'power-cord(black)',
#                    '13': 'network-cable(grey)'}

# classes_display = {'01': 'slipper', '02': 'dustbin', '03': 'dishcloth',
#                    '04': 'dustbin','05': 'slipper', '06': 'dishcloth',
#                    '07': 'shoes', '08': 'socks','09': 'socks',
#                    '10': 'socks', '11': 'line', '12': 'line',
#                    '13': 'line', '14':'None'}

classes_display = {'01': 'shoes', '02': 'dustbin', '03': 'dishcloth',
                   '04': 'dustbin', '05': 'shoes', '06': 'dishcloth',
                   '07': 'shoes', '08': 'socks', '09': 'dishcloth',
                   '10': 'socks', '11': 'line', '12': 'line',
                   '13': 'line', '14': 'None'}


def diag_sym_matrix(k=256):
    base_matrix = np.zeros((k, k))
    base_line = np.array(range(k))
    base_matrix[0] = base_line
    for i in range(1, k):
        base_matrix[i] = np.roll(base_line, i)
    base_matrix_triu = np.triu(base_matrix)
    return base_matrix_triu + base_matrix_triu.T


def cal_dist(hist):
    Diag_sym = diag_sym_matrix(k=256)
    hist_reshape = hist.reshape(1, -1)
    hist_reshape = np.tile(hist_reshape, (256, 1))
    return np.sum(Diag_sym * hist_reshape, axis=1)


def LC(image_gray):
    image_height, image_width = image_gray.shape[:2]
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    gray_dist = cal_dist(hist_array)

    image_gray_value = image_gray.reshape(1, -1)[0]
    image_gray_copy = [(lambda x: gray_dist[x])(x) for x in image_gray_value]
    image_gray_copy = np.array(image_gray_copy).reshape(image_height, image_width)
    image_gray_copy = (image_gray_copy - np.min(image_gray_copy)) / (np.max(image_gray_copy) - np.min(image_gray_copy))
    return image_gray_copy

# 显著性检测
def saliency_function(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saliency_image = LC(image_gray)
    # saliency_image = LC(image_gray[100:448, 0:448])

    threshMap = cv2.threshold((saliency_image * 255).astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(threshMap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))

    a = np.array(area)
    l = heapq.nlargest(1, range(len(a)), a.take)
    # print("&&&&&&&&&&&", l)
    if (l > [200]):
        for i in l:
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (153, 153, 0), 3)

        # cv2.namedWindow("Thresh", 0)
        # cv2.imshow("Thresh", threshMap)


# predict function
def process_predicts(predicts):
    p_classes = predicts[0, :, :, 0:14]
    C = predicts[0, :, :, 14:16]
    coordinate = predicts[0, :, :, 16:]
    p_classes = np.reshape(p_classes, (7, 7, 1, 14))
    C = np.reshape(C, (7, 7, 2, 1))
    P = C * p_classes

    # print P[5,1, 0, :]
    index = np.argmax(P)
    index = np.unravel_index(index, P.shape)
    class_num = index[3]
    coordinate = np.reshape(coordinate, (7, 7, 2, 4))
    max_coordinate = coordinate[index[0], index[1], index[2], :]

    xcenter = max_coordinate[0]
    ycenter = max_coordinate[1]
    w = max_coordinate[2]
    h = max_coordinate[3]
    xcenter = (index[1] + xcenter) * (448 / 7.0)
    ycenter = (index[0] + ycenter) * (448 / 7.0)

    w = w * 448
    h = h * 448
    xmin = xcenter - w / 2.0
    ymin = ycenter - h / 2.0
    xmax = xmin + w
    ymax = ymin + h
    return xmin, ymin, xmax, ymax, class_num


# compute and return the distance from the maker to the camera
def distance_to_camera(knownWidth, focalLength, perWidth, ymax):
    # Countinches = (ymax / 448) * np.e**(ymax / 200)
    # 虚拟比例尺，利用指数函数去拟合，距离越近变化越小，距离越远变化越快，其中2.75是拟合值根据摄像头焦距可变。
    Countinches = (ymax / 448) * np.e ** 2.75
    # 小孔成像测距，通过相似三角形原理进行计算
    Ostiole = (knownWidth * focalLength) / perWidth
    # 预研阶段需求变化快，以小孔成像测距为主，工装误差降低时，以虚拟比例尺测距为主
    if (abs(Ostiole - Countinches) < 5):
        return Ostiole
    else:
        return Ostiole + (Countinches / 40)
    # return Ostiole


# compute angle
def compute_angle(x1, y1, x2, y2):
    # 坐标值转化成单精度
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    # print("x1=%s, x2=%s, y1=%s, y2=%s" % (x1, x2, y1, y2))
    # 进行逻辑判断
    if (x2 - x1 == 0):  # 无偏角
        result = 0
    elif (y2 - y1 == 0):  # 计算导数为0的情况
        if (x2 < x1):
            result = -90
        else:
            result = 90
    else:
        # 计算两点的斜率得到tan值，然后求arctan得到角度
        k = -(y2 - y1) / (x2 - x1)
        result = np.arctan(k) * 57.29577
        # 进行角度值的正负换算，以便于输出理解
        if result < 0:
            result = -(90 - abs(result))
        else:
            result = 90 - abs(result)
    return result


# def count_focalLength(KNOWN_DISTANCE, KNOWN_WIDTH, KNOWN_HEIGHT):
#     # initialize the list of images that we'll be using
#     IMAGE_PATHS = ["./camera_40cm.jpg", "./camera_30cm.jpg"]
#
#     # load the furst image that contains an object that is KNOWN TO BE 2 feet
#     # from our camera, then find the paper marker in the image, and initialize
#     # the focal length
#     image = cv2.imread(IMAGE_PATHS[0])
#     marker = find_marker(image)
#     focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
#
#     # 通过摄像头标定获取的像素焦距
#     # focalLength = 811.82
#     return focalLength

if __name__ == '__main__':

    # # 使用CPU资源进行推断
    # # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # # config = tf.ConfigProto(gpu_options=gpu_options)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # config = tf.ConfigProto(device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
    #                         inter_op_parallelism_threads=0,
    #                         intra_op_parallelism_threads=0,
    #                         log_device_placement=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image-path',
                        type=str,
                        help='The path to the image file')
    parser.add_argument('-f', '--folder-path',
                        type=str,
                        help='The path to the image fi'
                             'le')
    parser.add_argument('-v', '--video-path',
                        type=str,
                        default="E:/JY_detection/xdsj_detection/image/video/distance.mp4",
                        help='The path to the video file')
    parser.add_argument('-vo', '--video-output-path',
                        type=str,
                        default='./image/video/output.avi',
                        help='The path of the output video file')

    model_path="E:/JY_detection/xdsj_detection/models/train_syh/model.ckpt-10000"

    focalLength = 191.82  # 811.82
    # KNOWN_WIDTH = 11.69 #19.68  #50cm inches 11.69  1 inches = 2.54 cm
    KNOWN_WIDTH = [13.19, 8.93, 7.94, 8.36, 13.69, 11.99,
                   14.17, 9.64, 8.84, 9.94, 11.69, 12.69,
                   11.69, 11.19]

    KNOWN_HEIGHT = 8.27  # 63cm inches 8.27
    # input_model = 0 #0/1/2
    FLAGS, unparsed = parser.parse_known_args()
    print("FLAGS = ", FLAGS)

    # import param
    common_params = {'image_size': 448, 'num_classes': 14, 'batch_size': 16}
    net_params = {'cell_size': 7, 'boxes_per_cell': 2, 'weight_decay': 0.0005}

    image_width = common_params['image_size']
    image_heigh = common_params['image_size']

    # import net
    net = YoloTinyNet(common_params, net_params, test=True)

    image = tf.placeholder(tf.float32, (1, image_width, image_heigh, 3))
    predicts = net.inference(image)

    # start session
    sess = tf.Session()

    # If both image and video files are given then raise error
    if FLAGS.image_path is None and FLAGS.video_path is None and FLAGS.folder_path is None:
        print('Neither path to an image or path to video provided')
        print('Starting Inference on Webcam')

    # Do inference with given image
    if FLAGS.image_path:
        print("This is image_path mode")
        # Read the image
        try:
            np_img = cv2.imread(FLAGS.image_path)

            # image_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(np_img, (image_width, image_heigh))
            Detectimg = copy.deepcopy(resized_img)
            np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            np_img[0:150, :, :] = 0

        except:
            print('Image cannot be loaded!\n\
                                Please check the path provided!')
        finally:
            # saliency detection
            saliency_function(Detectimg)
            cv2.namedWindow("image_output", 0)
            cv2.imshow("image_output", Detectimg)

            np_img = np_img.astype(np.float32)
            np_img = np_img / 255.0 * 2 - 1
            np_img = np.reshape(np_img, (1, image_width, image_heigh, 3))

            saver = tf.train.Saver(net.trainable_collection)
            # import model
            saver.restore(sess,model_path)

            # import image and net
            np_predict = sess.run(predicts, feed_dict={image: np_img})

            # predict
            start = time.time()
            xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)

            inches = 0
            marker = xmax - xmin
            if marker == 0:
                print(marker)

            xmin = min(max(xmin, 0), xmax)
            ymin = min(max(ymin, 0), ymax)
            xmax = max(min(xmax, image_heigh), xmin)
            ymax = max(min(ymax, image_width), ymin)

            inches = distance_to_camera(KNOWN_WIDTH[int(class_num)], focalLength, marker, ymax)

            totaltime = (time.time() - start) * 1000
            print("totaltime = ", totaltime * 100, "ms")

            l = 0
            # deg_tmp = compute_angle(image_width / 2, image_heigh, (xmin + xmax) / 2, ymax)
            deg_tmp = compute_angle(image_width / 2, image_heigh, (xmin + xmax) / 2, ymax - l)
            degleft_tmp = compute_angle(image_width / 2, image_heigh, xmin, ymax - l)
            degright_tmp = compute_angle(image_width / 2, image_heigh, xmax, ymax - l)
            deg_left_tmp = (xmin / 448) * 198.39

            # rectangle
            # class_name = classes_name[class_num]
            cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax) - 10), (0, 0, 255))
            # cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
            cv2.putText(resized_img, classes_display[class_num], (int(xmin), int(ymin - 10)), 2, 1.0, (0, 0, 255))
            cv2.putText(resized_img, "%.2fcm, %.2fdeg." % (inches * 30.48 / 12, deg_tmp),
                        (resized_img.shape[1] - 350, resized_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

            cv2.namedWindow("out.jpg", 0)
            cv2.imshow('out.jpg', resized_img)
            cv2.imwrite('./image/result/out.jpg', resized_img)
            cv2.waitKey(0)

    elif FLAGS.folder_path:
        print("This is folder_path mode")
        # Read the image
        for root, dir, files in os.walk(FLAGS.folder_path):
            for file in files:
                folder_image_path = root + "\\" + str(file)
                try:
                    np_img = cv2.imread(folder_image_path)
                    # cv2.imshow("np_img", np_img)
                    # cv2.waitKey(0)
                    # image_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
                    resized_img = cv2.resize(np_img, (image_width, image_heigh))
                    Detectimg = copy.deepcopy(resized_img)
                    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                    # np_img[0:165,:,:] = 0

                except:
                    print('Image cannot be loaded!\n\
                                         Please check the path provided!')

                finally:
                    # saliency detection
                    saliency_function(Detectimg)
                    # cv2.namedWindow("image_output", 0)
                    # cv2.imshow("image_output", Detectimg)
                    # cv2.waitKey(0)

                    np_img = np_img.astype(np.float32)
                    np_img = np_img / 255.0 * 2 - 1
                    np_img = np.reshape(np_img, (1, image_width, image_heigh, 3))

                    saver = tf.train.Saver(net.trainable_collection)
                    # import model
                    # saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
                    saver.restore(sess, model_path)

                    # import image and net
                    np_predict = sess.run(predicts, feed_dict={image: np_img})

                    # predict
                    start = time.time()
                    xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
                    # print("&&&&&7", class_num)

                    inches = 0
                    marker = xmax - xmin
                    if marker == 0:
                        print(marker)

                    xmin = min(max(xmin, 0), xmax)
                    ymin = min(max(ymin, 0), ymax)
                    xmax = max(min(xmax, image_heigh), xmin)
                    ymax = max(min(ymax, image_width), ymin)

                    inches = distance_to_camera(KNOWN_WIDTH[int(class_num)], focalLength, marker, ymax)

                    totaltime = (time.time() - start) * 1000
                    print("totaltime = ", totaltime * 100, "ms")

                    l = 0
                    deg_tmp = compute_angle(image_width / 2, image_heigh, (xmin + xmax) / 2, ymax - l)
                    degleft_tmp = compute_angle(image_width / 2, image_heigh, xmin, ymax - l)
                    degright_tmp = compute_angle(image_width / 2, image_heigh, xmax, ymax - l)
                    # deg_left_tmp = (xmin / 448) * 198.39

                    # rectangle
                    class_name = classes_name[class_num]
                    cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax) - l), (0, 0, 255))
                    # cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
                    # cv2.line(resized_img, (int(xmin), int(ymax - l)), (224, 448), (0, 0, 255), 1, cv2.LINE_AA)
                    # cv2.line(resized_img, (int(xmax), int(ymax - l)), (224, 448), (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.putText(resized_img, classes_display[class_name], (int(xmin), int(ymin)), 2, 1.0, (0, 0, 255))
                    # cv2.putText(resized_img, "%.2fcm, %.2fdeg." % (inches * 30.48 / 12, deg_tmp),
                    #             (resized_img.shape[1] - 350, resized_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    #             1.0, (0, 255, 0), 2)

                    cv2.putText(resized_img, "%.2fcm, left:%.2fdeg., right:%.2fdeg." % (
                        inches * 30.48 / 12, degleft_tmp, degright_tmp),
                                (resized_img.shape[1] - 420, resized_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 0), 2)

                    # cv2.circle(resized_img, (int((xmin + xmax) / 2), int(ymax - l)), 5, (0, 0, 255), -1)
                    # cv2.circle(resized_img, (224, 448), 5, (0, 0, 255), -1)

                    # cv2.namedWindow("out.jpg", 0)
                    # cv2.imshow('out.jpg', resized_img)
                    # cv2.imwrite(os.path.dirname(root) + "\\" + "outImage" + "\\" + "out_" + str(file), resized_img)
                    cv2.imwrite(".\\image\\" + "showImage" + "\\" + "out_" + str(file), resized_img)
                    print("********************")
                    print(os.path.dirname(root) + "\\" + "showImage" + "\\" + "out_" + str(file))
                    print("********************")
                    # cv2.waitKey(0)
                    cv2.waitKey(10)

    elif FLAGS.video_path:
        print("This is video_path mode")
        # Read the video
        while (True):
            try:
                vid = cv2.VideoCapture(FLAGS.video_path)
                height, width = None, None
                writer = None

            except:
                print('Video cannot be loaded!\n\
                                         Please check the path provided!')

            finally:
                saver = tf.train.Saver(net.trainable_collection)
                # import model
                saver.restore(sess, model_path)
                while (vid.isOpened()):
                    grabbed, np_img = vid.read()

                    # print("$$$$$$$$$4", image_width, image_heigh)

                    # Checking if the complete video is read
                    if not grabbed:
                        break

                    if width is None or height is None:
                        height, width = np_img.shape[:2]

                    resized_img = cv2.resize(np_img, (image_width, image_heigh))
                    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                    # 上方视野不需要处理，降低干扰，减少计算量
                    np_img[0:165, :, :] = 0

                    np_img = np_img.astype(np.float32)
                    np_img = np_img / 255.0 * 2 - 1
                    np_img = np.reshape(np_img, (1, image_width, image_heigh, 3))

                    # import image and net
                    start = time.time()
                    np_predict = sess.run(predicts, feed_dict={image: np_img})

                    # predict
                    xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)

                    totaltime = (time.time() - start) * 1000
                    print("totaltime = ", totaltime, "ms")

                    inches = 0
                    marker = xmax - xmin
                    if marker == 0:
                        print(marker)
                        continue

                    xmin = min(max(xmin, 0), xmax)
                    ymin = min(max(ymin, 0), ymax)
                    xmax = max(min(xmax, image_heigh), xmin)
                    ymax = max(min(ymax, image_width), ymin)

                    inches = distance_to_camera(KNOWN_WIDTH[int(class_num)], focalLength, marker, ymax)

                    # totaltime = (time.time() - start) * 1000
                    # print("totaltime = ", totaltime, "ms")

                    deg_tmp = compute_angle(image_width / 2, image_heigh, (xmin + xmax) / 2, ymax - 10)

                    print("(xmin + xmax)/2, ymax = ", (xmin + xmax) / 2, ymax - 10)

                    # rectangle
                    class_name = classes_name[class_num]
                    cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax) - 10), (0, 0, 255))

                    cv2.line(resized_img, (int(xmin), int(ymax - 10)), (224, 448), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.line(resized_img, (int(xmax), int(ymax - 10)), (224, 448), (0, 0, 255), 1, cv2.LINE_AA)

                    # cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
                    cv2.putText(resized_img, classes_display[class_name], (int(xmin), int(ymin - 10)), 2, 1.0,
                                (0, 0, 255))
                    cv2.putText(resized_img, "%.2fcm, %.2fdeg." % (inches * 30.48 / 12, deg_tmp),
                                (resized_img.shape[1] - 350, resized_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2)
                    cv2.circle(resized_img, (int((xmin + xmax) / 2), int(ymax - 10)), 5, (0, 0, 255), -1)
                    cv2.circle(resized_img, (224, 448), 5, (0, 0, 255), -1)

                    cv2.imshow('out.jpg', resized_img)
                    cv2.waitKey(10)

                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                 (np_img.shape[2], np_img.shape[1]), True)
                    # print("np_img.shape[1], np_img.shape[0] = ", np_img.shape[2], np_img.shape[1], np_img.shape[0])
                    writer.write(resized_img)

                print("[INFO] Cleaning up...")
                writer.release()
                vid.release()

    # Infer real-time on webcam

    else:
        print("This is camera mode")
        vid = cv2.VideoCapture(0)
        height, width = None, None

        saver = tf.train.Saver(net.trainable_collection)
        # import model
        saver.restore(sess, model_path)

        while True:
            grabbed, np_img = vid.read()
            resized_img = cv2.resize(np_img, (image_width, image_heigh))
            Detectimg = copy.deepcopy(resized_img)
            np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            # 上方视野不需要处理，降低干扰，减少计算量
            # np_img[0:165,:,:] = 0
            saliency_function(Detectimg)

            # 视频流是否读取
            if not grabbed:
                break

            if width is None or height is None:
                height, width = np_img.shape[:2]

            # 图像reshape
            np_img = np_img.astype(np.float32)
            np_img = np_img / 255.0 * 2 - 1
            np_img = np.reshape(np_img, (1, image_width, image_heigh, 3))

            # 导入图像和网络参数
            np_predict = sess.run(predicts, feed_dict={image: np_img})

            # 结果预测
            xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)

            inches = 0
            p = 35
            marker = max(xmax - xmin - 2 * p, 50)

            # 坐标校正
            xmin = min(max(xmin, 0), xmax)
            ymin = min(max(ymin, 0), ymax)
            xmax = max(min(xmax, image_heigh), xmin)
            ymax = max(min(ymax, image_width), ymin)

            # 距离计算
            inches = distance_to_camera(KNOWN_WIDTH[int(class_num)], focalLength, marker, ymax)
            if marker / 448 >= 0.5 and ymin < 80 and ymax > 350:
                inches = max(inches - 16, 4.12)

            l = 0
            # 角度计算
            # deg_tmp = compute_angle(image_width / 2, image_heigh, (xmin + xmax) / 2, max(ymax-p, 10))
            degleft_tmp = compute_angle(image_width / 2, image_heigh, min(xmin + p, 440), max(ymax - p, 10))
            degright_tmp = compute_angle(image_width / 2, image_heigh, max(xmax - p, 10), max(ymax - p, 10))
            # deg_left_tmp = (xmin / 448)* 198.39
            degleft_tmp = (max(degleft_tmp * 0.7, -69.83)) if degleft_tmp < -25 else (max(degleft_tmp * 0.5, -69.83))
            degright_tmp = (min(degright_tmp * 0.7, 69.83)) if degright_tmp > 25 else (min(degright_tmp * 0.5, 69.83))

            # 过近或过远时的距离校正
            if ymax > 400 and (xmin < 40 or xmax > 400) and marker < 200:
                inches = 7 + (inches / 40)
                degleft_tmp -= 11.1
                degright_tmp += 13.2

            # rectangle，类型区分不明时的后处理
            class_name = classes_name[class_num]
            if (class_name == '08' or (
                    (class_name == '06' or class_name == '01') and marker < 180)) and ymin > 200 and ymax > 380 and (
                    xmin < 60 or xmax > 400):
                inches = max(inches - 9.5, 6.8 + (inches / 20))
                class_name = '08'
            scale = (xmax - xmin) / max(1, ymax - ymin)

            # 剔除异常情况
            if (class_name != '14') and not (class_name == '11' and scale > 2):
                cv2.rectangle(resized_img, (int(min(xmin + p, 440)), int(ymin)),
                              (int(max(xmax - p, 10)), int(max(ymax - p, 10))), (
                                  0, 0, 255))
                # 图上添加线条，方便演示
                # cv2.line(resized_img, (int(xmin), int(ymax - l)), (224, 448), (0, 0, 255), 1, cv2.LINE_AA)
                # cv2.line(resized_img, (int(xmax), int(ymax - l)), (224, 448), (0, 0, 255), 1, cv2.LINE_AA)
                # cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
                cv2.putText(resized_img, classes_display[class_name], (int(xmin), int(ymin - 10)), 2, 1.0, (0, 0, 255))
                cv2.putText(resized_img,
                            "%.2fcm, left:%.2fdeg., right:%.2fdeg." % (inches * 30.48 / 12, degleft_tmp, degright_tmp),
                            (resized_img.shape[1] - 420, resized_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 0), 2)
            # 图上添加圆点，方便演示
            # cv2.circle(resized_img, (int((xmin + xmax) / 2), int(ymax - l)), 5, (0, 0, 255), -1)
            # cv2.circle(resized_img, (224, 448), 5, (0, 0, 255), -1)

            cv2.namedWindow('camera_output', 0)
            cv2.imshow('camera_output', resized_img)
            cv2.waitKey(10)

            # 点击s键，保存图像
            k = cv2.waitKey(0)
            if k == ord('s'):
                t = time.localtime()
                str_time = str(time.asctime(t)).replace(' ', '').replace(':', '')
                dir = "C:\\Users\\bai\\Desktop\\tensorflow-yolo-python\\data\\123\\" + str_time + ".jpg"
                dir_deal = "C:\\Users\\bai\\Desktop\\tensorflow-yolo-python\\data\\123\\" + str_time + "_deal.jpg"
                cv2.imwrite(dir, Detectimg)
                cv2.imwrite(dir_deal, resized_img)
            # 点击空格或回车键，计算下一帧图像
            elif k == 13 or k == 32:
                continue
        vid.release()
    # 结束会话
    cv2.destroyAllWindows()
    sess.close()
