#!usr/bin/python
# -*- coding: utf-8 -*-

# import the necessary packages
import numpy as np
import cv2

def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)

    print("cv2.minAreaRect(c) = ", cv2.minAreaRect(c))
    return cv2.minAreaRect(c)

# compute and return the distance from the maker to the camera
def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def count_focalLength(IMAGE_PATHS, KNOWN_DISTANCE, KNOWN_WIDTH, KNOWN_HEIGHT):


    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    image = cv2.imread(IMAGE_PATHS[0])
    marker = find_marker(image)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    # focalLength = 811.82
    print("focalLength = ", focalLength)
    return focalLength

if __name__ == "__main__":
    # initialize the list of images that we'll be using
    IMAGE_PATHS = ["./camera_40cm.jpg", "./camera_30cm.jpg"]
    # initialize the known distance from the camera to the object, which
    # in this case is 24 inches
    KNOWN_DISTANCE = 11.00
    #KNOWN_DISTANCE = 15.75 #11.81

    # initialize the known object width, which in this case, the piece of
    # paper is 11 inches wide
    KNOWN_WIDTH = 11.69
    KNOWN_HEIGHT = 8.27

    focalLength = count_focalLength(IMAGE_PATHS, KNOWN_DISTANCE, KNOWN_WIDTH, KNOWN_HEIGHT)

    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        # get a frame
        (grabbed, frame) = camera.read()
        marker = find_marker(frame)
        if marker == 0:
            print(marker)
            continue
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

        # draw a bounding box around the image and display it
        box = np.int0(cv2.boxPoints(marker))
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        cv2.putText(frame, "%.2fcm" % (inches * 30.48 / 12),
                    (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()