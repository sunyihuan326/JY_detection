# -*- coding: utf-8 -*-
# @Time    : 2021/12/14
# @Author  : sunyihuan
# @File    : blur_detector.py
import numpy as np
import cv2
import imutils


def detect_blur_fft(image, size=60, thresh=10):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return mean, mean <= thresh


if __name__ == "__main__":
    image_path = "F:/clear/01100044.jpg"
    orig = cv2.imread(image_path)
    orig = imutils.resize(orig, width=500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # apply our blur detector using the FFT
    (mean, blurry) = detect_blur_fft(gray)
    image = np.dstack([gray] * 3)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2)
    print("[INFO] {}".format(text))
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
