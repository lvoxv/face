# !/usr/bin/python3.10
# -*- coding: utf-8 -*-
# Author:_TRISA_
# File:demo.py
# Time:2022/4/12 16:45
# Software:PyCharm
# Email:1628791325@QQ.com
# -U2hhcmUlMjBhbmQlMjBMb3Zl-base64


import cv2
import numpy as np

image="./3.png"

def ellipse_detect(image):
    """
    :param image: 图片路径
    :return: None
    """
    # image: ".\test.png"
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 15), 43, 0, 360, (255, 255, 255), -1)

    YCRCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = YCRCB[i, j, 1]
            CB = YCRCB[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin[i, j] = 255
    cv2.namedWindow(image, cv2.WINDOW_NORMAL)
    cv2.imshow(image, img)
    dst = cv2.bitwise_and(img, img, mask=skin)
    cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    cv2.imshow("cutout", dst)
    cv2.waitKey()


def cr_otsu(image):
    """YCrCb颜色空间的Cr分量+Otsu阈值分割
    :param image: 图片路径
    :return: None
    """
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.namedWindow("image raw", cv2.WINDOW_NORMAL)
    cv2.imshow("image raw", img)
    cv2.namedWindow("image CR", cv2.WINDOW_NORMAL)
    cv2.imshow("image CR", cr1)
    cv2.namedWindow("Skin Cr+OTSU", cv2.WINDOW_NORMAL)
    cv2.imshow("Skin Cr+OTSU", skin)

    dst = cv2.bitwise_and(img, img, mask=skin)
    cv2.namedWindow("seperate", cv2.WINDOW_NORMAL)
    cv2.imshow("seperate", dst)
    cv2.waitKey()


def crcb_range_sceening(image):
    """
    :param image: 图片路径
    :return: None
    """
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)
    print(cr,y,cb)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    print(cr.shape)
    for i in range(0, x):
        for j in range(0, y):
            if (cr[i][j] > 140) and (cr[i][j]) < 175 and (cr[i][j] > 100) and (cb[i][j]) < 120:
                skin[i][j] = 255
            else:
                skin[i][j] = 0
    cv2.namedWindow(image, cv2.WINDOW_NORMAL)
    cv2.imshow(image, img)
    cv2.namedWindow(image + "skin2 cr+cb", cv2.WINDOW_NORMAL)
    cv2.imshow(image + "skin2 cr+cb", skin)

    dst = cv2.bitwise_and(img, img, mask=skin)
    cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    cv2.imshow("cutout", dst)

    cv2.waitKey()

if __name__ == '__main__':
    #ellipse_detect(image)
    #cr_otsu(image)
    crcb_range_sceening(image)