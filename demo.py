# !/usr/bin/python3.10
# -*- coding: utf-8 -*-
# Author:_TRISA_
# File:demo.py
# Time:2022/4/12 16:45
# Software:PyCharm
# Email:1628791325@QQ.com
# -U2hhcmUlMjBhbmQlMjBMb3Zl-base64

from tkinter import *
import cv2
import numpy as np
# import time

PCT = "./Photo_test/7.JPG"
Image = cv2.imread(PCT)
Size = Image.shape

# window = Tk()
# window.title("test")
# window.mainloop()
# def readimg():
#     image = cv2.imread(PCT)
#     size = image.shape
#     return image, size

def morphology(img):  # 形态学操作
    k = np.ones((3, 3), np.uint8)
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)  # 开运算
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, k)  # 闭运算
    cv2.namedWindow("open", cv2.WINDOW_NORMAL)
    cv2.imshow("open", open)
    cv2.namedWindow("close", cv2.WINDOW_NORMAL)
    cv2.imshow("close", close)
    cv2.waitKey()
    return close


def region(img):
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # 查看各个返回值
    # 连通域数量
    print('num_labels = ', num_labels)
    # 连通域的信息：对应各个轮廓的左上角坐标x、y、width、height和面积
    print('stats = ', stats)
    # 连通域的中心点
    print('centroids = ', centroids)
    # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    print('labels = ', labels)

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)
    # cv2.namedWindow("region", cv2.WINDOW_NORMAL)
    # cv2.imshow('region', output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return num_labels, stats


# def cr_otsu(image):
#     """YCrCb颜色空间的Cr分量+Otsu阈值分割
#     :param image: 图片路径
#     :return: None
#     """
#     img = cv2.imread(image, cv2.IMREAD_COLOR)
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#
#     (y, cr, cb) = cv2.split(ycrcb)
#     cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
#     _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # cv2.namedWindow("image raw", cv2.WINDOW_NORMAL)
#     # cv2.imshow("image raw", img)
#     # cv2.namedWindow("image CR", cv2.WINDOW_NORMAL)
#     # cv2.imshow("image CR", cr1)
#     # cv2.namedWindow("Skin Cr+OTSU", cv2.WINDOW_NORMAL)
#     # cv2.imshow("Skin Cr+OTSU", skin)
#     # morphology(skin)
#     # # dst = cv2.bitwise_and(img, img, mask=skin)
#     # # cv2.namedWindow("seperate", cv2.WINDOW_NORMAL)
#     # # cv2.imshow("seperate", dst)
#     # cv2.waitKey()
#     return skin


def selectskin(stats):
    num = len(stats)
    size_total = Size[1] * Size[0]
    tips = []
    for i in range(1, num):
        long = max(stats[i][2], stats[i][3])
        short = min(stats[i][2], stats[i][3])
        # ratio = long / short
        ratio = stats[i][3] / stats[i][2]
        size = stats[i][4]
        if 1.1 <= ratio <= 2 and size/size_total >= 0.001:
            tips.append(i)
        else:
            continue
    return tips

def signskin(tips, stats):
    if tips:
        signimg = Image
        for k in range(0, len(tips)):
            i = tips[k]
            pt1 = (stats[i][0], stats[i][1])
            pt2 = (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3])
            signimg = cv2.rectangle(Image, pt1, pt2, (0, 255, 0), 2, 4)
        return signimg
    else:
        print("Dont find face")
        return Image


def crcb_range_sceening(image):
    """
    :param image: 图片路径
    :return: None
    """
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)
    # print(cr,y,cb)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    # print(cr.shape)
    for i in range(0, x):
        for j in range(0, y):
            if (cr[i][j] > 160 ): # ) and (cr[i][j]) < 173 and (cb[i][j] > 77) and (cb[i][j]) < 127:
                skin[i][j] = 255
            else:
                skin[i][j] = 0
    print(skin)
    return skin
    # cv2.namedWindow(image, cv2.WINDOW_NORMAL)
    # cv2.imshow(image, img)
    # cv2.namedWindow(image + "skin2 cr+cb", cv2.WINDOW_NORMAL)
    # cv2.imshow(image + "skin2 cr+cb", skin)
    #
    # dst = cv2.bitwise_and(img, img, mask=skin)
    # cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    # cv2.imshow("cutout", dst)
    #
    # cv2.waitKey()

def main():
    skin = crcb_range_sceening(PCT)
    morskin = morphology(skin)
    num_labels, stats = region(morskin)
    skininstats = selectskin(stats)
    # 画框
    signimg = signskin(skininstats, stats)
    cv2.namedWindow("region", cv2.WINDOW_NORMAL)
    cv2.imshow('region', signimg)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ellipse_detect(image)
    # cr_otsu(image)
    main()
    # crcb_range_sceening(image)
    # face_date(image) #这个是后面那个自己想的代码 还没成功过


