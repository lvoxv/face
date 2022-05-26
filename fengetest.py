# !/usr/bin/python3.10
# -*- coding: utf-8 -*-
# Author:_TRISA_
# File:fengetest.py
# Time:2022/5/20 22:08
# Software:PyCharm
# Email:1628791325@QQ.com
# -U2hhcmUlMjBhbmQlMjBMb3Zl-base64


from tkinter import *
import cv2
import numpy as np
# import time
import matplotlib.pyplot as plt
# PCT = "./Photo_test/IMG_2140.JPG"



def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 第一步读入图片
# img = cv2.imread('./Photo_test/6.jpg')
img = cv2.imread('Photo_test/7.jpg')
# 第二步：对图片做灰度变化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 第三步：对图片做二值变化
ret, thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
cv_show(thresh,'thresh')


# 第四步：获得图片的轮廓值
contours,h= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 第五步：在图片中画出图片的轮廓值
draw_img = img.copy()
rets = cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 2)
# 第六步：画出带有轮廓的原始图片
cv_show(rets, 'ret')


m,n = img.shape
gray.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim=[0, x], ylim=[0, y], title='An Example Axes',
       ylabel='Y-Axis', xlabel='X-Axis')
plt.show()
