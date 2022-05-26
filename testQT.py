import sys
import cv2 as cv
import numpy
import numpy as np

import datetime
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import savgol_filter

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from UI.untitledtest import Ui_MainWindow



class PyQtMainEntry(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = cv.VideoCapture(0)
        self.is_camera_opened = False  # 摄像头有没有打开标记

        self.trainDate = []
        self.lable = []

        # 定时器：30ms捕获一帧
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(30)

    # 按钮操作函数 打开和关闭摄像头
    def btnOpenCamera_Clicked(self):
        '''
        打开和关闭摄像头
        '''
        self.is_camera_opened = ~self.is_camera_opened
        if self.is_camera_opened:
            self.btnOpenCamera.setText("关闭摄像头")
            self._timer.start()
        else:
            self.btnOpenCamera.setText("打开摄像头")
            self.labelCamera.clear()
            self.labelCamera.setText("摄像头")
            self._timer.stop()

    # 按钮操作函数 捕获摄像头图片
    def btnCapture_Clicked(self):
        '''
        捕获图片
        '''
        # 摄像头未打开，不执行任何操作
        if not self.is_camera_opened:
            return

        self.captured = self.frame

        self.Gray = cv.cvtColor(self.captured,cv.COLOR_RGB2GRAY)




        # 后面这几行代码几乎都一样，可以尝试封装成一个函数
        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        # Qt显示图片时，需要先转换成QImgage类型
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # 按钮操作函数 从本地读取图片
    def btnReadImage_Clicked(self):
        '''
        从本地读取图片
        '''
        # 打开文件选取对话框
        filename, _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.captured = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)

            self.Gray = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)##

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    # # 肤色分割人脸
    # def crcb_range_sceening(self):
    #     """
    #     # :param image: 图片路径
    #     # :return: None
    #     """
    #     # self.img = cv.imread(, cv.IMREAD_COLOR)
    #     ycrcb = cv.cvtColor(self.captured, cv.COLOR_RGB2YCR_CB)
    #     (y, cr, cb) = cv.split(ycrcb)
    #     # print(cr,y,cb)
    #     self.skin = np.zeros(cr.shape, dtype=np.uint8)
    #     (x, y) = cr.shape
    #     # print(cr.shape)
    #     for i in range(0, x):
    #         for j in range(0, y):
    #             if (cr[i][j] > 140) and (cr[i][j]) < 175 and (cr[i][j] > 100) and (cb[i][j]) < 120:
    #                 self.skin[i][j] = 255
    #             else:
    #                 self.skin[i][j] = 0
    #
    # # 形态学操作
    # def morphology(self):
    #     k = np.ones((3, 3), np.uint8)
    #     self.open = cv.morphologyEx(self.skin, cv.MORPH_OPEN, k)  # 开运算
    #     self.close = cv.morphologyEx(self.open, cv.MORPH_CLOSE, k)  # 闭运算
    #
    # # 连通域分析
    # def region(self):
    #     # 连通域分析
    #     num_labels, labels, self.stats, centroids = cv.connectedComponentsWithStats(self.close, connectivity=8)
    #
    #     # # 查看各个返回值
    #     # # 连通域数量
    #     # print('num_labels = ', num_labels)
    #     # # 连通域的信息：对应各个轮廓的左上角坐标x、y、width、height和面积
    #     # print('stats = ', stats)
    #     # # 连通域的中心点
    #     # print('centroids = ', centroids)
    #     # # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    #     # print('labels = ', labels)
    #
    #     # 不同的连通域赋予不同的颜色
    #     # self.output = np.zeros((self.close.shape[0], self.close.shape[1], 3), np.uint8)
    #     # for i in range(1, num_labels):
    #     #     mask = labels == i
    #     #     self.output[:, :, 0][mask] = np.random.randint(0, 255)
    #     #     self.output[:, :, 1][mask] = np.random.randint(0, 255)
    #     #     self.output[:, :, 2][mask] = np.random.randint(0, 255)
    #     # cv2.namedWindow("region", cv2.WINDOW_NORMAL)
    #     # cv2.imshow('region', output)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #
    # # 判断人脸连通域
    # def selectskin(self):
    #     num = len(self.stats)
    #     size_total = self.captured.shape[1] * self.captured.shape[0]
    #     self.tips = []
    #     for i in range(0, num):
    #         long = max(self.stats[i][2], self.stats[i][3])
    #         short = min(self.stats[i][2], self.stats[i][3])
    #         # ratio = long / short
    #         ratio = self.stats[i][3] / self.stats[i][2]
    #         size = self.stats[i][4]
    #         if 1.1 <= ratio <= 2 and size / size_total >= 0.001:
    #             self.tips.append(i)
    #         else:
    #             continue
    #
    # # 对人脸连通域画框
    # def signskin(self):
    #     if self.tips:
    #         self.result = self.captured
    #         for k in range(0, len(self.tips)):
    #             i = self.tips[k]
    #             pt1 = (self.stats[i][0], self.stats[i][1])
    #             pt2 = (self.stats[i][0] + self.stats[i][2], self.stats[i][1] + self.stats[i][3])
    #             self.result = cv.rectangle(self.result, pt1, pt2, (0, 255, 0), 2, 4)
    #     else:
    #         self.result = self.captured
    #         box = QtWidgets.QMessageBox()
    #         box.warning(self, "提示", "Dont find face")
    #         print("Dont find face")
    #
    #
    # def cutface(self):
    #     self.saveface =
    #
    # # 定位人眼并画框
    # def findeyes(self):
    #     count =0
    #     eyes = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")
    #     self.xxx = eyes.detectMultiScale(self.Gray,1.1,5)
    #     for (x,y,w,h) in self.xxx:
    #         self.result = cv.rectangle(self.captured,(x,y),(x+w,y+h),(255,0,0),2)
    #         count += 1
    #         face_img = cv.resize(self.Gray[y:y + h, x:x + w], (200, 200))
    #         face_filename = '%d.jpg' % (count)
    #         cv.imwrite(face_filename, face_img)

    def FaceDetect(self):
        # 肤色阈值分割
        ycrcb = cv.cvtColor(self.captured, cv.COLOR_RGB2YCR_CB)
        (y, cr, cb) = cv.split(ycrcb)
        # print(cr,y,cb)
        self.skin = np.zeros(cr.shape, dtype=np.uint8)
        (x, y) = cr.shape
        # print(cr.shape)
        for i in range(0, x):
            for j in range(0, y):
                if (cr[i][j] > 140) and (cr[i][j]) < 175 and (cr[i][j] > 100) and (cb[i][j]) < 120:
                    self.skin[i][j] = 255
                else:
                    self.skin[i][j] = 0

        # 形态学操作
        k = np.ones((3, 3), np.uint8)
        self.open = cv.morphologyEx(self.skin, cv.MORPH_OPEN, k)  # 开运算
        self.close = cv.morphologyEx(self.open, cv.MORPH_CLOSE, k)  # 闭运算

        # 连通域分析
        num_labels, labels, self.allstats, centroids = cv.connectedComponentsWithStats(self.close, connectivity=8)

        # 判断人脸连通域
        num = len(self.allstats)
        size_total = self.captured.shape[1] * self.captured.shape[0]
        self.tips = []
        for i in range(0, num):
            long = max(self.allstats[i][2], self.allstats[i][3])
            short = min(self.allstats[i][2], self.allstats[i][3])
            # ratio = long / short
            ratio = self.allstats[i][3] / self.allstats[i][2]
            size = self.allstats[i][4]
            if 1.1 <= ratio <= 2 and size / size_total >= 0.001:
                self.tips.append(i)
            else:
                continue

        # 对人脸连通域画框
        if self.tips:
            self.result = self.captured
            for k in range(0, len(self.tips)):
                i = self.tips[k]
                pt1 = (self.allstats[i][0], self.allstats[i][1])
                pt2 = (self.allstats[i][0] + self.allstats[i][2], self.allstats[i][1] + self.allstats[i][3])
                self.result = cv.rectangle(self.result, pt1, pt2, (0, 255, 0), 2, 4)
        else:
            self.result = self.captured
            box = QtWidgets.QMessageBox()
            box.warning(self, "提示", "Dont find face")
            print("Dont find face")

    def btnSignFace_Clicked(self):
        '''
        # ''
        # 灰度化
        # ''
        # # 如果没有捕获图片，则不执行操作
        # if not hasattr(self, "captured"):
        #     return
        #
        # # self.cpatured = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        # # 全局均衡化
        # # self.equalize = cv.equalizeHist(self.cpatured)
        # # 局部均衡化
        # ## createCLAHE(clipLimit=None, tileGridSize=None)
        # # clahe = cv.createCLAHE(tileGridSize=(5, 5))
        # # self.equalize = clahe.apply(self.cpatured)
        #
        # # ycrcb = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
        # r, g, b = cv.split(self.captured)
        # # R,G,B = cv.split(ycrcb)
        #
        # # r =cv.equalizeHist(r)
        # # g = cv.equalizeHist(g)
        # # b =cv.equalizeHist(b)
        #
        # clahe = cv.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
        # r = clahe.apply(r)
        # g = clahe.apply(g)
        # b = clahe.apply(b)
        #
        # self.equalize = cv.merge((r, g, b))
        # self.captured = self.equalize
        # # self.equalize = cv.cvtColor(aaa)
        '''
        self.FaceDetect()
        for i in self.tips:
            x,y,w,h,s = self.allstats[i]
            # 将灰度图人脸区域copy出来
            self.faceGray = self.Gray[y:y+h,x:x+w].copy()
            # 将原图彩色图人脸区域copy出来
            self.face = self.captured[y:y+h,x:x+w].copy()
            # loctime = datetime.datetime.now().strftime("%H_%M_%S.")
            # name = loctime + str(i) + ".png"
            # path = './Date/trainPhoto/'
            # cv.imwrite(path+name, self.retval, [cv.IMWRITE_PNG_COMPRESSION, 0])

        self.faceGray = cv.resize(self.faceGray,(256,384))
        self.face = cv.resize(self.face, (256, 384))
        # (y, cr, cb) = cv.split(faceycrcb)
        # # print(cr,y,cb)
        # self.mouth = np.zeros(cr.shape, dtype=np.uint8)
        # (x, y) = cr.shape
        # # print(cr.shape)
        # for i in range(0, x):
        #     for j in range(0, y):
        #         if (cr[i][j] > 160):
        #             self.mouth[i][j] = 255
        #         else:
        #             self.mouth[i][j] = 0
        # # 形态学操作
        # k = np.ones((3, 3), np.uint8)
        # open = cv.morphologyEx(self.mouth, cv.MORPH_OPEN, k)  # 开运算
        # close = cv.morphologyEx(open, cv.MORPH_CLOSE, k)  # 闭运算
        #
        # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(close, connectivity=8)


############################################


        m, n = self.faceGray.shape

        col = np.zeros(n)
        row = np.zeros(m)

        for i in range(1, m):
            r = 0
            for k in range(1, n):
                r += abs(int(self.faceGray[i, k]) - int(self.faceGray[i, k - 1]))
                row[i] = r

        for i in range(1, n):
            r = 0
            for k in range(1, m):
                r += abs(int(self.faceGray[k, i]) - int(self.faceGray[k - 1, i]))
                col[i] = r

        # 平滑行
        rrow = savgol_filter(row, 51, 3)
        # 平滑列
        rcol = savgol_filter(col, 51, 3)

        # 眼睛位置
        hang = int(np.where(rrow == np.max(rrow))[0])
        hang1 = hang - int(m / 12)
        hang2 = hang + int(m / 12)
        # 横向彩色眼睛
        hengxiang = self.face[hang1:hang2, 0:n]
        # 横向灰度眼睛
        hengxianghui = self.faceGray[hang1:hang2, 0:n]

        # 绘制表格
        # plt.figure(1)
        # plt.plot(list(range(0, m)), rrow)
        # plt.show()
        # plt.figure(2)
        # plt.plot(list(range(0, n)), rcol)
        # plt.show()

        # otsu
        _, eyesThreshold = cv.threshold(hengxianghui, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.bitwise_not(eyesThreshold, eyesThreshold)

        # cv.imshow('Eyes', hengxiang)
        # cv.imshow('EyesThreshold', eyesThreshold)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # ycrcb = cv.cvtColor(, cv2.COLOR_BGR2YCR_CB)
        faceycrcb = cv.cvtColor(self.face, cv.COLOR_RGB2YCR_CB)
        (Fy, Fcr, Fcb) = cv.split(faceycrcb)
        # print(cr,y,cb)
        mouth = np.zeros(Fcr.shape, dtype=np.uint8)
        (x, y) = Fcr.shape
        # print(cr.shape)
        for i in range(0, x):
            for j in range(0, y):
                if Fcr[i][j] > 160:
                    mouth[i][j] = 255
                else:
                    mouth[i][j] = 0
        self.Signresultold = mouth
        self.Signresultold[hang1:hang2, 0:n] = eyesThreshold
        # cv.imwrite('aaa.jpg',self.Signresult)
        self.Signresult = self.Signresultold.reshape(1,-1)
        self.Signresultlist = self.Signresult.tolist()

        self.trainDate.append(self.Signresultlist[0])

        lab = str(self.lineEdit.text())
        self.lable.append(lab)
        # print(self.lable)

        rows, columns = self.Signresultold.shape
        # rows, columns, channels = self.retval.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(self.Signresultold.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))




        



    # def train_Clicked(self):
    #     recognizer = cv.face.LBPHFaceRecognizer_create()
    #     recognizer.train(faces, np.array(ids))
    #     recognizer.write('trainer/trainer.yml')



    @QtCore.pyqtSlot()
    def _queryFrame(self):
        '''
        循环捕获图片
        '''
        ret, self.frame = self.camera.read()

        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols

        cv.cvtColor(self.frame, cv.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCamera.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # def lunkuo(self):
    #     k = cv.imread("./1.jpg")
    #     image,contours,hierarchy = cv.findContours(k, 2, 1)
    #     cv.imwrite("test.jpg", contours)

    def btnRecognize_Clicked(self):
        '''
        执行程序
        '''
        if not hasattr(self, "captured"):
            return

        # _, self.cpatured = cv.threshold(
        #     self.cpatured, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # self.crcb_range_sceening()
        # self.morphology()
        # self.region()
        # self.selectskin()
        # self.signskin()
        # self.findeyes()
        # self.lunkuo()

        self.FaceDetect()
        for i in self.tips:
            x, y, w, h, s = self.allstats[i]
            # 将灰度图人脸区域copy出来
            self.faceGray = self.Gray[y:y + h, x:x + w].copy()
            # 将原图彩色图人脸区域copy出来
            self.face = self.captured[y:y + h, x:x + w].copy()
            # loctime = datetime.datetime.now().strftime("%H_%M_%S.")
            # name = loctime + str(i) + ".png"
            # path = './Date/trainPhoto/'
            # cv.imwrite(path+name, self.retval, [cv.IMWRITE_PNG_COMPRESSION, 0])

        self.faceGray = cv.resize(self.faceGray, (256, 384))
        self.face = cv.resize(self.face, (256, 384))
        # (y, cr, cb) = cv.split(faceycrcb)
        # # print(cr,y,cb)
        # self.mouth = np.zeros(cr.shape, dtype=np.uint8)
        # (x, y) = cr.shape
        # # print(cr.shape)
        # for i in range(0, x):
        #     for j in range(0, y):
        #         if (cr[i][j] > 160):
        #             self.mouth[i][j] = 255
        #         else:
        #             self.mouth[i][j] = 0
        # # 形态学操作
        # k = np.ones((3, 3), np.uint8)
        # open = cv.morphologyEx(self.mouth, cv.MORPH_OPEN, k)  # 开运算
        # close = cv.morphologyEx(open, cv.MORPH_CLOSE, k)  # 闭运算
        #
        # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(close, connectivity=8)

        ############################################

        m, n = self.faceGray.shape

        col = np.zeros(n)
        row = np.zeros(m)

        for i in range(1, m):
            r = 0
            for k in range(1, n):
                r += abs(int(self.faceGray[i, k]) - int(self.faceGray[i, k - 1]))
                row[i] = r

        for i in range(1, n):
            r = 0
            for k in range(1, m):
                r += abs(int(self.faceGray[k, i]) - int(self.faceGray[k - 1, i]))
                col[i] = r

        # 平滑行
        rrow = savgol_filter(row, 51, 3)
        # 平滑列
        rcol = savgol_filter(col, 51, 3)

        # 眼睛位置
        hang = int(np.where(rrow == np.max(rrow))[0])
        hang1 = hang - int(m / 12)
        hang2 = hang + int(m / 12)
        # 横向彩色眼睛
        hengxiang = self.face[hang1:hang2, 0:n]
        # 横向灰度眼睛
        hengxianghui = self.faceGray[hang1:hang2, 0:n]

        # 绘制表格
        # plt.figure(1)
        # plt.plot(list(range(0, m)), rrow)
        # plt.show()
        # plt.figure(2)
        # plt.plot(list(range(0, n)), rcol)
        # plt.show()

        # otsu
        _, eyesThreshold = cv.threshold(hengxianghui, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.bitwise_not(eyesThreshold, eyesThreshold)

        # cv.imshow('Eyes', hengxiang)
        # cv.imshow('EyesThreshold', eyesThreshold)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # ycrcb = cv.cvtColor(, cv2.COLOR_BGR2YCR_CB)
        faceycrcb = cv.cvtColor(self.face, cv.COLOR_RGB2YCR_CB)
        (Fy, Fcr, Fcb) = cv.split(faceycrcb)
        # print(cr,y,cb)
        mouth = np.zeros(Fcr.shape, dtype=np.uint8)
        (x, y) = Fcr.shape
        # print(cr.shape)
        for i in range(0, x):
            for j in range(0, y):
                if Fcr[i][j] > 160:
                    mouth[i][j] = 255
                else:
                    mouth[i][j] = 0
        self.waitResult = mouth
        self.waitResult[hang1:hang2, 0:n] = eyesThreshold
        self.waitResult = self.waitResult.reshape(1, -1)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.trainDate,self.lable)
        pRe = knn.predict(self.waitResult)
        # print(knn.predict(self.waitResult))
        box = QtWidgets.QMessageBox()
        msg = "该人脸是" + pRe[0] + "的人脸"
        box.warning(self, "提示", msg)

        # 下面的是图像显示在GUI中代码
        rows, cols, channels = self.result.shape # 改
        bytesPerLine = channels * cols
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

