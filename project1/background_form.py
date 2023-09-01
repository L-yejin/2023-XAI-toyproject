# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'background_form.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

import numpy as np
import cv2
import os
import torch

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QPushButton,
    QSizePolicy, QTextBrowser, QToolButton, QWidget, QFileDialog)
from segmentation import predict_pt, ImageInference

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(489, 363)
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(160, 250, 151, 81))
        self.toolButton = QToolButton(Form)
        self.toolButton.setObjectName(u"toolButton")
        self.toolButton.setGeometry(QRect(220, 110, 191, 21))
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(70, 110, 131, 21))
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(70, 190, 81, 17))
        self.textBrowser = QTextBrowser(Form)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(50, 20, 371, 51))
        self.comboBox = QComboBox(Form)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(260, 190, 151, 25))
        self.toolButton_2 = QToolButton(Form)
        self.toolButton_2.setObjectName(u"toolButton_2")
        self.toolButton_2.setGeometry(QRect(220, 150, 191, 21))
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(70, 150, 141, 21))

        self.retranslateUi(Form)
        self.toolButton.clicked.connect(self.upload_image)
        self.toolButton_2.clicked.connect(self.upload_background)
        self.now_class = self.comboBox.currentText()
        self.comboBox.currentTextChanged.connect(self.set_class)
        # Enter
        self.pushButton.clicked.connect(self.click_enter)

        QMetaObject.connectSlotsByName(Form)
    # setupUi
    
    # Enter 박스 이름 찾아서 눌렀을 때 정해져있는 경로에서 Class이름.pt 를 가져와서 Segmentation Inference를 진행
    def click_enter(self):
        print('Click Enter!')
        
        # pt 경로 불러오기
        print('경로: ',os.getcwd())
        pt_name = self.now_class.split("_")[-1]
        pt = f'Final_pth/{pt_name}.pth' # f'./Final_pth/{self.now_class}.pth'
        pt = os.path.join(os.getcwd(),pt)
        print('pt 경로',pt)
        self.mask = predict_pt(pt, self.image_name)
        print('mask shape: ',self.mask.shape)
        print('Masking Finished!!')

        # self.image_name = QFileDialog.getOpenFileName()[0] # 업로드한 이미지 경로가 저장
        # self.back_name = QFileDialog.getOpenFileName()[0] # 업로드한 이미지 경로가 저장
        # self.now_class = self.comboBox.currentText()
        
        # 기존 코드에 추가함
        # ImageInference 클래스 인스턴스 생성
        inference = ImageInference()
        # model 가중치
        inference.load_model(self.now_class)
        # 추론
        prediction_result = inference.predict(self.image_name)
        # 백그라운드 교체
        inference.replace_background(False, self.image_name, self.back_name, prediction_result)

        # self.replace_background(False, self.image_name, self.back_name, self.mask)
        
    
    def upload_image(self):
        self.image_name = QFileDialog.getOpenFileName()[0] # 업로드한 이미지 경로가 저장
        print('Image Uploaded!!')
        print(self.image_name)
        
    def upload_background(self):
        self.back_name = QFileDialog.getOpenFileName()[0] # 업로드한 이미지 경로가 저장
        print('Background Uploaded!!')
        print(self.back_name)
        
    def set_class(self):
        print('Class Changed!!')
        self.now_class = self.comboBox.currentText()
        print(self.now_class)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"Enter", None))
        self.toolButton.setText(QCoreApplication.translate("Form", u"...", None))
        self.label.setText(QCoreApplication.translate("Form", u"Upload Your Image", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Class Name", None))
        self.textBrowser.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:20pt; font-style:italic; color:#2e3436;\">&lt; Change Your BackGroud &gt;</span></p></body></html>", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("Form", u"1_camera", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("Form", u"2_kettle", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("Form", u"3_cleaning-brush", None))
        self.comboBox.setItemText(3, QCoreApplication.translate("Form", u"4_book", None))
        self.comboBox.setItemText(4, QCoreApplication.translate("Form", u"5_bulb", None))
        self.comboBox.setItemText(5, QCoreApplication.translate("Form", u"6_speaker", None))
        self.comboBox.setItemText(6, QCoreApplication.translate("Form", u"7_ornament", None))
        self.comboBox.setItemText(7, QCoreApplication.translate("Form", u"8_big_camera", None))
        self.comboBox.setItemText(8, QCoreApplication.translate("Form", u"9_chair", None))
        self.comboBox.setItemText(9, QCoreApplication.translate("Form", u"10_radio", None))

        self.toolButton_2.setText(QCoreApplication.translate("Form", u"...", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Upload BackGround", None))
    # retranslateUi

