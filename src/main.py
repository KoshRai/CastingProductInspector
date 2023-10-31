from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog 
import numpy as np
import os
import torch
from Classifier import Classifier
import cv2 as cv
import time
from threading import Thread
import pandas as pd
import time


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.setFixedSize(800, 600)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.img_pane = QtWidgets.QLabel(self.centralwidget)
        self.img_pane.setGeometry(QtCore.QRect(70, 190, 321, 281))
        self.img_pane.setObjectName("img_pane")
        self.img_pane.setText('Please select a directory')
        self.img_pane.setStyleSheet('font-size: 15px;')

        self.insp_button = QtWidgets.QPushButton(self.centralwidget)
        self.insp_button.setGeometry(QtCore.QRect(460, 320, 201, 51))
        self.insp_button.setObjectName("insp_button")
        self.insp_button.clicked.connect(self.inspect_helper)
        
        self.insp_label = QtWidgets.QLabel(self.centralwidget)
        self.insp_label.setGeometry(QtCore.QRect(460, 400, 251, 101))
        self.insp_label.setObjectName("insp_label")
        self.insp_label.setStyleSheet('font-size: 15px;')
        
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(80, 20, 625, 131))
        self.title.setObjectName("title")
        self.title.setStyleSheet("font-family: Lucida Console; font-size: 41px;")
        
        self.browse_button = QtWidgets.QPushButton(self.centralwidget)
        self.browse_button.setGeometry(QtCore.QRect(460, 250, 201, 51))
        self.browse_button.setObjectName("browse_button")
        self.browse_button.clicked.connect(self.browse)

        self.demonstrate = False
        self.demonstrate_button = QtWidgets.QPushButton(self.centralwidget)
        self.demonstrate_button.setGeometry(QtCore.QRect(600, 500, 110, 60))
        self.demonstrate_button.setObjectName("demonstrate_button")
        self.demonstrate_button.setStyleSheet("color: Red;")
        self.demonstrate_button.clicked.connect(self.toggle_demonstrate)

        MainWindow.setCentralWidget(self.centralwidget)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CastingProductInspector"))
        self.img_pane.setText(_translate("MainWindow", "Please select a directory"))
        self.insp_button.setText(_translate("MainWindow", "Start Inspection"))
        self.insp_label.setText(_translate("MainWindow", ""))
        self.title.setText(_translate("MainWindow", "Casting Product Inspector"))
        self.browse_button.setText(_translate("MainWindow", "Browse"))
        self.demonstrate_button.setText(_translate("MainWindow", "Demonstration mode"))

    def toggle_demonstrate(self):
        self.demonstrate = not self.demonstrate
        if self.demonstrate: self.demonstrate_button.setStyleSheet("color: Green;")
        else: self.demonstrate_button.setStyleSheet("color: Red;")

    def browse(self):
        try:
            self.path = str(QFileDialog.getExistingDirectory(None, "Select Directory"))
            self.imgs = np.array([img for img in os.listdir(self.path)])
            self.img_pane.setText(f'Selected directory\nReady to start inspection!')
        except:
            return
        

    def inspect_helper(self):
        t1=Thread(target=self.inspect)
        t1.start()

    def inspect(self):
        d = {'Image':[], 'Inspection result':[]}
        model = torch.load('model.pt', map_location=torch.device('cpu'))
        print('Loaded model')
        self.title.setText('  ~Inspection Underway~')
        try:
            np.random.shuffle(self.imgs)
        except AttributeError: 
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setText("Please select a directory first!")
            msg_box.setWindowTitle("No directory selected")
            msg_box.exec_()
            return
        defective_count, okay_count = 0, 0
        t = time.process_time()
        for idx, img_path in enumerate(self.imgs):
            try:
                d['Image'].append(img_path)
                img_path = os.path.join(self.path, img_path)
                img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
                print(f'{idx}: {img_path}')
                self.img_pane.setPixmap(QtGui.QPixmap(img_path))
                self.insp_label.setText('')
            except:
                d['Image'].pop()
                continue
            img = torch.tensor(img).type(torch.FloatTensor)
            img = torch.permute(img, (2,0,1))
            img = torch.unsqueeze(img, 0)
            img = img/255
            pred = torch.argmax(model(img)).item()
            if pred == 1:
                x = 'Defective'
                defective_count += 1
                d['Inspection result'].append(x)
            else:
                x = 'Okay'
                okay_count += 1
                d['Inspection result'].append(x)
            
            if self.demonstrate: 
                if x == 'Okay': self.insp_label.setStyleSheet('font-size: 20px; color: green')
                else: self.insp_label.setStyleSheet('font-size: 20px; color: red;')
                self.insp_label.setText(x)
                time.sleep(1)
            else: 
                self.insp_label.setText('')
        elapsed_time = time.process_time() - t
        self.img_pane.setText(f"Finished Inspection!\n\nDefective: {defective_count}\nOkay: {okay_count}\n\nDefect Rate: {defective_count/(defective_count+okay_count+1e-10):.2f}%\nAvg. Throughput {elapsed_time/len(d['Image']):.2f} items per second")
        self.insp_label.setText('')
        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.path, 'results.csv'), index=False)
        print('Wrote results.csv')
        self.title.setText('Casting Product Inspector')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())