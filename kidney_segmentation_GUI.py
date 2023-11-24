import sys
import numpy as np
import os
import joblib
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsRectItem, QHBoxLayout, QMainWindow, QGraphicsScene, QGraphicsView, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF
import cv2
import aux_functions

class CTScanApp(QMainWindow):
    def __init__(self):
        
        super().__init__()
        self.initUI()
        self.dt = joblib.load('pretrained_decision_tree_model.joblib')

    def initUI(self):
        
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)


        # Create a horizontal layout for the buttons
        self.button_layout = QHBoxLayout()

        self.upload_button1 = QPushButton('Upload Kidney 1', self)
        self.upload_button1.clicked.connect(lambda: self.upload_scan(1))
        self.button_layout.addWidget(self.upload_button1)

        self.upload_button2 = QPushButton('Upload Kidney 2', self)
        self.upload_button2.clicked.connect(lambda: self.upload_scan(2))
        self.button_layout.addWidget(self.upload_button2)

        self.otsu_button = QPushButton('Segment (Otsu)', self)
        self.otsu_button.clicked.connect(lambda: self.segment_otsu(self.image1, self.image2))
        self.button_layout.addWidget(self.otsu_button)

        self.kmeans_button = QPushButton('Segment (K-Means)', self)
        self.kmeans_button.clicked.connect(lambda: self.segment_kmeans(self.image1, self.image2))
        self.button_layout.addWidget(self.kmeans_button)

        self.classify_button = QPushButton('Classify', self)
        self.classify_button.clicked.connect(lambda: self.classify(self.img_segm1, self.img_segm2))
        self.button_layout.addWidget(self.classify_button)

        
        # Add the button layout to the main layout
        self.layout.addLayout(self.button_layout)
        
        # Create a horizontal layout for the views and upload buttons
        self.horizontal_layout = QHBoxLayout()

        self.scene1 = QGraphicsScene(self)
        self.view1 = QGraphicsView(self.scene1)
        self.horizontal_layout.addWidget(self.view1)

        self.scene2 = QGraphicsScene(self)
        self.view2 = QGraphicsView(self.scene2)
        self.horizontal_layout.addWidget(self.view2)

        self.layout.addLayout(self.horizontal_layout)
        
        self.result_label1 = QLabel('Results image 1:', self)
        self.layout.addWidget(self.result_label1)

        self.result_label2 = QLabel('Results image 2:', self)
        self.layout.addWidget(self.result_label2)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('CT Scan Analysis')
        self.show()

    def upload_scan(self, image_number):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, f"Open CT Scan Image {image_number}", "", "Images (*.png *.jpg *.bmp *.tif)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            current_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

            if image_number == 1:
                self.image1 = current_image
                self.scene1.addPixmap(pixmap)
            else:
                self.image2 = current_image
                self.scene2.addPixmap(pixmap)
        
    def segment_otsu(self, img1, img2):
        img_otsu1 = aux_functions.segmentar(img1, 0)
        self.img_segm1 = aux_functions.bin2gray(img_otsu1, img1)
        
        img_otsu2 = aux_functions.segmentar(img2, 0)
        self.img_segm2 = aux_functions.bin2gray(img_otsu2, img2)

        # Convert the segmented NumPy array to QPixmap and display it
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm1)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene1.addItem(segmented_item)
        
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm2)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene2.addItem(segmented_item)

        return self.img_segm1, self.img_segm2

    def segment_kmeans(self, img1, img2):
        img_kmeans1 = aux_functions.segmentar(img1,1)
        self.img_segm1 = aux_functions.bin2gray(img_kmeans1, img1)
        
        img_kmeans2 = aux_functions.segmentar(img2,1)
        self.img_segm2 = aux_functions.bin2gray(img_kmeans2, img2)

        # Convert the segmented NumPy array to QPixmap and display it
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm1)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene1.addItem(segmented_item)
        
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm2)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene2.addItem(segmented_item)

        return self.img_segm1, self.img_segm2

    def ndarray_to_pixmap(self, image_array):
        # Convert a NumPy array to QPixmap
        height, width = image_array.shape[:2]
        qimage = QImage(image_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def classify(self, img1, img2):
        classif_labels = ['Cyst', 'Normal', 'Stones', 'Tumor']
        
        aux_functions.new_data_point(img1)
        classification1 = aux_functions.predict_probabilities(img1, self.dt, classif_labels)
        
        aux_functions.new_data_point(img2)
        classification2 = aux_functions.predict_probabilities(img2, self.dt, classif_labels)

        result_text = f', '.join([f'{classif}: {round(classification1[0][i]*100,2)}%' for i, classif in enumerate(classif_labels)])
        self.result_label1.setText(f'Classifier Results: {result_text}')
        
        result_text2 = f', '.join([f'{classif}: {round(classification2[0][i]*100,2)}%' for i, classif in enumerate(classif_labels)])
        self.result_label2.setText(f'Classifier Results: {result_text2}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CTScanApp()
    sys.exit(app.exec_())
