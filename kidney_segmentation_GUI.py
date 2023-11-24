import sys
import numpy as np
import os
import joblib
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QMainWindow, QGraphicsScene, QGraphicsView, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QImageReader
from PyQt5.QtCore import Qt, QRectF
import cv2
import aux_functions

class CTScanApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.roi_start = None
        self.roi_end = None
        self.dt = joblib.load('pretrained_decision_tree_model.joblib')

    def initUI(self):
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.upload_button = QPushButton('Upload CT Scan', self)
        self.upload_button.clicked.connect(self.upload_scan)

        self.otsu_button = QPushButton('Segment (Otsu)', self)
        self.otsu_button.clicked.connect(lambda: self.segment_otsu(self.img_segm))

        self.kmeans_button = QPushButton('Segment (K-Means)', self)
        self.kmeans_button.clicked.connect(lambda: self.segment_kmeans(self.img_segm))

        self.classify_button = QPushButton('Classify', self)
        self.classify_button.clicked.connect(lambda: self.classify(self.current_image))

        self.result_label = QLabel('Classifier Results:', self)

        # Create a central widget to hold the buttons and result label
        central_widget = QWidget(self)
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(self.upload_button)
        central_layout.addWidget(self.otsu_button)
        central_layout.addWidget(self.kmeans_button)
        central_layout.addWidget(self.classify_button)
        central_layout.addWidget(self.result_label)
        central_layout.addWidget(self.view)

        self.setCentralWidget(central_widget)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('CT Scan Analysis')
        self.show()

    def upload_scan(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CT Scan Image", "", "Images (*.png *.jpg *.bmp *.tif)", options=options)
        if file_name:
            self.scene.clear()
            pixmap = QPixmap(file_name)
            self.scene.addPixmap(pixmap)
            self.current_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # Store the current image as a NumPy array
            self.view.setScene(self.scene)

    
    def segment_otsu(self, img):
        img_otsu = aux_functions.segmentar(img, 0)
        self.img_segm = aux_functions.bin2gray(img_otsu, img)
        
        # Convert the segmented NumPy array to QPixmap and display it
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene.addItem(segmented_item)
        
        return self.img_segm

    def segment_kmeans(self, img):
        img_kmeans = aux_functions.segmentar(img,1)
        self.img_segm = aux_functions.bin2gray(img_kmeans, img)
        
        # Convert the segmented NumPy array to QPixmap and display it
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene.addItem(segmented_item)
        
        return self.img_segm
    
    def ndarray_to_pixmap(self, image_array):
        # Convert a NumPy array to QPixmap
        height, width = image_array.shape[:2]
        qimage = QImage(image_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def classify(self, img):
        # Implement the code to classify each kidney into categories.
        # Example: Show some dummy results (replace with actual classification results)
        classif_labels = ['Cyst', 'Normal', 'Stones', 'Tumor']
        aux_functions.new_data_point(img)
        classification = aux_functions.predict_probabilities(img, self.dt, classif_labels)
        
        result_text = f', '.join([f'{classif}: {round(classification[0][i]*100,2)}%' for i, classif in enumerate(classif_labels)])
        self.result_label.setText(f'Classifier Results: {result_text}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CTScanApp()
    sys.exit(app.exec_())
