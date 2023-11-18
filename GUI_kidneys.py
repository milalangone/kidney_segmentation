from PyQt5.QtWidgets import QApplication, QHBoxLayout, QRubberBand, QSplitter, QVBoxLayout, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsRectItem, QMainWindow
from PyQt5.QtCore import QByteArray, QEvent, QRect, Qt, QRectF
import numpy as np
import cv2


class KidneyClassifierApp(QSplitter):
    def initUI(self):
        
        # Create widgets for the left half
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.upload_button = QPushButton("Upload Image", left_widget)
        self.image_label = QLabel(left_widget)
        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.image_label)

        # Create widgets for the right half
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.title_label = QLabel("Image Processing", right_widget)
        self.title_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.title_label)

        # Create a horizontal layout for classify buttons
        classify_layout = QHBoxLayout()

        self.classify_button1 = QPushButton("Classify Image 1", right_widget)
        self.classify_button2 = QPushButton("Classify Image 2", right_widget)
        classify_layout.addWidget(self.classify_button1)
        classify_layout.addWidget(self.classify_button2)
        classify_layout.setAlignment(Qt.AlignTop)  # Align buttons to the top

        right_layout.addLayout(classify_layout)

        # Set font for buttons
        font = self.upload_button.font()
        font.setPointSize(12)  # Adjust the font size as needed
        self.upload_button.setFont(font)
        self.process_button.setFont(font)
        self.classify_button1.setFont(font)
        self.classify_button2.setFont(font)

        self.rect1_label = QLabel(right_widget)
        self.rect2_label = QLabel(right_widget)
        right_layout.addWidget(self.rect1_label)
        right_layout.addWidget(self.rect2_label)

        # Add the left and right widgets to the splitter
        self.addWidget(left_widget)
        self.addWidget(right_widget)

        # Connect button signals to functions
        self.upload_button.clicked.connect(self.upload_image)
        self.process_button.clicked.connect(self.process_image)
        self.classify_button1.clicked.connect(self.classify_image)
        self.classify_button2.clicked.connect(self.classify_image)

    

    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
            options=options,
        )

        if file_name:
            self.img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            self.display_image(file_name)     


    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio))
        self.selected_rectangles = []
        self.create_rectangle_selector(self.image_label)
        

    
    def process_selected_rectangle(self, top_left_x,top_left_y, bottom_right_x,bottom_right_y):
        self.imgs = []
        if len(self.selected_rectangles) < 2 : 
            self.selected_rectangles.append(self.selected_rect)
            height, width = bottom_right_y - top_left_y, bottom_right_x - top_left_x
            self.new_image = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    # Swap indices for robustness
                    self.new_image[i, j] = self.img[top_left_y + i, top_left_x + j]
            self.imgs.append(self.new_image)
            self.update_image_label()


    def update_image_label(self):
        # Create a copy of the original image
        original_pixmap = self.image_label.pixmap()

        # Create a new painter to draw the selected rectangles on the image
        painter = QPainter(original_pixmap)

        # Draw each selected rectangle on the image
        for selected_rect in self.selected_rectangles:
            painter.drawRect(selected_rect)

        # Update the image label with the modified image
        self.image_label.setPixmap(original_pixmap)

        qimage_rect1 = QImage(self.imgs[0], self.imgs[0].shape[1], self.imgs[0].shape[0],QImage.Format_Grayscale8)                                                                                                                                                                 
        pixmap_rect1 = QPixmap(qimage_rect1)                                                                                                                                                                               
        pixmap_rect1 = pixmap_rect1.scaled(640,400, Qt.KeepAspectRatio)                                                                                                                                                    
        self.rect1_label.setPixmap(pixmap_rect1)

        if len(self.imgs) == 2:
            qimage_rect2 = QImage(self.imgs[1], self.imgs[1].shape[1], self.imgs[1].shape[0], QImage.Format_Grayscale8)                                                                                                                                                                 
            pixmap_rect2 = QPixmap(qimage_rect2)                                                                                                                                                                               
            pixmap_rect2 = pixmap_rect2.scaled(640, 400, Qt.KeepAspectRatio)   
            self.rect2_label.setPixmap(pixmap_rect2)

    def create_rectangle_selector(self, image_view):
        def mousePressEvent(event):
            if event.button() == Qt.MouseButton.LeftButton:
                self.start_pos = event.pos()
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, image_view)
                self.rubber_band.show()
                self.image_label.update()

        def mouseMoveEvent(event):
            if self.rubber_band:
                self.rubber_band.setGeometry(
                    QRect(self.start_pos, event.pos()).normalized()
                )
                self.image_label.update()

        def mouseReleaseEvent(event):
            if self.rubber_band:
                self.rubber_band.hide()
                self.rubber_band.deleteLater()
                self.rubber_band = None
                self.image_label.update()

                # Get the selected rectangle
                self.selected_rect = QRectF(self.start_pos, event.pos()).normalized()
                top_left_x = int(self.selected_rect.x())
                top_left_y = int(self.selected_rect.y())
                bottom_right_x = int(self.selected_rect.right())
                bottom_right_y = int(self.selected_rect.bottom())

                # Process the selected rectangle
                self.process_selected_rectangle(top_left_x,top_left_y, bottom_right_x,bottom_right_y)
                

        image_view.mousePressEvent = mousePressEvent
        image_view.mouseMoveEvent = mouseMoveEvent
        image_view.mouseReleaseEvent = mouseReleaseEvent

        
    def process_image(self):
        # Implement image processing logic here
        pass

    def classify_image(self):
        # Implement image classification logic here
        pass

if __name__ == "__main__":
    app = QApplication([])
    window = KidneyClassifierApp()
    window.show()
    app.exec_()
