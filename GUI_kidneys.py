from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSplitter, QVBoxLayout, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap

class KidneyClassifierApp(QSplitter):
    def __init__(self):
        super().__init__(Qt.Horizontal)
        self.initUI()

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
        self.process_button = QPushButton("Process Image", right_widget)
        self.classify_button = QPushButton("Classify Image", right_widget)
        right_layout.addWidget(self.process_button)
        right_layout.addWidget(self.classify_button)

        # Add the left and right widgets to the splitter
        self.addWidget(left_widget)
        self.addWidget(right_widget)

        # Connect button signals to functions
        ##hola
        self.upload_button.clicked.connect(self.upload_image)
        self.process_button.clicked.connect(self.process_image)
        self.classify_button.clicked.connect(self.classify_image)

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
            self.display_image(file_name)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio))

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
