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
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.roi = (0, 0, 0, 0)
        self.img = None

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
        
        self.result_label1 = QLabel('Left Kidney Results:', self)
        self.layout.addWidget(self.result_label1)

        self.result_label2 = QLabel('Right Kidney Results: ', self)
        self.layout.addWidget(self.result_label2)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('CT Scan Analysis')
        self.show()

    def upload_scan(self, image_number):
        
        ''' CONSTANTS '''
        LABEL = 11 # 1 - 9 reserved for DICE, 10 is old orange gate, 11 is new black and red
        SQUARE = False

        ''' GLOBAL VARIABLES ''' 
        drawing = False # true if mouse is pressed

        
        clear_screen_cmd = os.system('cls' if os.name == 'nt' else 'clear')
        print(clear_screen_cmd) # clears terminal (ctrl + l)

        # counters
        global_roi_counter = 0
        local_roi_counter = 0
        img_counter = 0

        # dir stuff
        labels_dir_path = "C:/Users/lusim/Downloads/cut imgs" # where to store label txt files
        imgs_dir_path = "C:/Users/lusim/Downloads/sample imgs"  # where to grab images from
        rois_dir_path = "C:/Users/lusim/Downloads/cut imgs"  # where to store roi
        
        file_names_with_ext = self.get_file_names_from_dir(imgs_dir_path)
        file_names = [x.split(".")[0] for x in file_names_with_ext] # gets name only - discards extension
        number_of_imgs = len(file_names)

        # image stuff
        self.img = cv2.imread( imgs_dir_path + "/" + file_names[img_counter] + ".jpg" )
        img_l, img_w, ch = self.img.shape
        tmp_img = self.img.copy()

        # window/screen stuff
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.draw_rectangle)
        cv2.imshow("image", self.img)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"): # exit
                print("\n\tQ was pressed - Quitting\n")
                global_roi_counter += local_roi_counter
                print(str(global_roi_counter) + " ROI(s) were stored!\n")
                break
            
                
            elif(k == ord("t") ): # toggle square resize
                SQUARE = not SQUARE
                print("\n\tSquare toggled:", SQUARE)
            
            if (self.roi != (0, 0, 0, 0) ): # if ROI has been created
                roi_x, roi_y, roi_w, roi_h = self.roi
                x, y, w, h = self.get_roi(roi_x, roi_y, roi_w, roi_h) # returns the ROI from original image
                
                if (SQUARE):
                    x, y, w, h = self.roi_to_square(self.img, x, y, w, h) # squares the bounding box

                if ( (x, y, w, h) == (0, 0, 0, 0) ): # checks for bug when mouse is clicked but no box is created
                    img_roi = None
                else:
                    img_roi = self.img[y:h, x:w, :]

                if (img_roi is None): # bug check - need to identify small boxes
                    print("\nERROR:\tROI " + str(roi) + " is Out-of-Bounds OR not large enough")
                    # cv2.destroyWindow("roi")
                    self.roi = (0, 0, 0, 0) # might already be set
                    tmp_img = self.img.copy()
                    cv2.imshow("image", tmp_img)
                    
                elif(k == ord("v") ): # view roi
                    print("\n\tViewing ROI "  + str(self.roi) )
                    print("\t\tShape", img_roi.shape)
                    cv2.imshow("roi", img_roi)
                    cv2.moveWindow("roi", img_w, 87)
                    
                elif(k == ord("s") ): # save roi after viewing it
                    local_roi_counter += 1
                    path = rois_dir_path +'/' +file_names[img_counter] + "_" + str(local_roi_counter) + ".jpg"
                    print("\n* Saved ROI #" + str(local_roi_counter) + " " + str(self.roi) + " to: " + path)
                    
                    img_roi_resized = cv2.resize(img_roi, (img_roi.shape[1]*2, img_roi.shape[0]*2))
                    cv2.imwrite(path, img_roi_resized)
                    
                    txt_file_path = labels_dir_path + file_names[img_counter] + ".txt"
                    txt_file = self.create_txt_file(txt_file_path)
                    print(str(LABEL), x, y, w, h, file=txt_file)

                    
                    # cv2.destroyWindow("self.roi")
                    self.roi = (0, 0, 0, 0)
                    # tmp_img = self.img.copy()
                    # cv2.imshow("image", tmp_img)
                    
                    pixmap = QPixmap(path)
                    
                    
                    if image_number == 1:
                        current_roi = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        self.image1 = current_roi
                        self.scene1.addPixmap(pixmap)
                    else:
                        current_roi = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        self.image2 = current_roi
                        self.scene2.addPixmap(pixmap)
                    
                elif(k == ord("c") ): # clear roi
                    print("\n\tCleared ROI " + str(self.roi) )
                    roi = (0, 0, 0, 0)
                    cv2.destroyWindow("self.roi")
                    tmp_img = self.img.copy()
                    cv2.imshow("image", tmp_img)

   
            
        
    def segment_otsu(self, img1, img2):
        img_otsu1 = aux_functions.segmentar(img1, 0)
        self.img_segm1 = aux_functions.bin2gray(img_otsu1, img1)
        
        img_otsu2 = aux_functions.segmentar(img2, 0)
        self.img_segm2 = aux_functions.bin2gray(img_otsu2, img2)
        
        self.scene1.clear()
        self.scene2.clear()

        # Convert the segmented NumPy array to QPixmap and display it
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm1)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene1.addItem(segmented_item)
        
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm2)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene2.addItem(segmented_item)
        
        self.view1.update()
        self.view2.update()

        return self.img_segm1, self.img_segm2

    def segment_kmeans(self, img1, img2):
        img_kmeans1 = aux_functions.segmentar(img1,1)
        self.img_segm1 = aux_functions.bin2gray(img_kmeans1, img1)
        
        img_kmeans2 = aux_functions.segmentar(img2,1)
        self.img_segm2 = aux_functions.bin2gray(img_kmeans2, img2)
        
        self.scene1.clear()
        self.scene2.clear()

        # Convert the segmented NumPy array to QPixmap and display it
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm1)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene1.addItem(segmented_item)
        
        segmented_pixmap = self.ndarray_to_pixmap(self.img_segm2)
        segmented_item = QGraphicsPixmapItem(segmented_pixmap)
        segmented_item.setPos(0, 0)
        self.scene2.addItem(segmented_item)
        
        self.view1.update()
        self.view2.update()

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
        classification1 = aux_functions.predict_probabilities(img1, self.dt)
        
        result_text = f', '.join([f'{classif}: {round(classification1[0][i]*100,2)}%' for i, classif in enumerate(classif_labels)])
        self.result_label1.setText(f'Left Kidney Results: {result_text}')
        
        aux_functions.new_data_point(img2)
        classification2 = aux_functions.predict_probabilities(img2, self.dt)
        result_text2 = f', '.join([f'{classif}: {round(classification2[0][i]*100,2)}%' for i, classif in enumerate(classif_labels)])
        self.result_label2.setText(f'Right Kidney Results: {result_text2}')
    


    # normalizes the x, y, w, h coords when dragged from different directions
    def get_roi(self, x, y, w, h):
        if (y > h and x > w): # lower right to upper left
            return (w, h, x, y)
        elif (y < h and x > w): # upper right to lower left
            return (w, y, x, h)
        elif (y > h and x < w): # lower left to upper right
            return(x, h, w, y)
        elif (y == h and x == w):
            return (0, 0, 0, 0) # roi too small
        else:
            return (x, y, w, h) # upper left to lower right


    # reshape an ROI to a square image
    # tuple in, tuple out
    def roi_to_square(self, img, x, y, w, h):
        frame_y, frame_x, ch = img.shape # since img is global var
        x_shape = w - x # size of x ROI
        y_shape = h - y # size of y ROI
        
        if(x_shape > y_shape):
            y_diff = (x_shape - y_shape) // 2
            if( (y - y_diff) < 0):
                return (x, y, w, h + (y_diff * 2) ) # add only to bottom
            elif( (h + y_diff) > frame_y):
                return (x, y - (y_diff * 2), w, h) # add only to top
            else:
                return (x, y - y_diff, w, h + y_diff) # add to both top and bottom
        elif(y_shape > x_shape):
            x_diff = (y_shape - x_shape) // 2
            if( (x - x_diff) < 0):
                return (x, y, w + (x_diff * 2), h) # add only to right side
            elif( (w + x_diff) > frame_x):
                return (x - (x_diff * 2), y, w, h) # add only to left side
            else:
                return (x - x_diff, y, w + x_diff, h) # add to left and right side
        else:
            return (x, y, w, h) # already square
        

    # gets file names from a dir - includes file extension
    # returns as a list
    def get_file_names_from_dir(self, file_path):
        ls = []
        with os.scandir(file_path) as it:
            for entry in it:
                if not entry.name.startswith(".") and entry.is_file:
                    ls.append(entry.name)
            return sorted(ls)


    # creates a txt file to append the label and ROI coordinates
    # returns the file object
    def create_txt_file(self, path):
        try:
            txt_file = open(path, "a+")
        except:
            print("File I/O error")
            exit()
        return txt_file
        

    # mouse callback function for drawing boxes
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                tmp_img = self.img.copy()
                cv2.rectangle(tmp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 1)
                cv2.imshow("image", tmp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            tmp_img = self.img.copy()
            self.drawing = False
            self.roi = (min(self.ix, x), min(self.iy, y), max(self.ix, x), max(self.iy, y))
            cv2.rectangle(tmp_img, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (0, 255, 0), 1)
            cv2.imshow("image", tmp_img)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CTScanApp()
    sys.exit(app.exec_())
