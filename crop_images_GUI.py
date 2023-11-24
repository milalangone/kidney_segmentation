import cv2
import numpy as np
import os, sys
from subprocess import check_output

''' CONSTANTS '''
LABEL = 11 # 1 - 9 reserved for DICE, 10 is old orange gate, 11 is new black and red
SQUARE = True

''' GLOBAL VARIABLES ''' 
drawing = False # true if mouse is pressed
ix, iy = -1, -1
roi = (0, 0, 0, 0) # null value - handles numerical calcs
img = None


# normalizes the x, y, w, h coords when dragged from different directions
def get_roi(x, y, w, h):
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
def roi_to_square(x, y, w, h):
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
def get_file_names_from_dir(file_path):
    ls = []
    with os.scandir(file_path) as it:
        for entry in it:
            if not entry.name.startswith(".") and entry.is_file:
                ls.append(entry.name)
        return sorted(ls)


# creates a txt file to append the label and ROI coordinates
# returns the file object
def create_txt_file(path):
    try:
        txt_file = open(path, "a+")
    except:
        print("File I/O error")
        exit()
    return txt_file
    

# mouse callback function for drawing boxes
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi
    
    if (event == cv2.EVENT_LBUTTONDOWN):
        drawing = True
        ix, iy = x, y
        
    elif (event == cv2.EVENT_MOUSEMOVE):
        if drawing == True:
            #tmp_img = param.copy()
            tmp_img = img.copy()
            cv2.rectangle(tmp_img, (ix, iy), (x, y), (0, 255, 0), 1)
            cv2.imshow("image", tmp_img)
            
    elif (event == cv2.EVENT_LBUTTONUP):
        #tmp_img = param.copy()
        tmp_img = img.copy()
        drawing = False
        roi = (ix, iy, x, y)
        cv2.rectangle(tmp_img, (ix, iy), (x, y), (0, 255, 0), 1)


if __name__ == "__main__":

    clear_screen_cmd = os.system('cls' if os.name == 'nt' else 'clear')
    print(clear_screen_cmd) # clears terminal (ctrl + l)

    print(__doc__)

    # counters
    global_roi_counter = 0
    local_roi_counter = 0
    img_counter = 0

    # dir stuff
    labels_dir_path = "C:\\Users\\milal\\Downloads\\pib_recortado"  # where to store label txt files
    imgs_dir_path = "C:\\Users\\milal\\Downloads\\pib\\Cyst filtrado"  # where to grab images from
    rois_dir_path = "C:\\Users\\milal\\Downloads\\pib_recortado"  # where to store roi
    
    file_names_with_ext = get_file_names_from_dir(imgs_dir_path)
    file_names = [x.split(".")[0] for x in file_names_with_ext] # gets name only - discards extension
    number_of_imgs = len(file_names)

    # image stuff
    img = cv2.imread( imgs_dir_path + "\\" + file_names[img_counter] + ".jpg" )
    img_l, img_w, ch = img.shape
    tmp_img = img.copy()

    # window/screen stuff
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)
    cv2.imshow("image", img)

    while(1):
        
        k = cv2.waitKey(1) & 0xFF
        
        if (k == 27 or k == ord("q") ): # exit
            print("\n\tQ was pressed - Quitting\n")
            global_roi_counter += local_roi_counter
            print(str(global_roi_counter) + " ROI(s) were stored!\n")
            break
        
        elif (k == ord("f") and (img_counter < number_of_imgs - 1) ): # move forward to next image
            print("\n\tNext Image")
            roi = (0, 0, 0, 0)
            cv2.destroyWindow("roi")
            img_counter += 1 # for file name purposes
            global_roi_counter += local_roi_counter
            local_roi_counter = 0
            img_path = "./images/" + file_names[img_counter] + ".jpg"
            img = cv2.imread(img_path)
            tmp_img = img.copy()
            cv2.imshow("image", tmp_img)
            
        elif (k == ord("d") and (img_counter > 0) ): # move back to prev image
            print("\n\tPrevious Image")
            roi = (0, 0, 0, 0)
            cv2.destroyWindow("roi")
            img_counter -= 1 # for file name purposes
            global_roi_counter += local_roi_counter
            local_roi_counter = 0
            img_path = "./images/" + file_names[img_counter] + ".jpg"
            img = cv2.imread(img_path)
            tmp_img = img.copy()
            cv2.imshow("image", tmp_img)

        elif (k == ord("r") ): # refresh image (remove all markings on img)
            print("\n\tRefreshed Image")
            roi = (0, 0, 0, 0)
            cv2.destroyWindow("roi")
            tmp_img = img.copy()
            cv2.imshow("image", tmp_img)
            
        elif(k == ord("t") ): # toggle square resize
            SQUARE = not SQUARE
            print("\n\tSquare toggled:", SQUARE)
        
        if (roi != (0, 0, 0, 0) ): # if ROI has been created
            roi_x, roi_y, roi_w, roi_h = roi
            x, y, w, h = get_roi(roi_x, roi_y, roi_w, roi_h) # returns the ROI from original image
            
            if (SQUARE):
                x, y, w, h = roi_to_square(x, y, w, h) # squares the bounding box

            if ( (x, y, w, h) == (0, 0, 0, 0) ): # checks for bug when mouse is clicked but no box is created
                img_roi = None
            else:
                img_roi = img[y:h, x:w, :]

            if (img_roi is None): # bug check - need to identify small boxes
                print("\nERROR:\tROI " + str(roi) + " is Out-of-Bounds OR not large enough")
                cv2.destroyWindow("roi")
                roi = (0, 0, 0, 0) # might already be set
                tmp_img = img.copy()
                cv2.imshow("image", tmp_img)
                
            elif(k == ord("v") ): # view roi
                print("\n\tViewing ROI "  + str(roi) )
                print("\t\tShape", img_roi.shape)
                cv2.imshow("roi", img_roi)
                cv2.moveWindow("roi", img_w, 87)
                
            elif(k == ord("s") ): # save roi after viewing it
                local_roi_counter += 1
                path = rois_dir_path + file_names[img_counter] + "_" + str(local_roi_counter) + ".jpg"
                print("\n* Saved ROI #" + str(local_roi_counter) + " " + str(roi) + " to: " + path)
                cv2.imwrite(path, img_roi)
                
                txt_file_path = labels_dir_path + file_names[img_counter] + ".txt"
                txt_file = create_txt_file(txt_file_path)
                print(str(LABEL), x, y, w, h, file=txt_file)
                
                cv2.destroyWindow("roi")
                roi = (0, 0, 0, 0)
                tmp_img = img.copy()
                cv2.imshow("image", tmp_img)
                
            elif(k == ord("c") ): # clear roi
                print("\n\tCleared ROI " + str(roi) )
                roi = (0, 0, 0, 0)
                cv2.destroyWindow("roi")
                tmp_img = img.copy()
                cv2.imshow("image", tmp_img)

    cv2.destroyAllWindows()