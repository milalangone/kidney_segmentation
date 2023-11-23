import cv2
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
import aux_functions
from skimage import feature
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.transform import integral_image
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

path_tumor = "C:/Users/lusim/Downloads/kidneys-20231123T204336Z-001/kidneys/tumor"
path_cyst = "C:/Users/lusim/Downloads/kidneys-20231123T204336Z-001/kidneys/cysts"
path_stones = "C:/Users/lusim/Downloads/kidneys-20231123T204336Z-001/kidneys/stones"
path_normal = "C:/Users/lusim/Downloads/kidneys-20231123T204336Z-001/kidneys/normal"

paths = [path_tumor, path_cyst, path_stones, path_normal]

segmented_paths = []

for path in paths:
    filtered = aux_functions.gaussian(path,5,1)
    segmented_imgs = []
    for img in filtered:
        img_seg_kmeans = aux_functions.segmentar(img, 1)
        seg_gray = aux_functions.bin2gray(img_seg_kmeans, img)
        segmented_imgs.append(seg_gray)
    segmented_paths.append(segmented_imgs)
    

dfs = []
labels = ['Tumor', 'Cyst', 'Stones', 'Normal']


for path, label in zip(segmented_paths, labels):
    df_by_label = aux_functions.create_dataframe(path, label)
    dfs.append(df_by_label)

df = pd.concat(dfs, ignore_index = True)

X = df.drop(['label'], axis = 1)
y = df['label']

# Encode categorical labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

random_state = 13 # para asegurar reproducibilidad de los resultados
ts = 0.3 # test size, el estandar es 30% de la base de datos

xtrain, xtest, ytrain, ytest = train_test_split(X, y_encoded, random_state=random_state, test_size=ts)
print(f'Training on {ytrain.size} examples')

######## TRAINED MODEL
dt = DecisionTreeClassifier(max_depth = 3)
dt.fit(xtrain, ytrain)

# Predict probabilities on the test set
y_prob_test = dt.predict_proba(xtest)

# Convert probabilities to predicted labels
y_pred_test = dt.predict(xtest)