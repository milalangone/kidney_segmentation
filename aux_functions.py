import cv2
import numpy as np
import os
from skimage import feature
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.transform import integral_image
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def extract_intensity_features(image):
    intensity_features = {}
    intensity_features['mean_intensity'] = np.mean(image)
    intensity_features['max_value'] = np.max(image)
    intensity_features['std_intensity'] = np.std(image)
    intensity_features['skewness'] = skew(image.flatten())
    intensity_features['kurtosis'] = kurtosis(image.flatten())
    return intensity_features

def extract_texture_features(image):
    texture_features = {}

    glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)
    texture_features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    texture_features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    texture_features['energy'] = graycoprops(glcm, 'energy')[0, 0]
    texture_features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    texture_features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]

    return texture_features

def extract_features(image):
    features = {}

    intensity_features = extract_intensity_features(image)
    texture_features = extract_texture_features(image)

    features.update(intensity_features)
    features.update(texture_features)

    return features

def create_dataframe(path, label):

    data = []
    # img_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

    for img in path:
        # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        features = extract_features(img)
        features['label'] = label

        data.append(features)

    df = pd.DataFrame(data)

    return df

def gaussian(img_path, kernel_size, sigma):
  filtered = []
  all_item_dirs = os.listdir(img_path)
  item_files = [os.path.join(img_path, file) for file in all_item_dirs]

  for idx, img_path in enumerate(item_files):
    image = cv2.imread(img_path, 0)
    filt_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    filtered.append(filt_img)
  return filtered


def segmentar(img, select):
  img_new = np.zeros((len(img), len(img[0])))

  # select = [0, 1, 2] = [otsu, km, watershed]
  if select == 0:
      _, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      return img_otsu
  elif select == 1:
      # Convert image to float32 and flatten
      z = img.astype(np.float32).reshape((-1, 1))

      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      flags = cv2.KMEANS_RANDOM_CENTERS

      compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)
      center = np.uint8(centers)
      img_kmeans = center[labels.flatten()]
      img_kmeans = img_kmeans.reshape((img.shape))
      return img_kmeans


def bin2gray(img_seg, img_og):

  img_new = np.zeros((len(img_seg), len(img_seg[0])))
  for i in range(len(img_seg)):
    for j in range(len(img_seg[0])):
      if img_seg[i][j] != 0:
        img_new[i][j] = img_og[i][j]
  return img_new.astype(np.uint8)

def clicker_seg(img_r1, img_r2, select):
  img_seg1 = segmentar(img_r1, select)
  img_seg2 = segmentar(img_r2, select)
  img_gray1 = bin2gray(img_r1)
  img_gray2 = bin2gray(img_r2)
  return img_gray1, img_gray2

def new_data_point(img):

  data = []

  # Segment image
  img_seg_kmeans = segmentar(img, 1)
  seg_gray = bin2gray(img_seg_kmeans, img)

  # Extract image features
  features = extract_features(img)
  data.append(features)

  # Create dataframe
  x_new = pd.DataFrame(data)

  return x_new

def predict_probabilities(img, trained_model, classes):

  x = new_data_point(img)
  test_label_probabilities = trained_model.predict_proba(x)

  return test_label_probabilities
    