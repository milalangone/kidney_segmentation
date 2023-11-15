import cv2
import numpy as np
import os
from skimage import feature
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import regionprops
from skimage.transform import integral_image
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq

def extract_intensity_features(image):
    intensity_features = {}
    intensity_features['mean_intensity'] = np.mean(image)
    intensity_features['std_intensity'] = np.std(image)
    intensity_features['skewness'] = skew(image.flatten())
    intensity_features['kurtosis'] = kurtosis(image.flatten())
    intensity_features['histogram'] = itemfreq(image.flatten())[:, 1]
    return intensity_features

def extract_texture_features(image):
    texture_features = {}
    
    glcm = greycomatrix(image, [1], [0], symmetric=True, normed=True)
    texture_features['contrast'] = greycoprops(glcm, 'contrast')[0, 0]
    texture_features['homogeneity'] = greycoprops(glcm, 'homogeneity')[0, 0]
    texture_features['energy'] = greycoprops(glcm, 'energy')[0, 0]
    texture_features['correlation'] = greycoprops(glcm, 'correlation')[0, 0]
    texture_features['entropy'] = greycoprops(glcm, 'entropy')[0, 0]

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
    img_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    
    for img in img_files:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
        features = extract_features(img)
        features['label'] = label
        
        data.append(features)
    
    df = pd.DataFrame(data)
    
    return df