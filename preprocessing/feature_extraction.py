# extract_features.py
import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from skimage.feature import graycomatrix, graycoprops

def extract_color_histogram(image, bins=(8, 4, 4)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_haralick_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1],
                        angles=[0], levels=256,
                        symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    return np.array(features)

def extract_features(dataset_dir, output_csv, le=None, save_encoder=False):
    features = []
    labels = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path))

                image = cv2.imread(path)
                image = cv2.resize(image, (128, 128))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                color_feat = extract_color_histogram(image)
                haralick_feat = extract_haralick_features(gray)
                combined = np.concatenate((color_feat, haralick_feat))

                features.append(combined)
                labels.append(label)

    if le is None:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
    else:
        encoded_labels = le.transform(labels)

    df = pd.DataFrame(features)
    df['label'] = encoded_labels
    df.to_csv(output_csv, index=False)
    print(f"✅ Fitur disimpan di {output_csv}")

    if save_encoder:
        joblib.dump(le, 'label_encoder.pkl')
        print("💾 LabelEncoder disimpan di 'label_encoder.pkl'")

    return le

#output directory
output_dir = "extracted_features/knn"
# Jalankan untuk semua set
le = extract_features("augmented_knn/train", os.path.join(output_dir, "train_features.csv"), save_encoder=True)
extract_features("augmented_knn/val", os.path.join(output_dir,"val_features.csv"), le=le)
extract_features("augmented_knn/test", os.path.join(output_dir,"test_features.csv"), le=le)
