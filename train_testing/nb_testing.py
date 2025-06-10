# test_model_direct_image.py

import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops

# --- Feature Extraction Functions ---
def extract_color_histogram(image, bins=(8, 4, 4)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_haralick_features(gray):
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ])

# --- Load Test Images & Extract Features ---
def load_and_extract_features_from_images(root_dir, label_encoder):
    features = []
    labels = []

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, filename)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (128, 128))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                color_feat = extract_color_histogram(image)
                haralick_feat = extract_haralick_features(gray)
                combined = np.concatenate((color_feat, haralick_feat))

                features.append(combined)
                labels.append(class_name)

    features = np.array(features)
    labels_encoded = label_encoder.transform(labels)

    return features, labels_encoded

# --- Evaluation Function ---
def evaluate_model(name, model, X_test, y_test, label_encoder):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n📊 Model: {name}")
    print(f"Akurasi: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    test_dir = "augmented_nb/test"
    model_path = "model/nb/naive_bayes_model.pkl"
    pca_path = "model/nb/pca_nb_transform.pkl"
    encoder_path = "extracted_features/nb/label_encoder.pkl"

    # Load model components
    nb_model = joblib.load(model_path)
    nb_pca = joblib.load(pca_path)
    label_encoder = joblib.load(encoder_path)

    # Load test images and extract features
    X_test_raw, y_test = load_and_extract_features_from_images(test_dir, label_encoder)

    # Apply PCA
    X_test = nb_pca.transform(X_test_raw)

    # Evaluate model
    evaluate_model("Naive Bayes (Direct Image Test)", nb_model, X_test, y_test, label_encoder)