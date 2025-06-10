import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder

# HSV Segmentation
def segment_leaf(image):
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return cv2.bitwise_and(image, image, mask=mask)

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

# Main View
def index(request):
    return render(request, 'classifier/index.html')

def predict(request):
    if request.method == 'POST':
        image_file = request.FILES['image']
        algorithm = request.POST['algorithm']

        # Load image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = segment_leaf(image)
        image = cv2.resize(image, (128, 128))

        if algorithm in 'knn':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color_feat = extract_color_histogram(image)
            texture_feat = extract_haralick_features(gray)
            combined = np.concatenate((color_feat, texture_feat)).reshape(1, -1)

            pca = joblib.load(os.path.join(settings.MODELS_DATA_PATH, f"pca_{algorithm}_transform.pkl"))
            reduced = pca.transform(combined)

            model = joblib.load(os.path.join(settings.MODELS_DATA_PATH, f"{algorithm}_model.pkl"))
            le = joblib.load(os.path.join(settings.MODELS_DATA_PATH, f"{algorithm}_label_encoder.pkl"))

            prediction = model.predict(reduced)[0]
            label = le.inverse_transform([prediction])[0]
        elif algorithm == 'nb':
            feature = []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color_feat = extract_color_histogram(image)
            texture_feat = extract_haralick_features(gray)
            combined = np.concatenate((color_feat, texture_feat))
            feature.append(combined)

            pca = joblib.load(os.path.join(settings.MODELS_DATA_PATH, f"pca_{algorithm}_transform.pkl"))
            reduced = pca.transform(combined.reshape(1, -1))

            model = joblib.load(os.path.join(settings.MODELS_DATA_PATH, f"{algorithm}_model.pkl"))
            le = joblib.load(os.path.join(settings.MODELS_DATA_PATH, f"{algorithm}_label_encoder.pkl"))

            prediction = model.predict(reduced)[0]
            label = le.inverse_transform([prediction])[0]

        elif algorithm == 'cnn':
            class_names = ['apel', 'dodol', 'harum-manis']
            cnn_model = tf.keras.models.load_model(os.path.join(settings.MODELS_DATA_PATH, 'cnn_model.keras'))
            img = image.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)

            prediction = cnn_model.predict(img)
            predicted_class = np.argmax(prediction)
            label = class_names[predicted_class]

        else:
            return JsonResponse({'error': 'Invalid algorithm'}, status=400)

        return JsonResponse({'result': label})
