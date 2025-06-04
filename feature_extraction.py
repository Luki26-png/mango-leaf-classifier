import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder

# [hsv]: gambar sumber (input HSV image)
# [0, 1, 2]: channel yang dihitung → H, S, V
# None: tidak ada mask, artinya seluruh citra dihitung
# bins = (8, 4, 4): jumlah bin untuk masing-masing channel
# [0, 180, 0, 256, 0, 256]: rentang nilai untuk H (0–179), S (0–255), dan V (0–255)

def extract_color_histogram(image, bins=(8, 4, 4)):
    #ubah gambar ke ruang warna hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hitung histogram
    # [hsv]: gambar sumber (input HSV image)
    # [0, 1, 2]: channel yang dihitung → H, S, V
    # None: tidak ada mask, artinya seluruh citra dihitung
    # bins = (8, 4, 4): jumlah bin untuk masing-masing channel
    # [0, 180, 0, 256, 0, 256]: rentang nilai untuk H (0–179), S (0–255), dan V (0–255)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    #normalisasi histogram
    cv2.normalize(hist, hist)

    #ubah histogram jadi 1D vector
    return hist.flatten()

def extract_haralick_features(image_gray):
    # Hitung GLCM
    glcm = graycomatrix(
        image_gray, distances=[1], 
        angles=[0], levels=256, symmetric=True, normed=True)
    # Ekstraksi fitur
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    return np.array(features)


def extract_shape_features(image_gray):
    # Threshold untuk mendapatkan kontur
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(4)  # Tidak ditemukan kontur

    c = max(contours, key=cv2.contourArea)  # Ambil kontur terbesar
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    roundness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

    return np.array([aspect_ratio, extent, solidity, roundness])

# Mulai proses ekstraksi
dataset_path = "dataset_segmented"
output_data = []
labels = []

for label_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label_folder)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))

            # Ekstraksi warna
            color_features = extract_color_histogram(image)

            # Ekstraksi tekstur
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            haralick_features = extract_haralick_features(gray_image)

            # Ekstraksi bentuk
            shape_features = extract_shape_features(gray_image)

            # Gabungkan semua
            combined_features = np.concatenate([color_features, haralick_features, shape_features])
            output_data.append(combined_features)
            labels.append(label_folder)

# Label encoding dan simpan
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

df = pd.DataFrame(output_data)
df['label'] = encoded_labels
df.to_csv("ekstraksi_fitur_dengan_bentuk.csv", index=False)

print("✅ Fitur lengkap (warna + tekstur + bentuk) berhasil disimpan di 'ekstraksi_fitur_dengan_bentuk.csv'")
print("📌 Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
