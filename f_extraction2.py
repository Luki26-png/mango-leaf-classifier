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

            # Gabungkan semua
            combined_features = np.concatenate([color_features, haralick_features])
            output_data.append(combined_features)
            labels.append(label_folder)

# Label encoding dan simpan
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

df = pd.DataFrame(output_data)
df['label'] = encoded_labels
df.to_csv("ekstraksi_fitur.csv", index=False)

print("✅ Fitur lengkap (warna + tekstur + bentuk) berhasil disimpan di 'ekstraksi_fitur_dengan_bentuk.csv'")
print("📌 Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
