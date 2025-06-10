import cv2
import numpy as np
import joblib
import argparse
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA

# ==== Parameter HSV untuk segmentasi ====
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

# ==== Ekstraksi Fitur ====
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

# ==== Pipeline Prediksi ====
def predict_image(image_path, model_type='knn'):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Gagal membaca gambar: {image_path}")
        return

    # 2. Segmentasi HSV (warna hijau)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    segmented = cv2.bitwise_and(img, img, mask=mask)

    # 3. Resize ke 128x128
    resized = cv2.resize(segmented, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 4. Ekstraksi fitur
    color_feat = extract_color_histogram(resized)
    texture_feat = extract_haralick_features(gray)
    combined_feat = np.concatenate((color_feat, texture_feat)).reshape(1, -1)

    # 5. Load model, PCA, dan LabelEncoder
    if model_type == 'knn':
        model = joblib.load('model/knn/knn_model.pkl')
        pca = joblib.load('model/knn/pca_knn_transform.pkl')
        le = joblib.load('extracted_features/knn/label_encoder.pkl')
    elif model_type == 'nb':
        model = joblib.load('model/nb/naive_bayes_model.pkl')
        pca = joblib.load('model/nb/pca_nb_transform.pkl')
        le = joblib.load('extracted_features/nb/label_encoder.pkl')
    else:
        print("[!] Tipe model tidak dikenali. Gunakan 'knn' atau 'nb'.")
        return

    # 6. Reduksi dimensi dengan PCA
    reduced_feat = pca.transform(combined_feat)

    # 7. Prediksi
    pred = model.predict(reduced_feat)
    label = le.inverse_transform(pred)[0]

    print(f"\n📷 Gambar: {image_path}")
    print(f"🔎 Prediksi Kelas ({model_type.upper()}): {label}")

# ==== CLI ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi satu gambar menggunakan model KNN atau Naive Bayes.")
    parser.add_argument("image_path", help="Path ke file gambar (jpg/png)")
    parser.add_argument("--model", choices=["knn", "nb"], default="knn", help="Pilih model: knn atau nb")
    args = parser.parse_args()

    predict_image(args.image_path, model_type=args.model)

#python predict_single_image.py images/sample.jpg --model knn