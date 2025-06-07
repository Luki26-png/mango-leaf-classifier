import cv2
import numpy as np
import tensorflow as tf
import argparse
import os

# ==== Segmentasi HSV (warna hijau) ====
def segment_hsv_green(image):
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# ==== Preprocessing gambar ====
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"[!] Gagal membaca gambar: {image_path}")
    
    segmented = segment_hsv_green(img)
    resized = cv2.resize(segmented, target_size)
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)  # Tambahkan batch dimensi

# ==== Load label class (manual mapping sesuai urutan training) ====
class_names = ['apel', 'dodol', 'harum-manis']  # Pastikan sesuai urutan label di dataset

# ==== Pipeline prediksi CNN ====
def predict_with_cnn(image_path, model_path='model/cnn/mango_leaf_classifier.keras'):
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess
    input_image = preprocess_image(image_path)

    # Predict
    prediction = model.predict(input_image)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index]

    # Output
    print(f"\n📷 Gambar: {image_path}")
    print(f"🤖 Prediksi: {predicted_label} ({confidence * 100:.2f}%)")

# ==== CLI ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi satu gambar menggunakan model CNN.")
    parser.add_argument("image_path", help="Path ke file gambar (jpg/png)")
    parser.add_argument("--model", default="model/cnn/mango_leaf_classifier.keras", help="Path model .keras")
    args = parser.parse_args()

    predict_with_cnn(args.image_path, args.model)