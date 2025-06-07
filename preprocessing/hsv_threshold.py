import cv2
import numpy as np
import os

# Folder input dan output
input_dir = 'dataset'
output_dir = 'segmented_dataset'

# Rentang warna hijau (daun) di HSV — sesuaikan jika perlu
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

# Membuat folder output jika belum ada
os.makedirs(output_dir, exist_ok=True)

# Loop tiap kelas (subfolder)
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Buat folder output per kelas
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Loop semua gambar dalam kelas
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[!] Gagal membaca gambar: {img_path}")
            continue

        # Konversi ke HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Buat mask warna hijau
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Terapkan mask ke gambar asli
        result = cv2.bitwise_and(img, img, mask=mask)

        # Simpan gambar hasil segmentasi
        save_path = os.path.join(output_class_path, img_name)
        cv2.imwrite(save_path, result)

print("✅ Segmentasi selesai. Hasil disimpan di folder 'segmented_dataset'.")