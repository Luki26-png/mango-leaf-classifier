import cv2
import os

# Folder input dan output
input_dir = 'dataset_segmented'
output_dir = 'resized_dataset'

# Ukuran target
target_size = (128, 128)

# Buat folder output jika belum ada
os.makedirs(output_dir, exist_ok=True)

# Loop setiap kelas di dalam segmented_dataset
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Buat folder untuk kelas ini di output
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Loop semua gambar dalam kelas
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[!] Gagal membaca gambar: {img_path}")
            continue

        # Resize gambar
        resized_img = cv2.resize(img, target_size)

        # Simpan gambar hasil resize
        save_path = os.path.join(output_class_path, img_name)
        cv2.imwrite(save_path, resized_img)

print("✅ Resize selesai. Hasil disimpan di folder 'resized_dataset'.")