import os
import cv2
import albumentations as A

# ==== PILIH SALAH SATU KONFIGURASI BERIKUT DENGAN MENGHAPUS KOMENTARNYA ====

# --- Konfigurasi untuk CNN (augmentasi agresif) ---
transform = A.Compose([
    A.Rotate(limit=30),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
])
output_dir = 'augmented_cnn'

# --- Konfigurasi untuk KNN (augmentasi ringan) ---
# transform = A.Compose([
#     A.Rotate(limit=15, p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.3)
# ])
# output_dir = 'augmented_knn'

# --- Konfigurasi untuk Naive Bayes (augmentasi minimal) ---
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5)
# ])
# output_dir = 'augmented_nb'

# ==== ATURAN DASAR ====
input_dir = 'split_dataset'     # input root folder (with train/val/test)
augment_per_image = 5           # jumlah augmentasi per gambar

# Mulai proses augmentasi
for split in ['train', 'val', 'test']:
    split_input_path = os.path.join(input_dir, split)
    split_output_path = os.path.join(output_dir, split)
    
    if not os.path.isdir(split_input_path):
        continue

    for class_name in os.listdir(split_input_path):
        input_class_path = os.path.join(split_input_path, class_name)
        output_class_path = os.path.join(split_output_path, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for img_name in os.listdir(input_class_path):
            img_path = os.path.join(input_class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            for i in range(augment_per_image):
                augmented = transform(image=img)
                aug_img = augmented["image"]
                save_name = f"aug_{i}_{img_name}"
                save_path = os.path.join(output_class_path, save_name)
                cv2.imwrite(save_path, aug_img)

print(f"✅ Augmentasi selesai. Dataset disimpan di folder: {output_dir}")