import os
import shutil
import random

source_dir = 'augmented_nb'
target_dir = 'split_nb_dataset'
train_ratio = 0.7  # 70% train
val_ratio = 0.15   # 15% validation
test_ratio = 0.15  # 15% test

assert train_ratio + val_ratio + test_ratio == 1.0

# Set up reproducibility
random.seed(123)

# Loop through each class folder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # List all images in the class folder
    images = os.listdir(class_path)
    random.shuffle(images)

    num_images = len(images)
    train_split_index = int(num_images * train_ratio)
    val_split_index = train_split_index + int(num_images * val_ratio)

    train_images = images[:train_split_index]
    val_images = images[train_split_index:val_split_index]
    test_images = images[val_split_index:]

    # Define target paths
    train_class_dir = os.path.join(target_dir, 'train', class_name)
    val_class_dir = os.path.join(target_dir, 'val', class_name)
    test_class_dir = os.path.join(target_dir, 'test', class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy files to train directory
    for img in train_images:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(train_class_dir, img)
        shutil.copy(src_path, dst_path)

    # Copy files to validation directory
    for img in val_images:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(val_class_dir, img)
        shutil.copy(src_path, dst_path)

    # Copy files to test directory
    for img in test_images:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(test_class_dir, img)
        shutil.copy(src_path, dst_path)

print("Dataset split into train, validation, and test complete.")