import cv2
import numpy as np
import os

def robust_segmentation(img_path, output_dir=None):
    img = cv2.imread(img_path)
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Adaptive HSV thresholds (widened ranges)
    lower_green = np.array([30, 40, 40])  # Expanded from [35,40,40]
    upper_green = np.array([90, 255, 255])  # Expanded from [85,255,255]
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {img_path}")
        return original
    
    # Filter contours by area (remove tiny noise)
    min_area = img.shape[0] * img.shape[1] * 0.01  # At least 1% of image area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        print(f"No large contours in {img_path}")
        return original
    
    # Select by centrality AND size
    img_center = np.array([img.shape[1]//2, img.shape[0]//2])
    def centrality_score(c):
        center = cv2.minEnclosingCircle(c)[0]
        distance = np.linalg.norm(img_center - center)
        area = cv2.contourArea(c)
        return distance / (area ** 0.5)  # Balance distance and size
    
    best_cnt = min(valid_contours, key=centrality_score)
    
    # Create mask with smoothing
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = [255,255,255]
    
    return result

# Batch processing
input_root = "dataset"
output_root = "dataset_segmented"

for class_name in ["harum-manis", "apel", "dodol"]:
    class_dir = os.path.join(input_root, class_name)

    #route untuk save hasil
    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_dir, img_name)
            segmented_img = robust_segmentation(img_path, output_root)
            save_path = os.path.join(output_class_path, img_name)
            cv2.imwrite(save_path, segmented_img)

print("✅ Segmentasi selesai. Hasil disimpan di folder 'dataset_segmented'.")