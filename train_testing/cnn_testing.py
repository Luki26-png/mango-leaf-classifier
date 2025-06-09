import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# 1. Load Model
model = tf.keras.models.load_model('mango_leaf_classifier.keras')

# 2. Prepare Test Data
test_path = 'split_cnn_dataset/test'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Get class names from directory structure
class_names = sorted(os.listdir(test_path))  # ['apel', 'dodol', 'harum-manis']
num_classes = len(class_names)

#check if you can createa the label
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalization (must match training)
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = test_ds.map(normalize)

# 3. Get Predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    y_true.extend(labels.numpy())
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 4. Evaluation Metrics
print("\nTest Accuracy: {:.2%}".format(np.mean(y_true == y_pred)))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 5. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
