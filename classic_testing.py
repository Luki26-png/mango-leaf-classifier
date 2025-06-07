# test_model.py
# Load test dataset, apply PCA, predict using saved models (KNN or NB), show accuracy & confusion matrix

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_features(csv_file, pca):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X = pca.transform(X)
    return X, y

def evaluate_model(name, model, X_test, y_test, label_encoder):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n📊 Model: {name}")
    print(f"Akurasi: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Set paths for knn
test_csv_knn = "extracted_features/knn/test_features.csv" 
label_encoder_knn = joblib.load("extracted_features/knn/label_encoder.pkl")

# Set paths for nb
test_csv_nb = "extracted_features/nb/test_features.csv"
label_encoder_nb = joblib.load("extracted_features/nb/label_encoder.pkl")

# --- load knn model dan test data ---
knn_model = joblib.load("model/knn/knn_model.pkl")
knn_pca = joblib.load("model/knn/pca_knn_transform.pkl")
X_test_knn, y_test_knn = load_features(test_csv_knn, knn_pca)

# --- load naive bayes model dan test data ---
nb_model = joblib.load("model/nb/naive_bayes_model.pkl")
nb_pca = joblib.load("model/nb/pca_nb_transform.pkl")
X_test_nb, y_test_nb = load_features(test_csv_nb, nb_pca)

#evalute knn
evaluate_model("K-Nearest Neighbors", knn_model, X_test_knn, y_test_knn, label_encoder_knn)
#evaluate naive bayes
evaluate_model("Naive Bayes", nb_model, X_test_nb, y_test_nb, label_encoder_nb)