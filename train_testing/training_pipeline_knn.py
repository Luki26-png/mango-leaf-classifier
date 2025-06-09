import os
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_features(csv_file, pca=None, fit_pca=False, n_components=30):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if fit_pca:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return X, y, pca

# 📁 Path configuration
input_dir = "extracted_features/knn"
output_dir = "model/knn"
os.makedirs(output_dir, exist_ok=True)

# 📥 Load label encoder
le = joblib.load(os.path.join(input_dir, "label_encoder.pkl"))

# 📊 Load training and validation features
X_train, y_train, pca = load_features(os.path.join(input_dir, "train_features.csv"), fit_pca=True, n_components=30)
X_val, y_val, _ = load_features(os.path.join(input_dir, "val_features.csv"), pca=pca)

# 🔍 Hyperparameter tuning (k search)
best_k = None
best_acc = 0
best_model = None

print("\n🔧 Searching for best k using validation set...\n")
for k in range(1, 10, 2):  # Try odd values of k from 1 to 9
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"k = {k} → Validation Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_k = k
        best_model = model

# ✅ Save the best model and PCA
joblib.dump(best_model, os.path.join(output_dir, "knn_model.pkl"))
joblib.dump(pca, os.path.join(output_dir, "pca_knn_transform.pkl"))

print(f"\n✅ Best k = {best_k} with accuracy = {best_acc:.4f}")
print("📦 Model and PCA transformer saved.")

# 📈 Final Evaluation Report on Validation Set
val_preds = best_model.predict(X_val)
print("\n📊 Final Evaluation on Validation Set:")
print(classification_report(y_val, val_preds, target_names=le.classes_))