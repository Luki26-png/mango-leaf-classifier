# train_and_evaluate_naive_bayes.py
# Melakukan load .csv, reduksi dimensi PCA, pelatihan Naive Bayes, evaluasi, dan menyimpan model
import os
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

def load_features(csv_file, pca=None, fit_pca=False, n_components=15): 
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if fit_pca:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return X, y, pca

# 📁 Directory setup
input_dir = "extracted_features/nb"
output_dir = "model/nb"
os.makedirs(output_dir, exist_ok=True)

# 🏷️ Load LabelEncoder
le = joblib.load(os.path.join(input_dir, "label_encoder.pkl"))

# 📊 Load training and validation datasets
X_train, y_train, pca = load_features(os.path.join(input_dir, "train_features.csv"), fit_pca=True, n_components=15)

# 🧠 Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 💾 Save model and PCA
joblib.dump(nb_model, os.path.join(output_dir, "naive_bayes_model.pkl"))
joblib.dump(pca, os.path.join(output_dir, "pca_nb_transform.pkl"))

print("✅ Model Naive Bayes dan PCA berhasil disimpan.")

# 📈 Evaluate model
def evaluate(name, X, y):
    preds = nb_model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"\n📊 Evaluasi: {name}")
    print(f"Akurasi: {acc:.4f}")
    print(classification_report(y, preds, target_names=le.classes_))

evaluate("TRAINING", X_train, y_train)