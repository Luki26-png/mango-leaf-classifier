# train_and_evaluate_naive_bayes.py
# Melakukan load .csv, reduksi dimensi PCA, pelatihan Naive Bayes, evaluasi, dan menampilkan label asli
import os
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

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

#the dir source
input_dir = "extracted_features/nb"
# Load LabelEncoder dari file
le = joblib.load(os.path.join(input_dir,"label_encoder.pkl"))

# Load dataset dan lakukan PCA hanya pada data training
X_train, y_train, pca = load_features(os.path.join(input_dir,"train_features.csv"), fit_pca=True, n_components=15)
X_val, y_val, _ = load_features(os.path.join(input_dir,"val_features.csv"), pca=pca)
X_test, y_test, _ = load_features(os.path.join(input_dir,"test_features.csv"), pca=pca)

# Latih model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Simpan model dan PCA ke file
joblib.dump(nb_model, "model/nb/naive_bayes_model.pkl")
joblib.dump(pca, "model/nb/pca_nb_transform.pkl")

print("✅ Model Naive Bayes dan PCA berhasil disimpan.")

# Evaluasi model
def evaluate(name, X, y):
    preds = nb_model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"\n📊 Evaluasi: {name}")
    print(f"Akurasi: {acc:.4f}")
    print(classification_report(y, preds, target_names=le.classes_))

evaluate("TRAINING", X_train, y_train)
evaluate("VALIDATION", X_val, y_val)
evaluate("TEST", X_test, y_test)

