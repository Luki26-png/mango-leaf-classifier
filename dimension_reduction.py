import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# === KONFIGURASI ===
input_csv = "ekstraksi_fitur.csv"     # File hasil ekstraksi fitur
output_csv = "fitur_pca.csv"          # File hasil setelah PCA
n_components = 15                     # Jumlah fitur setelah reduksi

# === LOAD DATA ===
df = pd.read_csv(input_csv)

# Pisahkan fitur dan label (kolom terakhir adalah label)
features = df.iloc[:, :-1].values     # Semua kolom kecuali terakhir
labels = df.iloc[:, -1].values        # Kolom label

# === PCA ===
pca = PCA(n_components=n_components)
features_pca = pca.fit_transform(features)

# === Gabungkan kembali dengan label ===
final_data = np.hstack((features_pca, labels.reshape(-1, 1)))

# Simpan ke file baru
columns = [f'pca_{i+1}' for i in range(n_components)] + ['label']
df_pca = pd.DataFrame(final_data, columns=columns)
df_pca.to_csv(output_csv, index=False)

print(f"✅ PCA selesai. Hasil disimpan di: {output_csv}")