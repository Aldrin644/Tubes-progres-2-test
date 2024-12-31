import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration first
st.set_page_config(
    page_title="Prediksi Waktu Layar Smartphone",
    page_icon="📱",
)

# Path ke file data Anda
file_path = 'Smartphone Usage and Behavioral Dataset - mobile_usage_behavioral_analysis.csv'

# Membaca data
data = pd.read_csv(file_path)

# Menampilkan beberapa baris pertama data
st.write(data.head())

# Cek apakah ada nilai yang hilang
if data.isnull().sum().any():
    st.write("Missing values found, filling missing data with mean...")
    data.fillna(data.mean(), inplace=True)

# Memisahkan fitur (X) dan target (y)
X = data[[
    "Social_Media_Usage_Hours",
    "Productivity_App_Usage_Hours",
    "Gaming_App_Usage_Hours",
    "Total_App_Usage_Hours"
]]
y = data["Daily_Screen_Time_Hours"]  # Target: waktu layar aktif

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data (standarisasi)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Halaman untuk pengaturan model dan prediksi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Training Model", "Prediksi Waktu Layar", "Visualisasi Model", "Klasifikasi Pengguna"])

if page == "Beranda":
    st.title("Prediksi Waktu Layar Smartphone 📱")
    st.markdown("""
    Aplikasi ini memprediksi waktu layar aktif berdasarkan penggunaan aplikasi.
    Menggunakan model **K-Nearest Neighbors (KNN)**, aplikasi ini membantu memprediksi berapa banyak waktu yang akan dihabiskan di layar berdasarkan penggunaan aplikasi sosial media, produktivitas, gaming, dan total penggunaan aplikasi.
    """)

elif page == "Training Model":
    st.title("Training Model KNN")
    
    # Uji beberapa nilai K untuk mencari yang terbaik
    best_k = None
    best_r2 = float('-inf')
    for k in range(1, 21):
        knn = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, scoring='r2', cv=5)
        mean_score = scores.mean()
        if mean_score > best_r2:
            best_k = k
            best_r2 = mean_score

    st.write(f"Nilai K terbaik: {best_k} dengan R² rata-rata: {best_r2:.4f}")

    # Membuat model dengan K terbaik
    knn_regressor = KNeighborsRegressor(n_neighbors=best_k)
    knn_regressor.fit(X_train, y_train)

    # Menyimpan model di session state untuk digunakan di halaman lain
    st.session_state.knn_model = knn_regressor
    st.session_state.scaler = scaler

    # Memprediksi data uji
    y_pred = knn_regressor.predict(X_test)

    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R²): {r2}")

elif page == "Prediksi Waktu Layar":
    st.title("Hitung Prediksi Waktu Layar Smartphone ✏️")

    # Memastikan model dan scaler sudah tersedia
    if 'knn_model' not in st.session_state:
        st.error("Model belum dilatih. Silakan latih model terlebih dahulu.")
    else:
        knn_regressor = st.session_state.knn_model
        scaler = st.session_state.scaler

        col1, col2 = st.columns(2)
        with col1:
            social_media_usage = st.number_input("Masukkan Jam Penggunaan Media Sosial", 0, 24, 2)
            productivity_usage = st.number_input("Masukkan Jam Penggunaan Aplikasi Produktivitas", 0, 24, 3)
            gaming_usage = st.number_input("Masukkan Jam Penggunaan Aplikasi Gaming", 0, 24, 1)
            total_usage = st.number_input("Masukkan Total Jam Penggunaan Aplikasi", 0, 24, 6)
        
        # Menampilkan faktor-faktor yang mempengaruhi
        with col2:
            st.write("Faktor yang mungkin mempengaruhi:")
            st.write("Jika waktu penggunaan media sosial lebih banyak, waktu layar bisa lebih tinggi.")
            st.write("Jika waktu penggunaan aplikasi produktivitas lebih banyak, waktu layar bisa lebih rendah.")
            st.write("Jika waktu penggunaan aplikasi gaming lebih banyak, waktu layar bisa lebih tinggi.")
            st.write("Jika total waktu penggunaan aplikasi lebih banyak, waktu layar bisa lebih tinggi.")
        
        hitung = st.button("Prediksi Sekarang")
        if hitung:
            data_baru = pd.DataFrame([[social_media_usage, productivity_usage, gaming_usage, total_usage]], 
                                     columns=['Social_Media_Usage_Hours', 'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours', 'Total_App_Usage_Hours'])
            scaled_data_baru = scaler.transform(data_baru)
            prediksi = knn_regressor.predict(scaled_data_baru)
            st.success(f"Waktu layar diprediksi: {prediksi[0]:.2f} jam")

elif page == "Visualisasi Model":
    st.title("Visualisasi Performansi Model 📊")

    # Memastikan model dan scaler sudah tersedia
    if 'knn_model' not in st.session_state:
        st.error("Model belum dilatih. Silakan latih model terlebih dahulu.")
    else:
        knn_regressor = st.session_state.knn_model
        scaler = st.session_state.scaler

        # Memprediksi data uji
        y_pred = knn_regressor.predict(X_test)

        # Plot Residual (Error)
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, color='blue', alpha=0.6)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title("Residual Plot")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Residual (Error)")
        ax.grid(True)
        st.pyplot(fig)

        # Visualisasi Prediksi vs Nilai Sebenarnya
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(y_test, y_pred, alpha=0.6)
        ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
        ax2.set_title("Prediksi vs Nilai Sebenarnya")
        ax2.set_xlabel("Nilai Sebenarnya")
        ax2.set_ylabel("Prediksi")
        ax2.grid(True)
        st.pyplot(fig2)

elif page == "Klasifikasi Pengguna":
    st.title("Klasifikasi Pengguna Smartphone 💬")

    # Pengguna memberikan input total jam penggunaan
    total_usage_input = st.number_input("Masukkan Total Jam Penggunaan Aplikasi per Hari (dalam jam)", 0, 24, 5)

    if total_usage_input > 5:
        st.write("Anda termasuk **pengguna smartphone berat**.")
    else:
        st.write("Anda termasuk **pengguna smartphone ringan**.")
