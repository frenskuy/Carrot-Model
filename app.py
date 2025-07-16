import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Wortel", page_icon="🥕", layout="wide")

# Judul dan Deskripsi
st.markdown("<h1 style='text-align: center;'>🥕 Klasifikasi Wortel (Good vs Bad)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload gambar wortel untuk mengetahui apakah wortel tersebut <b>Good</b> atau <b>Bad</b>!</p>", unsafe_allow_html=True)

# Load model
svm_model = joblib.load("model_carrot.pkl")

# Fungsi ekstraksi fitur HOG
def extract_hog_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload gambar wortel (jpg, jpeg, png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(caption='📷 Gambar yang diupload', image, use_container_width=True)

    with st.spinner("🔍 Sedang memproses dan mengklasifikasikan gambar..."):
        # Konversi gambar
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        features = extract_hog_features(image_cv).reshape(1, -1)

        # Prediksi
        prediction = svm_model.predict(features)[0]
        confidence = svm_model.predict_proba(features)[0].max()

    # Hasil prediksi
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔎 Hasil Prediksi:")
        if prediction.lower() == "good":
            st.success(f"Wortel ini diprediksi sebagai: **GOOD** 🟢")
        else:
            st.error(f"Wortel ini diprediksi sebagai: **BAD** 🔴")

    with col2:
        st.markdown("### 📊 Confidence:")
        st.progress(confidence)
        st.write(f"**{confidence * 100:.2f}%** keyakinan model terhadap hasil prediksi.")

    st.markdown("---")
    st.info("Model ini menggunakan fitur HOG + SVM untuk mengklasifikasikan gambar wortel.")
