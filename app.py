import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Wortel", page_icon="🥕", layout="centered")

# Judul dan Deskripsi
st.markdown("<h1 style='text-align: center;'>🥕 Klasifikasi Wortel (Good vs Bad)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload gambar wortel untuk mengetahui apakah wortel tersebut <b>Good</b> atau <b>Bad</b>!</p>", unsafe_allow_html=True)

# Load model SVM yang sudah dilatih dengan MobileNetV2
svm_model = joblib.load("svm_mobilenet_features.pkl")

# Load MobileNetV2 untuk ekstraksi fitur (tanpa top layer)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
feature_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Fungsi ekstraksi fitur menggunakan MobileNetV2
def extract_mobilenet_features(image):
    image = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = img_to_array(image_rgb)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    features = feature_model.predict(image_array, verbose=0)
    return features

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload gambar wortel (jpg, jpeg, png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='📷 Gambar yang diupload', use_container_width=True)

    with st.spinner("🔍 Sedang memproses dan mengklasifikasikan gambar..."):
        # Konversi ke OpenCV dan ekstraksi fitur
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        features = extract_mobilenet_features(image_cv)

        # Prediksi
        prediction = svm_model.predict(features)[0]
        confidence = svm_model.predict_proba(features)[0].max()

    # Hasil prediksi
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔎 Hasil Prediksi:")
        if prediction.upper() == "GOOD":
            st.success(f"Wortel ini diprediksi sebagai: **GOOD** 🟢")
        else:
            st.error(f"Wortel ini diprediksi sebagai: **BAD** 🔴")

    with col2:
        st.markdown("### 📊 Confidence:")
        st.metric(label="Tingkat Keyakinan", value=f"{confidence * 100:.2f}%")

    st.markdown("---")
    st.info("Model ini menggunakan fitur **MobileNetV2 + SVM** untuk mengklasifikasikan gambar wortel.")

import tensorflow as tf
st.write("TF version:", tf.__version__)
