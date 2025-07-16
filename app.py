import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image

# Load model
svm_model = joblib.load("model_carrot.pkl")  # pastikan file ini ada

# Ekstrak fitur HOG
def extract_hog_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

# Streamlit UI
st.title("Klasifikasi Wortel (Good vs Bad) 🥕")
st.write("Upload gambar wortel untuk diklasifikasikan:")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Konversi PIL ke OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Ekstrak fitur dan prediksi
    features = extract_hog_features(image_cv).reshape(1, -1)
    prediction = svm_model.predict(features)[0]
    confidence = svm_model.predict_proba(features)[0].max()

    st.markdown(f"### Prediksi: `{prediction.upper()}`")
    st.markdown(f"**Confidence:** {confidence:.2f}")
