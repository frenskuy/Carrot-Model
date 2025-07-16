import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image

# Load model
with open('model_carrot.pkl', 'rb') as f:
    model = pickle.load(f)

def extract_features_pil(pil_img):
    img = pil_img.resize((64, 64))
    img_array = np.array(img)
    if img_array.ndim == 3:
        features = img_array.mean(axis=(0, 1))
    else:  # grayscale
        features = np.array([img_array.mean()] * 3)
    return features.reshape(1, -1)

st.title("Klasifikasi Wortel Baik / Buruk 🍠🥕")

uploaded_file = st.file_uploader("Unggah gambar wortel", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar diunggah', use_container_width=True)
    
    features = extract_features_pil(image)
    prediction = svm.predict(features)[0]

    label = "Wortel Baik ✅" if prediction == 1 else "Wortel Buruk ❌"
    st.subheader(f"Prediksi: {label}")
