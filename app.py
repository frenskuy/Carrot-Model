import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Fungsi augmentasi gambar
def augment_image(img):
    augmented_images = []
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)
    for angle in [15, -15]:
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented_images.append(rotated)
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    augmented_images.append(bright)
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented_images.append(noisy)
    return augmented_images

# Load dan augmentasi data
def load_data(uploaded_files):
    X, y = [], []
    label_map = {'good': 0, 'bad': 1}
    for uploaded_file in uploaded_files:
        label = 'good' if 'good' in uploaded_file.name else 'bad'
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((64, 64))
        img_np = np.array(img)
        X.append(img_np)
        y.append(label_map[label])
        for aug_img in augment_image(img_np):
            X.append(aug_img)
            y.append(label_map[label])
    return np.array(X), np.array(y)

# Ekstraksi fitur CNN
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu')
    ])
    return model

# Streamlit UI
st.title("Klasifikasi Wortel: CNN + SVM")

uploaded_files = st.file_uploader("Upload gambar (label harus ada di nama file, misal: good_1.jpg)", accept_multiple_files=True, type=['jpg', 'png'])

if uploaded_files:
    st.write("Jumlah gambar yang diupload:", len(uploaded_files))
    X, y = load_data(uploaded_files)
    X = X.astype('float32') / 255.0

    # Bangun CNN dan ekstrak fitur
    cnn = build_cnn_model()
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_output = cnn.predict(X, verbose=0)

    # Jika fitur CNN belum dilatih, latih di sini
    X_train, X_test, y_train, y_test = train_test_split(cnn_output, y, test_size=0.2, random_state=42)

    # SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Evaluasi
    st.text("Hasil Evaluasi SVM:")
    st.text(classification_report(y_test, y_pred, target_names=['Good', 'Bad']))
