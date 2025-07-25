import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Wortel (GOOD vs BAD)", layout="centered")

# Ukuran gambar input
IMG_SIZE = 50

# Definisi arsitektur model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].numel()
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("carrot_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Fungsi preprocessing gambar
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # konversi ke grayscale
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_tensor = torch.Tensor(image_array).view(-1, 1, IMG_SIZE, IMG_SIZE)
    return image_tensor

# Load model
model = load_model()

# Tampilan antarmuka Streamlit
st.title("Klasifikasi Wortel: GOOD vs BAD")
st.write("Unggah gambar wortel untuk diprediksi apakah termasuk **GOOD** atau **BAD**.")

uploaded_file = st.file_uploader("Pilih gambar wortel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        input_tensor = preprocess_image(img)

        with torch.no_grad():
            output = model(input_tensor)[0]
            prediction = torch.argmax(output).item()
            confidence = torch.max(output).item()
            prob_good = output[0].item()
            prob_bad = output[1].item()

        st.subheader("Hasil Prediksi:")
        if confidence < 0.75:
            st.warning("The model is not confident enough to classify this image as a good or bad.")
        else:
            label = "GOOD" if prediction == 0 else "BAD"
            st.success(f"Prediksi: **{label}** ({confidence*100:.2f}% yakin)")

        # Menampilkan probabilitas kedua kelas
        st.write(f"Confidence GOOD: `{prob_good:.4f}`")
        st.write(f"Confidence BAD: `{prob_bad:.4f}`")

    except UnidentifiedImageError:
        st.error("Gagal membaca gambar. Pastikan file yang diunggah adalah gambar yang valid.")
