import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Konfigurasi halaman
st.set_page_config(page_title="🥕 Klasifikasi Wortel", layout="centered")

# Load model carrot (ViT) dari file lokal
@st.cache_resource
def load_model():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load("vit_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Transformasi gambar (harus sesuai saat training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>🥕 Klasifikasi Wortel dengan Vision Transformer</h1>", unsafe_allow_html=True)
st.write("Upload gambar wortel untuk mengetahui apakah wortel tersebut termasuk kelas **GOOD** atau **BAD** berdasarkan model Vision Transformer (ViT).")

st.markdown("---")

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload gambar wortel (JPG, PNG, dll)", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"])

# Jika ada file gambar
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Gambar yang Diunggah", use_container_width=True)

        st.markdown("### 🔍 Hasil Prediksi")

        # Preprocessing
        input_tensor = transform(image).unsqueeze(0)

        # Preprocess and predict
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()

        class_labels = ["BAD", "GOOD"]
        confidence = probabilities[predicted_class].item()

        # Confidence threshold to determine if image might be neither
        confidence_threshold = 0.75

        if confidence < confidence_threshold:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold; color: orange;'>Class: Neither BAD or GOOD</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>The model is not confident enough to classify this image as a BAD or GOOD (Confidence: {confidence:.2%}).</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold;'>Class: {class_labels[predicted_class]}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>Confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True
            )


    except UnidentifiedImageError:
        st.error("❗ Gambar tidak valid atau rusak.")
    except Exception as e:
        st.error(f"❗ Terjadi error: {e}")
