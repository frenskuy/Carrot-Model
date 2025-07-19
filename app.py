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

# Label klasifikasi (disesuaikan dengan label saat training)
label_map = {0: "BAD", 1: "GOOD"}

# Transformasi gambar (harus sesuai saat training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])  # normalisasi ke [-1, 1]
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

        # Prediksi model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probabilities).item()
            confidence = probabilities[pred_class].item()
            label = label_map.get(pred_class, "UNKNOWN")

        # Tampilkan hasil
        if label == "GOOD":
            st.success(f"✅ Wortel ini diprediksi sebagai: **GOOD** dengan keyakinan {confidence:.2%}")
        elif label == "BAD":
            st.error(f"❌ Wortel ini diprediksi sebagai: **BAD** dengan keyakinan {confidence:.2%}")
        else:
            st.warning(f"⚠️ Kelas tidak dikenali: {pred_class}")

        st.markdown("---")
        st.info("Model ini dilatih menggunakan Vision Transformer (`vit_base_patch16_224`) dengan dataset wortel yang dibagi menjadi dua kelas: GOOD dan BAD.")

    except UnidentifiedImageError:
        st.error("❗ Gambar tidak valid atau rusak.")
    except Exception as e:
        st.error(f"❗ Terjadi error: {e}")
