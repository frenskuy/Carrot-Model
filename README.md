# 🥕 Carrot Quality Classification Model

Selamat datang di repositori **Carrot-Model**!  
Proyek ini berfokus pada pengembangan model machine learning berbasis Support Vector Machine (SVM) untuk mengklasifikasikan kualitas wortel menjadi dua kategori: **Good** dan **Bad**. Model ini memanfaatkan fitur citra menggunakan metode HOG (Histogram of Oriented Gradients) untuk ekstraksi fitur.

---

## 🚀 Fitur Utama

- 🔍 **Ekstraksi Fitur HOG** dari citra wortel.
- 🧠 **Klasifikasi dengan SVM** yang telah dilatih dan disimpan dalam format `.pkl`.
- 📂 Mendukung dataset hasil augmentasi (flip, rotate, zoom, dll).
- 🌐 **Aplikasi Streamlit** interaktif untuk mendeteksi kualitas wortel dari gambar input.
- 📊 Evaluasi performa model dengan metrik seperti Confusion Matrix dan Classification Report.

---

## 📁 Struktur Direktori
```bash
Carrot-Model/
├── model_carrot.pkl             # Model SVM yang telah dilatih
├── CARROT/                      # Folder dataset berisi gambar wortel (good/bad)
├── app.py                       # Aplikasi Streamlit untuk deteksi kualitas
├── train_and_save_model.ipynb   # Script pelatihan model
└── README.md                    # Dokumentasi ini
```
---

## 🛠️ Teknologi yang Digunakan
- Python
- OpenCV
- scikit-learn
- scikit-image (HOG)
- Streamlit
- NumPy

## 🌐 URL
[GOOD VS BAD CARROT](https://ml-modelcarrot.streamlit.app/)

## 👤 Author
Dikembangkan oleh [@frenskuy](https://github.com/frenskuy)
- 📧 frenkygilang@gmail.com
- 🌐 [Linkedln](https://www.linkedin.com/in/frenkyy/)

## 🤝 Kontribusi
Pull request dan isu baru sangat disambut! Jika Anda memiliki saran atau ingin berkontribusi, silakan fork dan submit PR Anda.

