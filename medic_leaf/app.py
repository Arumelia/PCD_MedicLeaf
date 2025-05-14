import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Konfigurasi halaman ---
st.set_page_config(page_title="MedicLeaf - Klasifikasi Tanaman Obat", layout="centered")

# --- Judul Aplikasi ---
st.title("🌿 MedicLeaf")
st.subheader("Klasifikasi Tanaman Obat dari Gambar Daunnya")

# --- Label Kelas ---
CLASS_NAMES = ['kumis_kucing', 'pegagan', 'sirih']

# --- Load model ---
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)  # Tidak gunakan pretrained karena sudah dilatih
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Transformasi gambar ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Fungsi prediksi dengan confidence ---
def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).squeeze().numpy()
    predicted_index = np.argmax(probs)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = probs[predicted_index]
    return predicted_label, confidence, probs

# --- Upload gambar ---
uploaded_file = st.file_uploader("📤 Unggah gambar daun tanaman obat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)

    with st.spinner("🔎 Mengklasifikasi daun..."):
        label, conf, all_probs = predict(image)
        conf_pct = conf * 100

        if conf < 0.6:
            st.warning("⚠️ Model tidak yakin terhadap gambar ini. Karena keterbatasan daun yang dikenali oleh MedicLeaf.")
            st.info(f"Prediksi terdekat: **{label}** dengan confidence {conf_pct:.2f}%")
        else:
            st.success(f"🌱 Jenis Tanaman Obat: **{label}** (Confidence: {conf_pct:.2f}%)")

            # --- Info daun ---
            if label == "kumis_kucing":
                st.info("🌿 Kumis Kucing: Bermanfaat untuk mengatasi gangguan saluran kemih dan batu ginjal.")
            elif label == "sirih":
                st.info("🌿 Sirih: Umumnya digunakan sebagai antiseptik alami dan mengatasi bau mulut.")
            elif label == "pegagan":
                st.info("🌿 Pegagan: Dipercaya dapat meningkatkan fungsi otak dan mempercepat penyembuhan luka.")
else:
    st.info("Silakan unggah gambar daun tanaman obat untuk diklasifikasi.")
