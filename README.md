# 🌿 Medic Leaf - Klasifikasi Daun Tanaman Obat

**Medic Leaf** adalah aplikasi desktop berbasis Python yang memungkinkan pengguna untuk mengklasifikasikan jenis daun tanaman obat menggunakan model deep learning. Aplikasi ini dirancang dengan antarmuka pengguna yang sederhana dan modern menggunakan `Tkinter`.

## 🚀 Fitur

- 🔍 Klasifikasi otomatis daun menjadi salah satu dari:
  - Kumis Kucing
  - Pegagan
  - Sirih
- 📷 Deteksi daun otomatis dari gambar atau kamera
- 🖼️ Bounding box pada area daun yang terdeteksi
- 🧠 Model deep learning ringan berbasis Keras
- 🖥️ Antarmuka GUI sederhana dan responsif

## 🗂️ Struktur Proyek

```
├── main.py             # Aplikasi utama
├── keras_model.h5      # Model Keras hasil training
├── labels.txt          # Daftar label klasifikasi
├── requirements.txt    # Daftar dependensi
└── README.md           # Dokumentasi ini
```

## 🛠️ Instalasi

1. Pastikan Python 3.8 atau lebih baru telah terpasang.
2. Install semua dependensi dengan perintah:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
python main.py
```

## 📦 Dependensi

Semua dependensi tercantum di `requirements.txt`, yaitu:

- `tensorflow==2.13.0` atau `tensorflow-cpu==2.13.0` (untuk sistem tanpa GPU)
- `opencv-python`
- `numpy`
- `Pillow`

## 💡 Catatan

- Pastikan webcam terhubung jika ingin menggunakan fitur kamera.
- Gunakan gambar dengan kualitas cukup agar deteksi optimal.

## 📜 Lisensi

Aplikasi ini dapat digunakan bebas untuk keperluan edukasi dan riset.
