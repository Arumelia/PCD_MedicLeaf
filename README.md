
# MedicLeaf - Aplikasi Klasifikasi Daun Tanaman Obat

MedicLeaf adalah aplikasi berbasis Streamlit untuk mengklasifikasikan jenis tanaman obat dari gambar daun menggunakan model CNN (MobileNetV2).

## Struktur Proyek

```
MedicLeaf/
├── dataset/                # Folder berisi subfolder gambar tiap kelas daun
│   ├── kumis_kucing/
│   ├── pegagan/
│   └── sirih/
├── app.py                 # Aplikasi utama Streamlit
├── train_model.py         # Skrip untuk melatih model
├── generate_labels.py     # Skrip untuk menghasilkan labels.txt dari dataset
├── labels.txt             # Daftar nama kelas yang diambil dari folder dataset
├── model.pth              # Model hasil training
├── requirements.txt       # Daftar dependensi Python
└── all_launch.py          # Jalankan pipeline otomatis: label → training → app
```

## Langkah Penggunaan

1. **Siapkan Dataset**
   - Tempatkan gambar-gambar daun ke dalam subfolder sesuai jenisnya di dalam folder `dataset/`.

2. **Generate Label**
   Jalankan file berikut untuk menghasilkan `labels.txt`:
   ```bash
   python generate_labels.py
   ```

3. **Training Model**
   Latih model MobileNetV2 menggunakan:
   ```bash
   python train_model.py
   ```
   Hasil model akan disimpan sebagai `model.pth`.

4. **Jalankan Aplikasi**
   Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

5. **Atau Jalankan Semua Sekaligus**
   Jalankan:
   ```bash
   streamlit run all_launch.py
   ```
   untuk secara otomatis generate label → training model → membuka aplikasi.

## Catatan Penting

- File `labels.txt` akan disusun otomatis dari folder dalam `dataset/`.
- Jika ada folder yang kosong atau hanya berisi file rusak, label tersebut tidak akan dimasukkan.
- Pastikan nama folder tidak mengandung spasi atau huruf besar untuk konsistensi.

## Dependencies

Instal semua dependensi dengan:
```bash
pip install -r requirements.txt
```

## Lisensi

Proyek ini untuk keperluan edukasi dan non-komersial.
