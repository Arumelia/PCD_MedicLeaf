import streamlit as st
import os
import subprocess

# Fungsi untuk menjalankan generate_labels.py
def generate_labels():
    subprocess.run(["python", "generate_labels.py"])

# Fungsi untuk menjalankan train_model.py
def train_model():
    subprocess.run(["python", "train_model.py"])

# Fungsi untuk menjalankan aplikasi utama
def run_app():
    subprocess.run(["streamlit", "run", "app.py"])

# Buat tombol untuk menjalankan semua langkah
if st.button('Generate Label → Train Model → Run App'):
    st.text('📊 Proses dimulai...')
    
    # Langkah 1: Generate labels
    generate_labels()
    st.text('✔️ Label berhasil dibuat.')

    # Langkah 2: Train model
    train_model()
    st.text('✔️ Model selesai dilatih.')

    # Langkah 3: Run aplikasi
    run_app()
    st.text('🌿 Aplikasi siap dijalankan!')