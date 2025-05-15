import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# === Fungsi Memuat Label dari File ===
def load_labels(path='labels.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# === Konfigurasi Model ===
model = load_model('keras_model.h5')
class_names = load_labels('labels.txt')
IMG_SIZE = (224, 224)

# === Preprocessing dan Prediksi ===
def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_leaf(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]
    label = f"{class_names[class_idx]} ({confidence*100:.2f}%)"
    return label

# === Deteksi dan Bounding Box ===
def detect_leaf_and_classify(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_label = "Tidak Terdeteksi"
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1000:
            roi = frame[y:y+h, x:x+w]
            label = predict_leaf(roi)
            final_label = label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, final_label

# === GUI Modern ===
class MedicLeafApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medic Leaf - Klasifikasi Tanaman Obat")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Gaya modern
        style = ttk.Style()
        style.theme_use("clam")

        # === Frame Tampilan Gambar ===
        self.image_panel = ttk.Label(root)
        self.image_panel.pack(pady=20)

        # === Label Hasil ===
        self.result_label = ttk.Label(root, text="Hasil: -", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        # === Tombol Aksi ===
        button_frame = ttk.Frame(root)
        button_frame.pack(pady=20)

        self.btn_upload = ttk.Button(button_frame, text="Upload Gambar", command=self.upload_image)
        self.btn_upload.grid(row=0, column=0, padx=10)

        self.btn_camera = ttk.Button(button_frame, text="Buka Kamera", command=self.open_camera)
        self.btn_camera.grid(row=0, column=1, padx=10)

    def display_image(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img = img.resize((600, 400))
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_panel.configure(image=imgtk)
        self.image_panel.image = imgtk

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            result, label = detect_leaf_and_classify(image)
            self.display_image(result)
            self.result_label.config(text=f"Hasil: {label}")

    def open_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result, label = detect_leaf_and_classify(frame.copy())
            cv2.imshow('Medic Leaf - Tekan Q untuk keluar', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Jalankan Aplikasi
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicLeafApp(root)
    root.mainloop()
