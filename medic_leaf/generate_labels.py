from torchvision import datasets

# Lokasi dataset
data_dir = 'dataset'  # Sesuaikan dengan lokasi dataset Anda

# Load dataset
dataset = datasets.ImageFolder(data_dir)

# Ambil nama-nama kelas
class_names = dataset.classes

# Simpan ke file labels.txt
with open('labels.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print("Label berhasil disimpan ke labels.txt")
