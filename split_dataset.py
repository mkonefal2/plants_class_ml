import os
import zipfile
import random
import shutil

# Ścieżki do plików
zip_path = r'D:\Projekty\plants_class_ml\data\archive.zip'
extract_path = r'D:\Projekty\plants_class_ml\data\extracted'
train_path = r'D:\Projekty\plants_class_ml\data\train'
val_path = r'D:\Projekty\plants_class_ml\data\val'

# Procent danych do walidacji
val_split = 0.2

# Rozpakowywanie pliku ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Lista wszystkich folderów gatunków
categories = os.listdir(os.path.join(extract_path, 'house_plant_species'))

# Tworzenie folderów docelowych, jeśli nie istnieją
for category in categories:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(val_path, category), exist_ok=True)

# Przenoszenie plików do folderów train/val
for category in categories:
    category_path = os.path.join(extract_path, 'house_plant_species', category)
    images = os.listdir(category_path)
    random.shuffle(images)  # Mieszanie danych

    split_point = int(len(images) * (1 - val_split))

    train_images = images[:split_point]
    val_images = images[split_point:]

    # Przenoszenie danych do folderu train
    for img in train_images:
        shutil.move(os.path.join(category_path, img), os.path.join(train_path, category, img))

    # Przenoszenie danych do folderu val
    for img in val_images:
        shutil.move(os.path.join(category_path, img), os.path.join(val_path, category, img))

print("Podział danych zakończony pomyślnie.")
