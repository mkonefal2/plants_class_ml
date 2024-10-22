import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import Config

# Konfiguracja generatorów
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Ładowanie danych
train_generator = train_datagen.flow_from_directory(
    Config.TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    Config.VAL_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

print(f"Liczba próbek treningowych: {train_generator.samples}")
print(f"Liczba próbek walidacyjnych: {val_generator.samples}")
