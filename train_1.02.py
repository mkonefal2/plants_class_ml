
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.tensorflow

# Parametry modelu
img_height, img_width = 224, 224
batch_size = 32  # Mniejszy batch size
epochs = 50
learning_rate = 1e-4  # Niższy współczynnik uczenia

# Ścieżki do danych
train_data_dir = 'D:\\Projekty\\plants_class_ml\\data\\train'
val_data_dir = 'D:\\Projekty\\plants_class_ml\\data\\val'

# Przygotowanie generatorów danych z bardziej agresywną augmentacją
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Większa rotacja
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],  # Zmienność jasności
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Generatory danych
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Mapowanie klas na nazwy roślin
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Odwrócenie mapowania

# Ładowanie modelu MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Dodanie własnych warstw na szczycie
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Dopasowanie liczby klas
])

# Odblokowanie większej liczby warstw do fine-tuningu
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Odblokowanie ostatnich 50 warstw
    layer.trainable = False

# Kompilacja modelu z niskim learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Wczesne zatrzymanie i zapis najlepszego modelu
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('D:\\Projekty\\plants_class_ml\\model\\house_plant_model_1.05.keras', monitor='val_loss', save_best_only=True)

# Uruchomienie śledzenia eksperymentu w MLflow
with mlflow.start_run():
    # Trenowanie modelu z wczesnym zatrzymaniem
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint]  # Early stopping i zapis najlepszego modelu
    )
    
    # Logowanie modelu w MLflow
    mlflow.tensorflow.log_model(model, "model")

# Zapisanie modelu lokalnie (opcjonalne)
model.save('D:\\Projekty\\plants_class_ml\\model\\house_plant_model_1.05.keras')

# Wyświetlenie mapowania klas (nazwa -> numer)
print("Class labels (rośliny):")
print(class_labels)
