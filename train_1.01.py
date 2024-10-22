import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow

# Parametry modelu
img_height, img_width = 224, 224
batch_size = 64  # Zwiększony rozmiar batcha
epochs = 50  # Większa liczba epok

# Ścieżki do danych
train_data_dir = 'D:\\Projekty\\plants_class_ml\\data\\train'
val_data_dir = 'D:\\Projekty\\plants_class_ml\\data\\val'

# Przygotowanie generatorów danych z mniejszą augmentacją
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Zmiana kąta obrotu obrazu o losową wartość z zakresu [-20, 20] stopni dla każdego obrazu w zbiorze danych aby model mógł nauczyć się rozpoznawać obrazy z różnymi kątami obrotu
    width_shift_range=0.1, #  Przesunięcie w poziomie - zmiana położenia obrazu w poziomie dla każdego obrazu w zbiorze danych aby model mógł nauczyć się rozpoznawać obrazy z różnymi położeniami
    height_shift_range=0.1, # Przesunięcie w pionie - zmiana położenia obrazu w pionie dla każdego obrazu w zbiorze danych aby model mógł nauczyć się rozpoznawać obrazy z różnymi położeniami
    zoom_range=0.1, # Przybliżenie - losowe przybliżenie obrazu dla każdego obrazu w zbiorze danych aby model mógł nauczyć się rozpoznawać obrazy z różnymi poziomami przybliżenia
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
class_indices = train_generator.class_indices # Uzyskanie indeksów klas z generatora treningowego (mapowanie nazwa -> numer) 
class_labels = {v: k for k, v in class_indices.items()}  # Odwrócenie mapowania (numer -> nazwa)

# Ładowanie pretrenowanego modelu ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Dodanie własnych warstw na szczycie
model = models.Sequential()
model.add(base_model) # Dodanie modelu bazowego ResNet50 do modelu własnego modelu (transfer learning) 
model.add(layers.GlobalAveragePooling2D()) # Globalne uśrednianie cech - obliczenie średniej z każdej mapy cech dla każdego obrazu w zbiorze danych aby zmniejszyć liczbę parametrów modelu i zapobiec nadmiernemu dopasowaniu
model.add(layers.Dense(1024, activation='relu')) # Dodanie warstwy ukrytej z 1024 neuronami i funkcją aktywacji ReLU (Rectified Linear Unit) 
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))  # Dopasowanie liczby klas

# Odblokowanie ostatnich warstw modelu bazowego do fine-tuningu
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Zablokowanie wcześniejszych warstw
    layer.trainable = False

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ustawienie wczesnego zatrzymania
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Uruchomienie śledzenia eksperymentu w MLflow
with mlflow.start_run():
    # Trenowanie modelu z wczesnym zatrzymaniem
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping]
    )
    
    # Logowanie modelu w MLflow
    mlflow.tensorflow.log_model(model, "model")

# Zapisanie modelu lokalnie (opcjonalne)
model.save('D:\\Projekty\\plants_class_ml\\model\\house_plant_model.h5')

# Wyświetlenie mapowania klas (nazwa -> numer)
print("Class labels (rośliny):")
print(class_labels)
