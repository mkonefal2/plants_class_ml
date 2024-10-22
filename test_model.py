import mlflow.pyfunc
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Załaduj model z MLflow
model_uri = "models:/house_plant_classification/1"
model = mlflow.pyfunc.load_model(model_uri)

# Tworzenie generatora danych (aby uzyskać mapowanie klas)
train_data_dir = "D:\\Projekty\\plants_class_ml\\data\\train"  # Ustal ścieżkę do swoich danych treningowych
img_height, img_width = 500, 500
batch_size = 32

# Używamy generatora danych, aby uzyskać mapowanie klas
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Mapowanie klas
class_indices = train_generator.class_indices  # Uzyskanie indeksów klas
class_labels = {v: k for k, v in class_indices.items()}  # Odwrócenie mapy, aby uzyskać nazwę na podstawie numeru klasy

# Wczytaj i przetwórz obraz
img_path = "D:\\Projekty\\plants_class_ml\\data\\train\\Aloe Vera\\4.jpg"  # Ścieżka do Twojego obrazu
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalizacja

# Przewidywanie klasy
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Wyświetlenie przewidywanej klasy (nazwa zamiast numeru)
predicted_class_name = class_labels[predicted_class[0]]
print(f'Predicted class name: {predicted_class_name}')

# Opcjonalnie: Wyświetlenie obrazu
plt.imshow(img)
plt.title(f'Predicted class: {predicted_class_name}')
plt.show()
