import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
import mlflow
import mlflow.tensorflow
from config import Config

def main():
    # Konfiguracja augmentacji danych
    # Augmentacja danych to technika polegająca na generowaniu nowych danych treningowych
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Wczytanie danych
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

    # Inicjalizacja modelu
    model = create_model()

    # Kompilacja modelu
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    # �ledzenie eksperyment�w z MLflow
    mlflow.set_experiment("house_plant_classification")
    
    with mlflow.start_run():
        history = model.fit(
            train_generator,
            epochs=Config.EPOCHS,
            validation_data=val_generator
        )
        # Logowanie modelu
        mlflow.tensorflow.log_model(model, "model")

if __name__ == "__main__":
    main()
