import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.tensorflow
import json
import os

# **Enable GPU Memory Growth (Optional but Recommended)**
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable dynamic memory allocation
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Using CPU.")

# **Model Parameters**
img_height, img_width = 224, 224
batch_size = 32  # Adjust based on GPU memory capacity
epochs = 50
learning_rate = 1e-4  # Lower learning rate for fine-tuning

# **Data Paths**
train_data_dir = 'D:\\Projekty\\plants_class_ml\\data\\train'
val_data_dir = 'D:\\Projekty\\plants_class_ml\\data\\val'

# **Data Generators with Augmentation**
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,       # Increased rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],  # Brightness variation
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# **Data Generators**
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

# **Class Indices Mapping**
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

# **Save Class Labels for Future Use**
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)

# **Load Pre-trained MobileNetV2 Base Model**
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# **Freeze Layers Except the Last 50**
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# **Define the Model Using the Functional API**
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
outputs = layers.Dense(train_generator.num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# **Compile the Model**
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# **Callbacks**
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
checkpoint_path = 'D:\\Projekty\\plants_class_ml\\model\\house_plant_model_final.keras'
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True
)

# **Ensure the Model Directory Exists**
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# **Train the Model**
with mlflow.start_run():
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint]
    )
    # **Log the Model with MLflow**
    mlflow.tensorflow.log_model(model, "model")

# **Save the Model Locally**
model.save(checkpoint_path)

# **Display Class Labels**
print("Class labels (plants):")
print(class_labels)
