import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# **Load the Model**
model = tf.keras.models.load_model('D:\\Projekty\\plants_class_ml\\model\\house_plant_model_final.keras')

# **Load Class Labels**
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

# **Path to the Test Image**
img_path = "D:\\Projekty\\plants_class_ml\\data\\test\\AdobeStock_328965612.jpg"

# **Load and Preprocess the Image**
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize

# **Predict the Class**
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_labels[predicted_class_idx]

# **Display the Prediction**
print(f'Predicted class: {predicted_class_name}')
