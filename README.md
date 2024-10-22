
# Plants Classification using MobileNetV2

This project is focused on building a machine learning model for classifying house plant species using images. The model leverages **MobileNetV2**, a pre-trained convolutional neural network, and fine-tunes it on a dataset of plant images.

## Project Overview

This repository contains the code for training a deep learning model to classify different species of house plants using images. The dataset used for training consists of multiple images representing different classes of house plants. The model is based on **MobileNetV2** and is fine-tuned using transfer learning.

### Key Features
- **MobileNetV2** as the backbone model for feature extraction.
- **Image augmentation** applied during training to improve generalization.
- **Checkpointing** and **early stopping** to save the best model and avoid overfitting.
- Uses **MLflow** to track experiments and model versions.
  
## Dataset

The dataset used in this project comes from Kaggle: [House Plant Species Dataset](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species/data). The dataset contains images of various house plant species organized into subdirectories for each class. It is loaded using `ImageDataGenerator` for both training and validation.

- **Train images**: `data/train/`
- **Validation images**: `data/val/`

## Model Architecture

The model is based on **MobileNetV2**, a lightweight pre-trained model for efficient image classification tasks. A Global Average Pooling layer is added, followed by two Dense layers:

- **Base Model**: MobileNetV2 (pre-trained on ImageNet, `include_top=False`)
- **Additional Layers**: 
  - `GlobalAveragePooling2D`
  - `Dense(1024, activation='relu')`
  - `Dense(num_classes, activation='softmax')` (where `num_classes` is the number of plant species)

## How to Run the Project

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- MLflow
- Required Python packages can be installed using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Training the Model

1. Download the dataset from Kaggle and prepare it in the appropriate directory structure (see **Dataset** section).
2. Adjust the paths to the training and validation directories in the script.
3. Run the training script:
   
```bash
python src/train_1.02.py
```

The script includes **checkpointing** and **early stopping** to save the best model during training.

### Testing the Model

You can test the trained model in two different ways:

#### 1. **Using a locally saved model**:
   You can test a model saved locally (e.g., after training with checkpointing) using the `test_2.0.py` script.

   ```bash
   python src/test_2.0.py --image_path path_to_test_image
   ```

#### 2. **Using a model saved in MLflow**:
   If the model is registered in MLflow, you can use the `test_model.py` script to test it:

   ```bash
   python src/test_model.py --image_path path_to_test_image
   ```

### Model Training Adjustments

During the project, due to certain issues with loading models, I switched to using **checkpointing** to save intermediate models. This was necessary because of the challenges with the model's architecture not being properly built when loading weights directly.

Checkpointing allowed the training to resume from the last saved state, which significantly improved the stability of the model training process.

To use checkpointing, I modified the code to ensure that models are regularly saved and loaded from checkpoint files if necessary.

### MLflow Tracking

This project uses **MLflow** for experiment tracking. You can view the logs and metrics for the model by launching MLflow UI:

```bash
mlflow ui
```

## Example Results

After training, the model should be able to predict the species of house plants from an image. Example predictions are:

```
Predicted class: Aloe Vera
```

