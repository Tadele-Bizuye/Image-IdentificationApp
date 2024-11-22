import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 2. Data Collection and Loading
# Steps:

# - Download the dataset from Kaggle Intel Image Classification.
# - Unzip the dataset and structure folders if needed.
# - Load images into NumPy arrays using OpenCV.
# Define the dataset paths
dataset_path = "C:/Users/TadeleBizuye/Data"
dataset_path_train = os.path.join(dataset_path, "seg_train")
dataset_path_test = os.path.join(dataset_path, "seg_test")
dataset_path_pred = os.path.join(dataset_path, "seg_pred")

# Load and preprocess images function
def load_images(folder_path, img_size=(150, 150)):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))  # Sorted for consistent label ordering
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):  # Skip if not a directory
            continue
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load training and validation images
train_images, train_labels = load_images(dataset_path_train)
val_images, val_labels = load_images(dataset_path_test)

# Check data shapes to ensure proper loading
print(f"Training Data: {train_images.shape}, Training Labels: {train_labels.shape}")
print(f"Validation Data: {val_images.shape}, Validation Labels: {val_labels.shape}")


# 3. Data Preprocessing
# Steps:

# - Resize all images to 150x150.
# - Normalize pixel values to [0, 1].
# - Augment data using rotation, flipping, zooming, etc

# Normalize data
train_images = train_images / 255.0
val_images = val_images / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(train_images)

# 4. Building the CNN Model
# Steps:

# - Design a CNN with TensorFlow/Keras.
# - Use layers like Conv2D, MaxPooling2D, Dense, Dropout.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

model = Sequential([
    Input(shape=(150, 150, 3)),  # Define input shape here
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Model Training and Evaluation
# Steps:

# - Train the model using fit with training and validation data.
# - Visualize metrics using Matplotlib.

history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    validation_data=(val_images, val_labels),
                    epochs=20)

# Plot metrics
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# 6. Model Optimization
# Steps:

# - Add Dropout, Batch Normalization, and Learning Rate Scheduling to improve performance.
# - Fine-tune hyperparameters like the number of layers, learning rate, etc

from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
model.fit(datagen.flow(train_images, train_labels, batch_size=32),
          validation_data=(val_images, val_labels),
          epochs=20,
          callbacks=[lr_scheduler])

# 7. Model Deployment

model.save("image_classifier.keras")

# Streamlit App

import streamlit as st
from tensorflow.keras.models import load_model

model = load_model("image_classifier.keras")

def predict_image(image):
    img = cv2.resize(image, (150, 150)) / 255.0
    img = img[np.newaxis, ...]
    predictions = model.predict(img)
    return predictions.argmax()

st.title("Image Classification App")
st.write("Upload an image to classify it into one of the categories: buildings, forest, glacier, mountain, sea, street.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file:
    image = cv2.imread(uploaded_file)
    st.image(image)
    class_idx = predict_image(image)
    st.write(f"Predicted Class: {class_idx}")