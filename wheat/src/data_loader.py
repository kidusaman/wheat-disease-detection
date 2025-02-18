import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size
IMG_SIZE = 128

# Function to load and preprocess images
def load_data(image_path, label):
    images = []
    labels = []

    for file in os.listdir(image_path):
        img_path = os.path.join(image_path, file)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable images

        # Resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values (0 to 1)
        img = img / 255.0

        # Store image and label
        images.append(img)
        labels.append(label)

    return images, labels

# Function to load dataset
def load_dataset():
    healthy_images, healthy_labels = load_data('dataset/healthy', 0)  # 0 for healthy
    rust_images, rust_labels = load_data('dataset/rust', 1)  # 1 for rust

    # Combine the two datasets
    X = np.array(healthy_images + rust_images)
    y = np.array(healthy_labels + rust_labels)

    # Split into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to create an augmented data generator
def get_data_generator():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen
