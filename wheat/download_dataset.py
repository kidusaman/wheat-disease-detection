import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

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

# Load dataset with 3 classes: Healthy (0), Mild Rust (1), Severe Rust (2)
def load_dataset():
    healthy_images, healthy_labels = load_data('dataset/healthy', 0)
    mild_rust_images, mild_rust_labels = load_data('dataset/mild_rust', 1)
    severe_rust_images, severe_rust_labels = load_data('dataset/severe_rust', 2)

    # Combine all classes
    X = np.array(healthy_images + mild_rust_images + severe_rust_images)
    y = np.array(healthy_labels + mild_rust_labels + severe_rust_labels)

    # Split into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
