import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset path
data_dir = "./wheat_data"  # Make sure this matches your actual dataset folder

# Define class labels
classes = ["healthy", "stripe_rust", "septoria"]

# Count images per category
image_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}
print("Image Counts:", image_counts)

# Display some images
def show_sample_images():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, cls in enumerate(classes):
        img_path = os.path.join(data_dir, cls, os.listdir(os.path.join(data_dir, cls))[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(cls)
        axes[i].axis("off")
    plt.show()

show_sample_images()

# Data Preprocessing
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,  # 80% training, 20% validation
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)