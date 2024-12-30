import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define paths to the data folders
data_paths = {
    "Control-Axial": "Multiple Sclerosis/Control-Axial",
    "Control-Sagittal": "Multiple Sclerosis/Control-Sagittal",
    "MS-Axial": "Multiple Sclerosis/MS-Axial",
    "MS-Sagittal": "Multiple Sclerosis/MS-Sagittal",
}

# Labels: 0 for Control, 1 for MS
labels_dict = {
    "Control-Axial": 0,
    "Control-Sagittal": 0,
    "MS-Axial": 1,
    "MS-Sagittal": 1,
}

# Load and preprocess data
def load_data():
    images = []
    labels = []
    for label, path in data_paths.items():
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Resize to 224x224 for VGG/ResNet
                img = np.stack((img,)*3, axis=-1)  # Convert to 3-channel
                images.append(img)
                labels.append(labels_dict[label])
    return np.array(images), np.array(labels)

# Load data
images, labels = load_data()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define a function to build a model using transfer learning
def build_model(base_model_name='VGG16'):
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Invalid base model name. Choose either 'VGG16' or 'ResNet50'.")

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

    # Define the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Build models for VGG16 and ResNet50
vgg16_model = build_model('VGG16')
resnet50_model = build_model('ResNet50')

# Compile the models
vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
resnet50_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation for training
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Train VGG16 model
print("Training VGG16 model...")
vgg16_model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                validation_data=(X_test, y_test), epochs=10, verbose=1)

# Evaluate VGG16 model
vgg16_predictions = (vgg16_model.predict(X_test) > 0.5).astype("int32")
print("VGG16 Classification Report:")
print(classification_report(y_test, vgg16_predictions))
print(f"VGG16 Accuracy: {accuracy_score(y_test, vgg16_predictions)}")

# Save VGG16 model
vgg16_model.save("vgg16_mri_classification_model.h5")

