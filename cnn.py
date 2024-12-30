import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to datasets
data_paths = {
    "Control-Axial": "Multiple Sclerosis\Control-Axial",
    "Control-Sagittal": "Multiple Sclerosis\Control-Sagittal",
    "MS-Axial": "Multiple Sclerosis\MS-Axial",
    "MS-Sagittal": "Multiple Sclerosis\MS-Sagittal",
}

# Labels: 0 for Control, 1 for MS
labels_dict = {
    "Control-Axial": 0,
    "Control-Sagittal": 0,
    "MS-Axial": 1,
    "MS-Sagittal": 1,
}

# Load images and labels
def load_data():
    images = []
    labels = []
    for label, path in data_paths.items():
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize for consistency
                images.append(img)
                labels.append(labels_dict[label])
    images = np.array(images).reshape(-1, 128, 128, 1)  # Reshape for CNN
    labels = np.array(labels)
    return images, labels

# Load data
images, labels = load_data()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to categorical
y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=1, validation_data=(X_test, y_test))
from sklearn.metrics import classification_report
import numpy as np

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

# Generate classification report
report = classification_report(y_test_classes, y_pred_classes, target_names=['Control', 'MS'])
print("Classification Report:")
print(report)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save model
model.save("ms_detection_model.h5")