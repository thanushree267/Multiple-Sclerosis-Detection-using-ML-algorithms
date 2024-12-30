import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Paths to datasets
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

# Load images and masks
def load_data():
    images = []
    masks = []  # Store segmentation masks
    for label, path in data_paths.items():
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize for consistency
                images.append(img)

                # Create a mask for segmentation
                mask = np.zeros((128, 128), dtype=np.uint8)  # Create a blank mask
                mask[img > 0] = labels_dict[label]  # Set mask pixels to label
                masks.append(mask)
    
    images = np.array(images).reshape(-1, 128, 128, 1)  # Reshape for CNN
    masks = np.array(masks).reshape(-1, 128, 128, 1)  # Reshape masks for U-Net
    return images, masks

# Load data
images, masks = load_data()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert masks to categorical format (for segmentation)
y_train = to_categorical(np.squeeze(y_train), num_classes=2)  # Convert masks to categorical format
y_test = to_categorical(np.squeeze(y_test), num_classes=2)    # Convert masks to categorical format

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

# U-Net Model
def unet_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Contracting path
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Expanding path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(2, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Initialize U-Net model
input_shape = (128, 128, 1)
unet = unet_model(input_shape)

# Compile the U-Net model
unet.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the U-Net model
history = unet.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=1, validation_data=(X_test, y_test))
# Predict on the test set
from sklearn.metrics import classification_report
y_pred = unet.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)  # Convert predictions to class labels
y_test_classes = np.argmax(y_test, axis=-1)  # Convert ground truth to class labels

# Flatten the predictions and ground truth for each pixel
y_pred_flat = y_pred_classes.flatten()
y_test_flat = y_test_classes.flatten()

# Generate classification report
report = classification_report(y_test_flat, y_pred_flat, target_names=['Control', 'MS'])
print("Classification Report:")
print(report)
# Evaluate U-Net model
loss, accuracy = unet.evaluate(X_test, y_test)
print(f"U-Net Test Loss: {loss}")
print(f"U-Net Test Accuracy: {accuracy}")
unet.save('unet_model.h5')
print("Model saved as unet_model.h5")