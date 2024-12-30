import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the trained VGG16 model (or ResNet50 model)
model = load_model("vgg16_mri_classification_model.h5")

# Function to preprocess and predict on a single image
def predict_single_image(image_path, model, target_size=(224, 224)):
    # Load the image and convert it to RGB (3 channels)
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = (prediction > 0.5).astype("int32")  # Convert to binary class (0 or 1)

    return predicted_class[0][0], prediction[0][0]  # Class and confidence

# Path to the test image
image_path = r"C:\Users\Sharon Zachariah\OneDrive\Desktop\multiple sclerosis\Multiple Sclerosis\MS-Axial\MS-A (22).png" # Update with your image path

# Predict the image class and confidence
predicted_class, confidence = predict_single_image(image_path, model)

# Print the results
print(f"Predicted Class: {'MS' if predicted_class == 1 else 'Control'}")
print(f"Confidence: {confidence * 100:.2f}%")

# Optionally: Display the image with the prediction
img = load_img(image_path, target_size=(224, 224))
plt.imshow(img)
plt.title(f"Predicted: {'MS' if predicted_class == 1 else 'Control'}\nConfidence: {confidence * 100:.2f}%")
plt.axis('off')
plt.show()
