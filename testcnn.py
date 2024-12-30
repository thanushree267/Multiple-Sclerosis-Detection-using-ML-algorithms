import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import cv2

# Load your trained model
model = load_model('ms_detection_model.h5')

# Function to preprocess and predict on a single image
def predict_single_image(image_path, model, target_size):
    # Load the image in grayscale mode (single channel)
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    
    # Convert image to array and normalize
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class
    confidence = np.max(predictions)  # Get the confidence score

    return predicted_class, confidence

# Specify the image path and target size
image_path = r"C:\Users\Sharon Zachariah\OneDrive\Desktop\multiple sclerosis\Multiple Sclerosis\MS-Axial\MS-A (11).png"
target_size = (128, 128)  # Use the input size (128, 128) as used during training

# Run prediction
predicted_class, confidence = predict_single_image(image_path, model, target_size)

# Display the result
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence * 100:.2f}%")

# Optional: Display the image
img = load_img(image_path, target_size=target_size, color_mode='grayscale')  # Load as grayscale for display
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {predicted_class}, Confidence: {confidence * 100:.2f}%")
plt.axis('off')
plt.show()
