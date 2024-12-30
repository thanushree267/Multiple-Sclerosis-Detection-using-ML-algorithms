import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the pre-trained U-Net model
unet_model = load_model('unet_model.h5')  # Replace with the correct path to your trained U-Net model

# Function to preprocess the image, predict the segmentation, and display with classification % 
def preprocess_predict_and_display(image_path, model, target_size=(128, 128)):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image {image_path} could not be loaded.")
        return

    # Save original shape for later resizing
    original_shape = img.shape
    
    # Resize the image for prediction
    img_resized = cv2.resize(img, target_size) / 255.0  # Normalize
    img_resized = img_resized.reshape(1, target_size[0], target_size[1], 1)  # Reshape for model input

    # Predict the segmentation
    prediction = model.predict(img_resized)
    
    # Get the class with the highest probability (MS or Control)
    predicted_class = np.argmax(prediction, axis=-1)[0, :, :]
    
    # Create a mask where the predicted class is 1 (MS)
    mask = np.zeros_like(predicted_class, dtype=np.uint8)
    mask[predicted_class == 1] = 255  # Set MS pixels to white in the mask

    # Resize the mask to the original image size
    mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))  # Resize mask to match original image size

    # Convert the original image to BGR for color overlay
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Overlay the mask on the original image (highlight MS areas in red)
    overlay[mask_resized == 255] = [0, 0, 255]  # Highlight MS areas in red
    
    # Calculate the classification probabilities (MS vs Control)
    ms_prob = prediction[0, :, :, 1].mean()  # Mean probability for MS
    control_prob = prediction[0, :, :, 0].mean()  # Mean probability for Control
    
    # Display the original image with overlaid mask
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Original Image with Predicted Mask\nMS: {ms_prob*100:.2f}% | Control: {control_prob*100:.2f}%")
    plt.axis('off')
    plt.show()

# Path to the single image you want to test
single_image_path = r"C:\Users\Sharon Zachariah\OneDrive\Desktop\multiple sclerosis\Multiple Sclerosis\MS-Axial\MS-A (14).png"  # Replace with your image path

# Run the function to predict and display the result
preprocess_predict_and_display(single_image_path, unet_model)
