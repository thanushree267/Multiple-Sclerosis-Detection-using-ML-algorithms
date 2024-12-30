import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt

# Load the saved Gradient Boosting model
model_filename = 'gradient_boosting_model.pkl'
try:
    gb_model = joblib.load(model_filename)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"The model file '{model_filename}' was not found. Make sure the model is trained and saved.")
    exit()

# Function to preprocess, predict, and display the result with confidence level
def preprocess_predict_and_display(image_path, model, target_size=(128, 128)):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image {image_path} could not be loaded.")
        return None, None

    # Resize and flatten image
    img_resized = cv2.resize(img, target_size)
    img_flattened = img_resized.flatten().reshape(1, -1)  # Reshape for model input

    # Make the prediction and get confidence level
    prediction = model.predict(img_flattened)
    confidence = model.predict_proba(img_flattened)

    # Determine predicted class and confidence level
    predicted_class = "MS" if prediction[0] == 1 else "Control"
    confidence_level = confidence[0][prediction[0]]  # Confidence for the predicted class

    # Display the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {predicted_class} \nConfidence: {confidence_level * 100:.2f}%")
    plt.axis('off')
    plt.show()

# Path to the single image you want to test
single_image_path = r"C:\Users\Sharon Zachariah\OneDrive\Desktop\multiple sclerosis\Multiple Sclerosis\MS-Axial\MS-A (13).png" # Replace with your image path

# Run the function to predict and display the result
preprocess_predict_and_display(single_image_path, gb_model)
