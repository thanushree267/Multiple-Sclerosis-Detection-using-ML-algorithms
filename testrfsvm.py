import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained models
rf_model = joblib.load("rf_model_mri.pkl")
svm_model = joblib.load("svm_model_mri.pkl")

# Function to preprocess, predict, and display results for a single image
def preprocess_predict_and_display(image_path, rf_model, svm_model, target_size=(128, 128)):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image {image_path} could not be loaded.")
        return

    # Resize and normalize the image
    img_resized = cv2.resize(img, target_size) / 255.0  # Normalize
    img_flattened = img_resized.flatten().reshape(1, -1)  # Flatten and reshape for model input

    # Random Forest Prediction and Confidence
    rf_prediction = rf_model.predict(img_flattened)
    rf_confidence = rf_model.predict_proba(img_flattened)[0][rf_prediction[0]]
    rf_class = "MS" if rf_prediction[0] == 1 else "Control"

    # SVM Prediction and Confidence
    svm_prediction = svm_model.predict(img_flattened)
    svm_confidence = svm_model.predict_proba(img_flattened)[0][svm_prediction[0]]
    svm_class = "MS" if svm_prediction[0] == 1 else "Control"

    # Display the image with prediction results
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"RF Prediction: {rf_class} (Confidence: {rf_confidence*100:.2f}%)\n"
              f"SVM Prediction: {svm_class} (Confidence: {svm_confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Path to the single image you want to test
single_image_path = r"C:\Users\Sharon Zachariah\OneDrive\Desktop\multiple sclerosis\Multiple Sclerosis\MS-Axial\MS-A (14).png"  # Replace with your image path

# Run the function to predict and display the result
preprocess_predict_and_display(single_image_path, rf_model, svm_model)
