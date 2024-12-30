import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Define the paths for each dataset category
data_paths = {
    "Control-Axial": "Multiple Sclerosis/Control-Axial",
    "Control-Sagittal": "Multiple Sclerosis/Control-Sagittal",
    "MS-Axial": "Multiple Sclerosis/MS-Axial",
    "MS-Sagittal": "Multiple Sclerosis/MS-Sagittal",
}

# Assign labels to each category: 0 for Control, 1 for MS
labels_dict = {
    "Control-Axial": 0,
    "Control-Sagittal": 0,
    "MS-Axial": 1,
    "MS-Sagittal": 1,
}

# Load and preprocess images
def load_data():
    images = []
    labels = []
    for label, path in data_paths.items():
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Please check the path.")
            continue
            
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Image {img_path} could not be loaded.")
                continue
            
            # Resize image and flatten
            img_resized = cv2.resize(img, (128, 128))
            images.append(img_resized.flatten())
            labels.append(labels_dict[label])

    if len(images) == 0:
        raise ValueError("No images were loaded. Please check the image paths and formats.")
    
    return np.array(images), np.array(labels)

try:
    # Load images and labels
    images, labels = load_data()

    # Check the shape of the data
    print("Data loaded successfully.")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Initialize and train the Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Make predictions
    y_pred = gb_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the trained model
    model_filename = 'gradient_boosting_model.pkl'
    joblib.dump(gb_model, model_filename)
    print(f"Model saved as {model_filename}")

except Exception as e:
    print(f"An error occurred: {e}")
