import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving the model

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

# Load and preprocess data
def load_data():
    images = []
    labels = []
    for label, path in data_paths.items():
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize for consistency
                images.append(img.flatten())      # Flatten to 1D for ML algorithms
                labels.append(labels_dict[label])
    return np.array(images), np.array(labels)

# Load data
images, labels = load_data()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Option 1: Train with Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")

# Save the Random Forest model
joblib.dump(rf_model, "rf_model_mri.pkl")

# Option 2: Train with SVM
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")

# Save the SVM model
joblib.dump(svm_model, "svm_model_mri.pkl")
