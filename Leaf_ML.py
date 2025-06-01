import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import GridSearchCV

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define the base path to the "Leaf" folder
base_path = '/content/drive/MyDrive/Leaf'

leaf_dir = os.path.join(base_path, 'Leaf')
non_leaf_dir = os.path.join(base_path, 'Non_Leaf')

# Function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize to 128x128

    # Statistical Features
    mean = np.mean(image)
    std = np.std(image)
    skewness = np.mean((image - mean) ** 3) / (std ** 3)
    kurtosis = np.mean((image - mean) ** 4) / (std ** 4)

    # Texture Features using GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return [mean, std, skewness, kurtosis, contrast, homogeneity, energy, correlation]  # Add more features if needed

# Function to load the dataset
def load_data(leaf_dir, non_leaf_dir):
    features = []
    labels = []

    # Load leaf images
    for img_name in os.listdir(leaf_dir):
        img_path = os.path.join(leaf_dir, img_name)
        features.append(extract_features(img_path))
        labels.append(1)  # Label 1 for leaf

    # Load non-leaf images
    for img_name in os.listdir(non_leaf_dir):
        img_path = os.path.join(non_leaf_dir, img_name)
        features.append(extract_features(img_path))
        labels.append(0)  # Label 0 for non-leaf

    return np.array(features), np.array(labels)


# Load data
X, y = load_data(leaf_dir, non_leaf_dir)

# Scale the features before applying feature selection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SelectKBest for feature selection (choose top k features)
k_best = SelectKBest(f_classif, k=6)  # You can change k to any value
X_selected = k_best.fit_transform(X_scaled, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


import random
# Function to get a random image from a directory
def get_random_image_from_class(class_dir):
    image_files = os.listdir(class_dir)
    random_image = random.choice(image_files)
    image_path = os.path.join(class_dir, random_image)
    return cv2.imread(image_path)

# Get random images from both classes
leaf_image = get_random_image_from_class(leaf_dir)
non_leaf_image = get_random_image_from_class(non_leaf_dir)

# Convert images to grayscale and resize them
leaf_image_gray = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)
non_leaf_image_gray = cv2.cvtColor(non_leaf_image, cv2.COLOR_BGR2GRAY)

leaf_image_resized = cv2.resize(leaf_image_gray, (128, 128))
non_leaf_image_resized = cv2.resize(non_leaf_image_gray, (128, 128))

# Plot the original and preprocessed images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Display the original images
axes[0, 0].imshow(cv2.cvtColor(leaf_image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Leaf - Original")
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(non_leaf_image, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Non-Leaf - Original")
axes[0, 1].axis('off')

# Display the preprocessed images (grayscale and resized)
axes[1, 0].imshow(leaf_image_resized, cmap='gray')
axes[1, 0].set_title("Leaf - Preprocessed")
axes[1, 0].axis('off')

axes[1, 1].imshow(non_leaf_image_resized, cmap='gray')
axes[1, 1].set_title("Non-Leaf - Preprocessed")
axes[1, 1].axis('off')

plt.show()
# Define parameter grids for each model
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Initialize classifiers
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)

# Initialize GridSearchCV for each model
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)

# Train models using GridSearchCV
models = {'KNN': grid_knn, 'Random Forest': grid_rf, 'SVM': grid_svm}
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    print(f"Training {model_name} with GridSearchCV...")
    model.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = model.best_estimator_
    print(f"Best parameters for {model_name}: {model.best_params_}")

    # Evaluate the best model
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = best_estimator


# Visualize the results of the best model
if best_model is not None:
    print(f"Best model: {best_model}")
    y_pred = best_model.predict(X_test)

    # Plot clearer confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Leaf', 'Leaf'],
                yticklabels=['Non-Leaf', 'Leaf'])
    plt.title('Confusion Matrix (Best Model)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.show()