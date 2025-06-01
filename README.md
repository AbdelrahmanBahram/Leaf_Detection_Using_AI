🌿 Leaf Detection AI Project
Project Description
This project focuses on developing robust image classification models to distinguish between "leaf" and "non-leaf" images. The primary objective is to create an accurate and efficient system capable of serving as a crucial pre-filter in agricultural applications, such as plant disease detection pipelines. Both traditional Machine Learning (ML) and Deep Learning (DL) methodologies have been explored and implemented to address this binary classification problem.

Motivation and Use Case
Accurate leaf detection is a foundational step in many plant-related AI applications, including automated monitoring, health assessment, and precision agriculture. By reliably identifying leaf regions, subsequent, more complex analyses (e.g., disease diagnosis) can be focused and made more efficient. This project aims to provide a reliable and potentially edge-device-friendly solution for this initial classification.

Dataset Overview
The models were trained and evaluated on a combined dataset comprising:

Leaf Images: Sourced from the PlantVillage dataset on Kaggle, including approximately 6,000 images across various healthy and diseased leaf classes.
Non-Leaf Images: Consisting of approximately 5,000 random images from the COCO Dataset, featuring diverse non-leaf objects such as soil, tools, human faces, flowers, and other common scenes.
Images in the dataset are in PNG and JPG formats with varying resolutions.

Machine Learning Approach
Intuition
The ML approach relies on extracting meaningful, hand-crafted features from images that describe characteristics unique to leaves (e.g., shape, texture, color patterns) and then training a classifier on these features. The model learns to differentiate between "leaf" and "non-leaf" based on these extracted properties.

Process
Data Preprocessing:
Images were loaded, converted to grayscale, and uniformly resized to 128×128 pixels.
The dataset was split into 80% for training and 20% for testing.
Feature Extraction: A comprehensive set of features was extracted from each preprocessed image:
Statistical Features: Mean, Standard Deviation, Skewness, and Kurtosis of pixel intensities.
Texture Features: Derived from the Gray Level Co-occurrence Matrix (GLCM), including Contrast, Homogeneity, Energy, and Correlation.
Feature Selection:
Features were standardized using StandardScaler.
SelectKBest with ANOVA F-test (f_classif) was applied to select the top 6 most discriminative features from the initial 8 extracted.
Model Training and Hyperparameter Tuning:
Three classical ML classifiers were evaluated:
K-Nearest Neighbors (KNN)
Random Forest
Support Vector Machine (SVM)
GridSearchCV with 5-fold cross-validation was used to optimize hyperparameters for each model, with accuracy as the primary scoring metric.
Evaluation: Model performance was assessed using standard metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.


Deep Learning Approach
Intuition
The DL approach leverages Convolutional Neural Networks (CNNs) to automatically learn hierarchical features directly from raw image data. Instead of manually extracting features, the CNN learns to identify patterns, textures, and shapes that are relevant for distinguishing between leaves and non-leaves. Transfer learning is employed to benefit from pre-trained knowledge on large datasets.

Process
Data Augmentation:
ImageDataGenerator was used to apply real-time data augmentation during training, including horizontal and vertical flips, zoom, and brightness adjustments. This helps in increasing the diversity of the training data and improving model generalization.
Transfer Learning:
A pre-trained MobileNetV3Small model, initialized with imagenet weights, was used as the base network.
The base model's layers were frozen (base_model.trainable = False) to retain the learned features from the large imagenet dataset.
Model Architecture:
Custom layers were added on top of the frozen MobileNetV3Small base:
GlobalAveragePooling2D to reduce dimensionality.
Dropout for regularization.
Dense layers with ReLU activation for feature transformation.
A final Dense layer with softmax activation for binary classification (leaf/non-leaf).
Input images were resized to 224×224 pixels (RGB).
Training and Evaluation:
The model was compiled with the Adam optimizer and binary_crossentropy loss.
Training included EarlyStopping to prevent overfitting and ReduceLROnPlateau for adaptive learning rate adjustment.
Performance was evaluated on both validation and test sets.
Key Results
Validation Accuracy: 100%
Test Accuracy: 100%
Inference Time: Approximately <5 ms per image on a mobile CPU, making it suitable for edge device deployment.
The Deep Learning model achieved exceptional performance, demonstrating its capability as an effective pre-filter.
