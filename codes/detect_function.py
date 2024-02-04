# Sample Python code for calculating the decision boundary (hyperplane) in SVM

import numpy as np
from sklearn.svm import SVC

# Define the features for each application (1 for presence, 0 for absence)
# For simplicity, we're using only three features.
features = np.array([
    [1, 0, 1],  # App A has Feature 1 and Feature 3
    [0, 1, 0],  # App B has Feature 2
    [1, 1, 1],  # App C has all features
    # Additional apps can be added here
])

# Define the labels for each application (1 for malicious, 0 for benign)
labels = np.array([1, 0, 1])  # Assuming App A and C are malicious, and App B is benign

# Train the SVM classifier
clf = SVC(kernel='linear')
clf.fit(features, labels)

# Get the weight vector w and the intercept b from the trained model
w = clf.coef_[0]
b = clf.intercept_[0]

# Display the weight vector and the intercept
print(f"Weight vector (w): {w}")
print(f"Intercept (b): {b}")

# Function to classify a new application based on the SVM model
def classify_app(new_app_features, weight_vector, intercept):
    # Calculate the dot product plus the intercept
    decision_value = np.dot(weight_vector, new_app_features) + intercept
    # Return the classification based on the decision value
    return 'malicious' if decision_value > 0 else 'benign'

# Example classification of a new application
new_app = np.array([1, 0, 0])  # New App with Feature 1 only
classification = classify_app(new_app, w, b)
print(f"The new application is classified as: {classification}")

