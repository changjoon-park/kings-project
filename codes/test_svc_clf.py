import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Define a larger set of all possible features across all applications using real feature names
all_features = [
    "SEND_SMS",
    "ACCESS_FINE_LOCATION",
    "READ_CONTACTS",
    "WRITE_EXTERNAL_STORAGE",
    "GET_ACCOUNTS",
    "RECEIVE_SMS",
    "READ_SMS",
    "USE_CAMERA",
]

# Define more applications with their respective features as dictionaries
applications = {
    "App A": {"SEND_SMS", "READ_CONTACTS"},
    "App B": {"ACCESS_FINE_LOCATION"},
    "App C": {"SEND_SMS", "ACCESS_FINE_LOCATION", "READ_CONTACTS"},
    "App D": {"WRITE_EXTERNAL_STORAGE", "GET_ACCOUNTS"},
    "App E": {"RECEIVE_SMS", "READ_SMS"},
    "App F": {"USE_CAMERA", "ACCESS_FINE_LOCATION", "GET_ACCOUNTS"},
    "App G": {"SEND_SMS", "RECEIVE_SMS", "USE_CAMERA"},
}


# Define the indicator function I(x, s)
def indicator_function(app_features, feature):
    return 1 if feature in app_features else 0


# Implement the embedding function phi(x) to create binary vector representations
def embed_app_in_vector_space(app_features, all_possible_features):
    # Use the indicator function to create a binary vector
    return [
        indicator_function(app_features, feature) for feature in all_possible_features
    ]


# Embed each application into the vector space using the defined functions
embedded_applications = {
    app_name: embed_app_in_vector_space(features, all_features)
    for app_name, features in applications.items()
}

# Create a feature matrix and labels for SVM training
# Here, we'll consider that the first half of the apps are malicious and the second half benign
feature_matrix = np.array(list(embedded_applications.values()))
labels = np.array(
    [1 if i < len(applications) / 2 else 0 for i in range(len(applications))]
)

# Train the SVM classifier
clf = SVC(kernel="linear", C=1.0)
clf.fit(feature_matrix, labels)

# Get the weight vector w and the intercept b from the trained model
w = clf.coef_[0]
b = clf.intercept_[0]

"""
# ! Function to classify a new application based on the SVM model (using the dot product manually)
def classify_app(new_app_features, weight_vector, intercept):
    decision_value = np.dot(weight_vector, new_app_features) + intercept
    return 'malicious' if decision_value > 0 else 'benign'
"""


# ! Function to classify a new application based on the SVM model (using the model's predict function)
def classify_app(new_app_features, model):
    # The model's predict function returns the classification
    return model.predict([new_app_features])


# Pick random features for a new app from 'all_features' list
new_app_features = random.sample(all_features, k=random.randint(1, len(all_features)))

# Convert the list of random features into a binary vector
new_app_vector = embed_app_in_vector_space(set(new_app_features), all_features)

# Classify the new app with random features
new_app_classification = classify_app(new_app_vector, clf)

# Display the results
print(f"Randomly selected features for the new app: {new_app_features}")
print(f"Binary vector for the new app: {new_app_vector}")
print(f"Weight vector (w): {w}")
print(f"Intercept (b): {b}")
print(
    f"Classification for the new app: {'Malicious' if new_app_classification[0] == 1 else 'Benign'}"
)

# ? Visualize the decision boundary

# Perform PCA for dimensionality reduction for visualization purposes
pca = PCA(n_components=2)  # reduce to two dimensions
reduced_features = pca.fit_transform(feature_matrix)

# Train a new SVM on the reduced features for visualization
clf_vis = SVC(kernel="linear", C=1.0)
clf_vis.fit(reduced_features, labels)

# Plot the points
plt.scatter(
    reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="bwr", alpha=0.5
)

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_vis.decision_function(xy).reshape(XX.shape)
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)

# Add labels and title
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.title("Application Classification with SVM")
plt.show()
