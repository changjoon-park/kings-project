import json
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

def load_X(dataset_path, reduced=False):
    print("Loading data (feature representation X, and feature names)...")
    # Load the reduced 10k features
    if reduced:
        with open('{}/X-10k.p'.format(dataset_path), 'rb') as f:
            X = pickle.load(f)
        with open('{}/f-10k.p'.format(dataset_path), 'rb') as f:
            feature_names = pickle.load(f)
    else:
        with open('{}/X.json'.format(dataset_path), 'r') as f:
            X = json.load(f)

        # Convert to numpy array and get feature names
        vec = DictVectorizer()
        X = vec.fit_transform(X).astype("float32")
        feature_names = vec.get_feature_names_out()

    return X, feature_names


def load_y(dataset_path):
    print('Loading labels...')
    with open('{}y.json'.format(dataset_path), 'rt') as f:
        y = json.load(f)
    y = np.asarray(y)
    return y


def load_metadata(dataset_path):
    print('Loading metadata...')
    with open('{}meta.json'.format(dataset_path), 'rt') as f:
        metadata = json.load(f)
    return metadata


if __name__ == '__main__':

    dataset_path = './extended-features/'

    # Flag reduced=True loads only the top 10k features; reduced=False loads the entire dataset of featues
    X, feature_names = load_X(dataset_path, reduced=True)
    y = load_y(dataset_path)
    metadata = load_metadata(dataset_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Machine learning model: Linear SVM (liblinear implementation)
    model = LinearSVC(C=1, dual=True, max_iter=10000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)

    # Print the F1 score
    print("F1 score:", f1)

