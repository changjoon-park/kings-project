# Python code implementation for embedding a larger set of applications into a vector space with binary vectors

# Define a larger set of all possible features across all applications using real feature names
all_features = [
    'SEND_SMS', 'ACCESS_FINE_LOCATION', 'READ_CONTACTS', 'WRITE_EXTERNAL_STORAGE',
    'GET_ACCOUNTS', 'RECEIVE_SMS', 'READ_SMS', 'USE_CAMERA'
]

# Define more applications with their respective features as dictionaries
applications = {
    'App A': {'SEND_SMS', 'READ_CONTACTS'},
    'App B': {'ACCESS_FINE_LOCATION'},
    'App C': {'SEND_SMS', 'ACCESS_FINE_LOCATION', 'READ_CONTACTS'},
    'App D': {'WRITE_EXTERNAL_STORAGE', 'GET_ACCOUNTS'},
    'App E': {'RECEIVE_SMS', 'READ_SMS'},
    'App F': {'USE_CAMERA', 'ACCESS_FINE_LOCATION', 'GET_ACCOUNTS'},
    'App G': {'SEND_SMS', 'RECEIVE_SMS', 'USE_CAMERA'}
}

# Define the indicator function I(x, s)
def indicator_function(app_features, feature):
    return (1, feature) if feature in app_features else (0, feature)

# Implement the embedding function phi(x) to create binary vector representations
def embed_app_in_vector_space(app_features, all_possible_features):
    # Use the indicator function to create a binary vector
    return [indicator_function(app_features, feature) for feature in all_possible_features]

# Embed each application into the vector space using the defined functions
embedded_applications = {app_name: embed_app_in_vector_space(features, all_features)
                         for app_name, features in applications.items()}

print("[All features]")
for feature in all_features:
    print(f"  - {feature}")


print("\n[App: Vector - Feature(present)]")
for app, indicator in embedded_applications.items():
    present = []
    vector = []
    for element, feature in indicator:
        vector.append(element)
        if element == 1:
            present.append(feature)
    print(f"{app}: {vector} - {present}")
