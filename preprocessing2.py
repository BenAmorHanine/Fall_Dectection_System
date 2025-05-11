import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import random

# Paths to your dataset
fall_files_path = "./SisFall_recleaned/Fall"
non_fall_files_path = "./SisFall_recleaned/Not_Fall"

# Collect all file names
fall_files = [os.path.join(fall_files_path, f) for f in os.listdir(fall_files_path)]
non_fall_files = [os.path.join(non_fall_files_path, f) for f in os.listdir(non_fall_files_path)]

print(f"Fall files: {len(fall_files)}")
print(f"Not Fall files: {len(non_fall_files)}")

# Combine fall and non-fall files with labels
all_files = fall_files + non_fall_files
labels = [1] * len(fall_files) + [0] * len(non_fall_files)

# Perform stratified split
train_files, test_files, train_labels, test_labels = train_test_split(
    all_files, labels, test_size=0.2, random_state=42, stratify=labels
)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.125, random_state=42, stratify=train_labels
)

# Check dataset distribution for debugging
print(f"Training label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
print(f"Validation label distribution: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
print(f"Testing label distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

# Print the sizes
print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Testing files: {len(test_files)}")

# Function to load the data
def load_data(file_list, label_map):
    data = []
    labels = []
    print('Processing files...')
    for file in file_list:
        # Read the file with headers
        try:
            try:
                df = pd.read_csv(file, header=0)  # Read the file with headers
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
        
            # Extract features and labels
            features = df.iloc[:, :-1]  # All columns except the last
            label = df.iloc[:, -1]  # The last column
            
            # Print the unique values of the label column
            #print(f"Processing file {file}, label values: {label.unique()}")  # Debugging
            
            # Skip rows where label is 'label' (incorrect label value)
            if 'label' in label.values:
                print(f"Skipping file {file} due to incorrect label values.")
                continue

            # Map labels (ensure only 'f' and 'd' are used)
            mapped_label = label.map(label_map)
            
            # Check for unexpected labels after mapping
            if mapped_label.isnull().any():
                print(f"Unexpected label values in file {file}: {label.unique()}")
                continue

            # Append features and labels
            data.append(features)
            labels.append(mapped_label)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if len(data) == 0:
        print("No data loaded. Please check the files or label mapping.")
        return None, None

        # Combine features and labels
    data = pd.concat(data, ignore_index=True)
    labels = pd.concat(labels, ignore_index=True)
    

        
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return data, labels

# Function to extract basic features from the data
def extract_features(data):
    features = data.describe().T[['mean', 'std', 'min', 'max']]
    
    # Print the shape of the extracted features
    print(f"Extracted features shape: {features.shape}")
    #print("First few rows of extracted features:\n", features.head())
    
    return features

# Preprocessing function
def preprocess_data(train_files, val_files, test_files):
    # Define a consistent label mapping
    label_map = {'f': 1, 'd': 0}

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Testing files: {len(test_files)}")

    # Load the data with the updated label mapping
    train_data, train_labels = load_data(train_files, label_map)
    val_data, val_labels = load_data(val_files, label_map)
    test_data, test_labels = load_data(test_files, label_map)

    # Check combined label distribution
    print(f"Train labels: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"Validation labels: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    print(f"Test labels: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
    
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # Encode labels (if required later, otherwise they are already numeric)
    # Save the preprocessed data
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump((train_data_scaled, val_data_scaled, test_data_scaled, train_labels, val_labels, test_labels), f)
    print("Preprocessed data saved to 'preprocessed_data.pkl'")

    return train_data_scaled, val_data_scaled, test_data_scaled, train_labels, val_labels, test_labels


if __name__ == "__main__":
    # Call preprocess_data with the file paths
    preprocess_data(train_files, val_files, test_files)
