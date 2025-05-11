import os
from sklearn.model_selection import train_test_split

# Paths to your dataset
fall_files_path = "./SisFall_split/Fall"
non_fall_files_path = "./SisFall_split/Not_Fall"

# Collect all file names
fall_files = [os.path.join(fall_files_path, f) for f in os.listdir(fall_files_path)]
non_fall_files = [os.path.join(non_fall_files_path, f) for f in os.listdir(non_fall_files_path)]

# Split the data into train (80%) and test (20%) while preserving class ratio
fall_train, fall_test = train_test_split(fall_files, test_size=0.2, random_state=42)
non_fall_train, non_fall_test = train_test_split(non_fall_files, test_size=0.2, random_state=42)

# Combine train and test sets for both classes
train_files = fall_train + non_fall_train
test_files = fall_test + non_fall_test

# Shuffle the combined datasets
import random
random.shuffle(train_files)
random.shuffle(test_files)

# Optional: Further split the train set into train and validation sets
train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)  # 10% of total for validation

# Print the sizes
print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Testing files: {len(test_files)}")
