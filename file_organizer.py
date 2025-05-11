"""
nekess les fichiers qui changent les fichier en csv et annotate
et ensuite de faire le dossier SisFall_dataset_csv_annotated_files
noublie pas de mettre la dataset originale ici aussi
"""
import os
import shutil

# Define the source folder and target folders
source_folder = "SisFall_dataset_csv_annotated_files"
fall_folder = "SisFall_split/Fall"
not_fall_folder = "SisFall_split/Not_Fall"

# Create target directories if they don't exist
os.makedirs(fall_folder, exist_ok=True)
os.makedirs(not_fall_folder, exist_ok=True)

# Iterate through files in the source folder
for filename in os.listdir(source_folder):
    # Full path of the file
    source_path = os.path.join(source_folder, filename)
    
    # Skip if it's not a file
    if not os.path.isfile(source_path):
        continue

    # Check if the file starts with 'f'
    if filename.lower().startswith('f'):
        target_folder = fall_folder
    else:
        target_folder = not_fall_folder
    
    # Destination path
    destination_path = os.path.join(target_folder, filename)
    
    # Move the file
    shutil.move(source_path, destination_path)
    print(f"Moved {filename} to {target_folder}")
