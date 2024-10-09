import os
from sklearn.model_selection import train_test_split
import shutil, yaml

# Path to your dataset where each label is a directory
dataset_dir = "justinsdeeplearningutilities/dataset"
train_dir = "yolodataset/train"
val_dir = "yolodataset/val"

# Get class directories (label names)
class_dirs = os.listdir(dataset_dir)


# Function to move files
def move_files(file_list, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for file_name in file_list:
        # Move image file to the correct directory
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


# Loop through each class and split its images
for class_name in class_dirs:
    class_path = os.path.join(dataset_dir, class_name)

    if os.path.isdir(class_path):
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

        # Create train/val split for this class
        train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

        # Create target directories for train and val for each class
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)

        # Move the files
        move_files(train_files, class_path, class_train_dir)
        move_files(val_files, class_path, class_val_dir)


train_path = train_dir
val_path = val_dir

# Use the class directory names as the class labels
class_names = class_dirs
num_classes = len(class_names)

# YAML content
data = {
    'train': train_path,
    'val': val_path,
    'nc': num_classes,
    'names': class_names
}

# Save to data.yaml
with open('data.yaml', 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)