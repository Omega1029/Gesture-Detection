import os
import shutil

# Get all files in dataset
DIR = 'dataset'  # Root dataset directory containing images and labels
SPLIT_SIZE = 0.15  # 15% for validation

# Paths to store train/validation images and labels
TRAIN_IMAGE_DIR = 'yolo8dataset/images/train'
VAL_IMAGE_DIR = 'yolo8dataset/images/val'
TRAIN_LABEL_DIR = 'yolo8dataset/labels/train'
VAL_LABEL_DIR = 'yolo8dataset/labels/val'

# Ensure train and validation directories exist for both images and labels
os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

# Loop over the dataset and split into train/validation sets
dataset = os.listdir(DIR)
for label in dataset[:15]:  # Assuming the first 15 directories are label folders
    label_dir = os.path.join(DIR, label)
    images = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]  # List image files only
    size = len(images)

    train_count = int(size * (1 - SPLIT_SIZE))  # 85% for training
    val_count = size - train_count  # 15% for validation

    # Copy images and labels to train/val folders
    for i, image in enumerate(images):
        # Paths to the image and label
        src_image_path = os.path.join(label_dir, image)
        src_label_path = os.path.join(label_dir, image.replace('.jpg', '.txt'))  # Assuming labels are .txt files

        if i < train_count:
            dst_image_path = os.path.join(TRAIN_IMAGE_DIR, image)
            dst_label_path = os.path.join(TRAIN_LABEL_DIR, image.replace('.jpg', '.txt'))
        else:
            dst_image_path = os.path.join(VAL_IMAGE_DIR, image)
            dst_label_path = os.path.join(VAL_LABEL_DIR, image.replace('.jpg', '.txt'))

        # Copy the image
        shutil.copy(src_image_path, dst_image_path)

        # Copy the corresponding label if it exists
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)

    print(f"Label: {label}, Train images: {train_count}, Validation images: {val_count}")
