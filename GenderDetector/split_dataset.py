import os
import shutil
import random

# Define the source folder containing "smiling" and "unsmiling" sub-folders
source_folder = "./dataset"

# Define the destination folders
dest_train = "./dataset/train"
dest_val = "./dataset/validation"
dest_test = "./dataset/test"

# Remove existing destination folders if they exist
if os.path.exists(dest_train):
    shutil.rmtree(dest_train)
if os.path.exists(dest_val):
    shutil.rmtree(dest_val)
if os.path.exists(dest_test):
    shutil.rmtree(dest_test)

# Recreate destination folders
os.makedirs(dest_train, exist_ok=True)
os.makedirs(dest_val, exist_ok=True)
os.makedirs(dest_test, exist_ok=True)


# Function to copy files randomly to the destination folders
def copy_files(source_path, dest_train, dest_val, dest_test, train_ratio, val_ratio, category):
    filenames = os.listdir(source_path)
    random.shuffle(filenames)
    total_files = 80000  # Take only 80,000 photos

    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)

    train_files = filenames[:train_count]
    val_files = filenames[train_count:train_count + val_count]
    test_files = filenames[train_count + val_count:total_files]

    for file in train_files:
        src = os.path.join(source_path, file)
        dest = os.path.join(dest_train, f"{category}{file}")
        shutil.copy(src, dest)

    for file in val_files:
        src = os.path.join(source_path, file)
        dest = os.path.join(dest_val, f"{category}{file}")
        shutil.copy(src, dest)

    for file in test_files:
        src = os.path.join(source_path, file)
        dest = os.path.join(dest_test, f"{category}{file}")
        shutil.copy(src, dest)

# Copy "smiling" images
smiling_source = os.path.join(source_folder, "Male")
copy_files(smiling_source, dest_train, dest_val, dest_test, 0.72, 0.08, "Male")

# Copy "unsmiling" images
unsmiling_source = os.path.join(source_folder, "Female")
copy_files(unsmiling_source, dest_train, dest_val, dest_test, 0.72, 0.08, "Female")
