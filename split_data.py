import os
import shutil
import random
# --- CONFIGURATION ---
SOURCE_DIR = '.\Documents\GitHub\AutoSort\source_data'
DATASET_DIR = '.\Documents\GitHub\AutoSort\dataset'
SPLIT_SIZE = 0.8  # 80% Training, 20% Validation

def create_dirs():
    # Create the training and validation directories
    for split in ['train', 'val']:
        for category in ['trash', 'recycle']:
            dir_path = os.path.join(DATASET_DIR, split, category)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")

def split_data():
    # Check if source exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: '{SOURCE_DIR}' not found. Please create it and add 'trash' and 'recycle' folders inside.")
        return

    # Process each class
    for category in ['trash', 'recycle']:
        source_path = os.path.join(SOURCE_DIR, category)
        train_dest = os.path.join(DATASET_DIR, 'train', category)
        val_dest = os.path.join(DATASET_DIR, 'val', category)

        # Get all files
        files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
        
        # Shuffle to ensure random split
        random.shuffle(files)

        # Calculate split index
        split_point = int(len(files) * SPLIT_SIZE)
        train_files = files[:split_point]
        val_files = files[split_point:]

        print(f"Processing {category}: {len(train_files)} training, {len(val_files)} validation")

        # Copy files (Using copy instead of move so you keep your originals safe)
        for f in train_files:
            shutil.copyfile(os.path.join(source_path, f), os.path.join(train_dest, f))
            
        for f in val_files:
            shutil.copyfile(os.path.join(source_path, f), os.path.join(val_dest, f))

if __name__ == "__main__":
    # Clean previous dataset if needed
    if os.path.exists(DATASET_DIR):
        user_input = input(f"'{DATASET_DIR}' already exists. Delete and rebuild? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(DATASET_DIR)
        else:
            print("Aborting.")
            exit()

    create_dirs()
    split_data()
    print("Data split complete.")