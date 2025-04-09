import os
import shutil
import random

# Original preprocessed folder
SOURCE_DIR = '../data/Preprocessed_data_main2'  # Change path if needed
DEST_DIR = '../data/Preprocessed_split2'

TRAIN_DIR = os.path.join(DEST_DIR, 'train_subjects')
TEST_DIR = os.path.join(DEST_DIR, 'test_subjects')

TRAIN_RATIO = 0.8  # 80% train, 20% test

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# List all subject folders
subject_folders = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
random.shuffle(subject_folders)

# Split
split_index = int(len(subject_folders) * TRAIN_RATIO)
train_subjects = subject_folders[:split_index]
test_subjects = subject_folders[split_index:]

# Move folders
for subj in train_subjects:
    shutil.copytree(os.path.join(SOURCE_DIR, subj), os.path.join(TRAIN_DIR, subj))

for subj in test_subjects:
    shutil.copytree(os.path.join(SOURCE_DIR, subj), os.path.join(TEST_DIR, subj))

print(f" Done. {len(train_subjects)} subjects in train, {len(test_subjects)} in test.")
