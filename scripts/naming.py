import os

def rename_npy_files(root_dir):
    renamed_count = 0
    for subject_folder in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        
        for filename in os.listdir(subject_path):
            if filename.endswith('_preprocessed.npy'):
                old_path = os.path.join(subject_path, filename)
                new_filename = filename.replace('_preprocessed', '')
                new_path = os.path.join(subject_path, new_filename)

                if not os.path.exists(new_path):  # Avoid overwriting
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    renamed_count += 1
                else:
                    print(f"Skipped (already exists): {new_filename}")
    
    print(f"\nTotal files renamed: {renamed_count}")

if __name__ == "__main__":
    preprocessed_root = "../data/Preprocessed_data1"  # adjust if needed
    rename_npy_files(preprocessed_root)
