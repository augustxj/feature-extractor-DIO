import pickle
import os
from src.utils.config import FILENAMES_FILE, IMAGES_PATH

# Load the filenames list
with open(FILENAMES_FILE, 'rb') as f:
    filenames_list = pickle.load(f)

# Update the paths in filenames_list
updated_filenames_list = [os.path.join(IMAGES_PATH, os.path.basename(filename)) for filename in filenames_list]

# Save the updated filenames list
with open(FILENAMES_FILE, 'wb') as f:
    pickle.dump(updated_filenames_list, f)

print("Filenames list updated successfully.")