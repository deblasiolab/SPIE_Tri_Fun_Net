import os
import numpy as np
from astropy.io import fits
from typing import Tuple, List
from collections import Counter
from src.constants import DATA_PATH, SAVE_PATH

def load_fits_data(data_path):
    # Initialize lists for data and labels
    x_fits_files = []
    y_labels = []
    
    # Get class folders and process FITS files
    class_folders = [f for f in os.listdir(data_path) 
                    if os.path.isdir(os.path.join(data_path, f))]
    
    for class_folder in sorted(class_folders):
        folder_path = os.path.join(data_path, class_folder)
        fits_files = [f for f in os.listdir(folder_path) 
                     if f.endswith('.fits')]
        
        for fits_file in fits_files:
            file_path = os.path.join(folder_path, fits_file)
            try:
                with fits.open(file_path) as hdul:
                    data = hdul[0].data.astype(np.float32)
                x_fits_files.append(data)
                y_labels.append(class_folder)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
    
    return np.array(x_fits_files, dtype=np.float32), np.array(y_labels)

def print_dataset_stats(x, y):
    # Basic dataset information
    print(f"Dataset shape: {x.shape}")
    print(f"Data type: {x.dtype}")
    print(f"Total samples: {len(x)}")
    
    # Class distribution
    class_dist = Counter(y)
    print("\nSamples per class:")
    for label, count in sorted(class_dist.items()):
        print(f"{label}: {count}")
    
    # Statistics per class
    print("\nMean values per class:")
    for label in np.unique(y):
        class_mask = y == label
        class_mean = x[class_mask].mean()
        class_std = x[class_mask].std()
        print(f"{label}: mean={class_mean:.4f}, std={class_std:.4f}")

if __name__ == "__main__":
    # Load the data
    x, y = load_fits_data(DATA_PATH)
    print_dataset_stats(x, y)

    # Save the data
    np.save(os.path.join(SAVE_PATH, 'x_raw_slitless_spectro_single_channel_images.npy'), x)
    np.save(os.path.join(SAVE_PATH, 'y_satellite_string_class_labels.npy'), y)
