import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from collections import Counter
from src.constants import SLIT_IMGS_PATH, SLIT_LABL_PATH, FOLDS_FOLDER_PATH, PLOTS_FOLDER_PATH, TOTAL_FOLDS, SEED, VAL_SPLIT_RATIO

# Ensure the necessary folders exist
os.makedirs(FOLDS_FOLDER_PATH, exist_ok=True)
os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

# Load data
img_features = np.load(SLIT_IMGS_PATH).astype(np.float32)
img_labels = np.load(SLIT_LABL_PATH)

# Print dataset summary
print(f"""
===============================================
                 Data Loaded
===============================================
- Features (X): {img_features.shape}
- Labels   (Y): {img_labels.shape}
- Unique Labels: {np.unique(img_labels)}
===============================================
""")

# Function to calculate percentages
def calculate_percentages(label_counts, total_count):
    return {label: (count / total_count) * 100 for label, count in label_counts.items()}

# Function to plot and save label distributions
def plot_fold_distribution(train_labels, val_labels, test_labels, fold_idx, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_labels = sorted(set(train_labels) | set(val_labels) | set(test_labels))
    
    # Count labels
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)
    
    # Calculate percentages
    train_percentages = calculate_percentages(train_counts, sum(train_counts.values()))
    val_percentages = calculate_percentages(val_counts, sum(val_counts.values()))
    test_percentages = calculate_percentages(test_counts, sum(test_counts.values()))
    
    # Create bar positions
    bar_width = 0.25
    x = np.arange(len(all_labels))
    
    # Plot bars
    ax.bar(x - bar_width, [train_percentages.get(label, 0) for label in all_labels], 
           width=bar_width, label=f'Training ({sum(train_counts.values())} samples)', alpha=0.8)
    ax.bar(x, [val_percentages.get(label, 0) for label in all_labels], 
           width=bar_width, label=f'Validation ({sum(val_counts.values())} samples)', alpha=0.8)
    ax.bar(x + bar_width, [test_percentages.get(label, 0) for label in all_labels], 
           width=bar_width, label=f'Testing ({sum(test_counts.values())} samples)', alpha=0.8)
    
    # Customize plot
    ax.set_title(f'Label Distribution for Fold {fold_idx}', fontsize=18, weight='bold')
    ax.set_xlabel('Class', fontsize=16, weight='bold')
    ax.set_ylabel('Percentage', fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plot_path = f'{save_path}/fold_{fold_idx}_distribution.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot for fold {fold_idx} at {plot_path}")

# Generate 10 folds using StratifiedKFold
skf = StratifiedKFold(n_splits=TOTAL_FOLDS, shuffle=True, random_state=SEED)
print(f"Generating {TOTAL_FOLDS} stratified folds with train-validation splits...")

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(img_features, img_labels)):
    # Create train-validation split from the training indices
    train_data_idx, val_data_idx = train_test_split(
        train_idx,
        test_size=VAL_SPLIT_RATIO,
        stratify=img_labels[train_idx],
        random_state=SEED
    )
    
    # Save indices for this fold
    np.save(f'{FOLDS_FOLDER_PATH}/train_indices_fold_{fold_idx}.npy', train_data_idx)
    np.save(f'{FOLDS_FOLDER_PATH}/val_indices_fold_{fold_idx}.npy', val_data_idx)
    np.save(f'{FOLDS_FOLDER_PATH}/test_indices_fold_{fold_idx}.npy', test_idx)
    
    # Print fold shapes
    print(f"""
===============================================
              Fold {fold_idx}
===============================================
- Training Set : Features {img_features[train_data_idx].shape}, Labels {img_labels[train_data_idx].shape}
- Validation Set: Features {img_features[val_data_idx].shape}, Labels {img_labels[val_data_idx].shape}
- Testing Set  : Features {img_features[test_idx].shape}, Labels {img_labels[test_idx].shape}
===============================================
""")
    
    # Plot and save distribution for this fold
    plot_fold_distribution(
        train_labels=img_labels[train_data_idx],
        val_labels=img_labels[val_data_idx],
        test_labels=img_labels[test_idx],
        fold_idx=fold_idx,
        save_path=PLOTS_FOLDER_PATH
    )

print(f"All {TOTAL_FOLDS} folds with train-validation-test splits generated and saved in '{FOLDS_FOLDER_PATH}'.")
print(f"Plots of label distributions saved in '{PLOTS_FOLDER_PATH}'.")
