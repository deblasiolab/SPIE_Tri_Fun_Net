import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def analyze_data_splits(x_train, y_train, x_val, y_val, x_test, y_test, save_path=None):
    total_samples = len(y_train) + len(y_val) + len(y_test)
    train_pct = len(y_train) / total_samples * 100
    val_pct = len(y_val) / total_samples * 100
    test_pct = len(y_test) / total_samples * 100

    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))

    train_dist = {label: np.sum(y_train == label) for label in unique_labels}
    val_dist = {label: np.sum(y_val == label) for label in unique_labels}
    test_dist = {label: np.sum(y_test == label) for label in unique_labels}

    train_means = {label: np.mean(x_train[y_train == label]) for label in unique_labels}
    val_means = {label: np.mean(x_val[y_val == label]) for label in unique_labels}
    test_means = {label: np.mean(x_test[y_test == label]) for label in unique_labels}

    # Create visualization
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    ax1 = plt.subplot(gs[0])
    bar_width = 0.25
    index = np.arange(len(unique_labels))

    # Calculate percentages for each split
    train_pcts = [train_dist[label]/len(y_train)*100 for label in unique_labels]
    val_pcts = [val_dist[label]/len(y_val)*100 for label in unique_labels]
    test_pcts = [test_dist[label]/len(y_test)*100 for label in unique_labels]

    ax1.bar(index - bar_width, train_pcts, bar_width, label=f'Train ({train_pct:.1f}%)', color='#2ecc71')
    ax1.bar(index, val_pcts, bar_width, label=f'Validation ({val_pct:.1f}%)', color='#e74c3c')
    ax1.bar(index + bar_width, test_pcts, bar_width, label=f'Test ({test_pct:.1f}%)', color='#3498db')

    ax1.set_ylabel('Percentage of Samples (%)')
    ax1.set_title('Label Distribution Across Splits')
    ax1.set_xticks(index)
    ax1.set_xticklabels(unique_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')

    # Prepare sample count table
    columns_samples = ['Split', 'Total Samples'] + [str(label) for label in unique_labels]
    train_row = ['Train', len(y_train)] + [train_dist[label] for label in unique_labels]
    val_row = ['Validation', len(y_val)] + [val_dist[label] for label in unique_labels]
    test_row = ['Test', len(y_test)] + [test_dist[label] for label in unique_labels]
    table_data_samples = [train_row, val_row, test_row]

    table1 = ax2.table(cellText=table_data_samples,
                      colLabels=columns_samples,
                      loc='center',
                      cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1.2, 1.5)
    ax2.set_title("Number of Samples per Split", pad=20)

    # Create mean pixel values table
    ax3 = plt.subplot(gs[2])
    ax3.axis('off')
    columns_means = ['Split'] + [str(label) for label in unique_labels]
    train_means_row = ['Train'] + [f'{train_means[label]:.3f}' for label in unique_labels]
    val_means_row = ['Validation'] + [f'{val_means[label]:.3f}' for label in unique_labels]
    test_means_row = ['Test'] + [f'{test_means[label]:.3f}' for label in unique_labels]
    table_data_means = [train_means_row, val_means_row, test_means_row]

    table2 = ax3.table(cellText=table_data_means,
                      colLabels=columns_means,
                      loc='center',
                      cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.5)
    ax3.set_title('Mean Pixel Values per Label', pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.clf()