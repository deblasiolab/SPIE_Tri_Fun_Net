import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_class_samples(x_train, y_train, selected_indices_arr=None, save_path=None):
    unique_labels = np.unique(y_train)
    n_classes = len(unique_labels)
    n_figs = math.ceil(n_classes / 3)

    all_selected_indices = []

    for fig_num in range(n_figs):
        start_idx = fig_num * 3
        end_idx = min((fig_num + 1) * 3, n_classes)
        current_labels = unique_labels[start_idx:end_idx]

        current_selected_indices = []

        fig = plt.figure(figsize=(20, 4*len(current_labels)))
        gs = gridspec.GridSpec(len(current_labels), 4, figure=fig)

        for idx, label in enumerate(current_labels):
            label_indices = np.where(y_train == label)[0]
            if selected_indices_arr is None:
                selected_indices = np.random.choice(label_indices, size=2, replace=False)
            else:
                selected_indices = selected_indices_arr[start_idx + idx]

            current_selected_indices.append(selected_indices)

            for i, sample_idx in enumerate(selected_indices):
                img = x_train[sample_idx]

                # Plot image
                ax_img = fig.add_subplot(gs[idx, i*2])
                im = ax_img.imshow(img, cmap='jet', aspect='auto')
                ax_img.set_title(f'{label} - Sample {i+1}')
                ax_img.axis('off')

                # Add colorbar
                divider = make_axes_locatable(ax_img)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax)

                # Plot histogram
                ax_hist = fig.add_subplot(gs[idx, i*2 + 1])
                ax_hist.hist(img.flatten(), bins=50, color='blue', alpha=0.7)
                ax_hist.set_title(f'Pixel Distribution\nμ={np.mean(img):.2f}, σ={np.std(img):.2f}')
                ax_hist.set_xlabel('Pixel Value')
                ax_hist.set_ylabel('Frequency')

                # Add statistics text box
                stats_text = f'Min: {np.min(img):.2f}\nMax: {np.max(img):.2f}\n'
                stats_text += f'Median: {np.median(img):.2f}'
                ax_hist.text(0.95, 0.95, stats_text,
                            transform=ax_hist.transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            base_path = save_path.rsplit('.', 1)[0]
            ext = save_path.rsplit('.', 1)[1]
            current_save_path = f"{base_path}_part{fig_num+1}.{ext}"
            plt.savefig(current_save_path, dpi=400, bbox_inches='tight')
        plt.show()
        plt.clf()

        all_selected_indices.extend(current_selected_indices)

    return all_selected_indices if selected_indices_arr is None else None