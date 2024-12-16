import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_image_dimensions(x_train, save_path=None):
    # Calculate mean image
    mean_img = np.mean(x_train, axis=0)
    height, width = mean_img.shape

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Plot mean image with dimensions
    ax1 = plt.subplot(gs[0])
    im = ax1.imshow(mean_img, cmap='jet', aspect='auto')

    # Add colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    # Add dimension annotations
    ax1.annotate('', xy=(width+50, 0), xytext=(width+50, height),
                arrowprops=dict(arrowstyle='<->'))
    ax1.text(width+100, height/2, f'Height\n{height} pixels',
            verticalalignment='center')
    ax1.annotate('', xy=(0, height+20), xytext=(width, height+20),
                arrowprops=dict(arrowstyle='<->'))
    ax1.text(width/2, height+50, f'Width: {width} pixels',
            horizontalalignment='center')

    # Add title
    ax1.set_title(f'Mean Image Dimensions: {height} × {width} pixels', pad=40)

    # Create statistics table
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    aspect_ratio = width/height

    # Calculate statistics for mean image
    mean_val = np.mean(mean_img)
    std_val = np.std(mean_img)
    min_val = np.min(mean_img)
    max_val = np.max(mean_img)
    median_val = np.median(mean_img)

    # Create info text with both dimensions and statistics
    info_text = (
        f"Dimension Details:\n"
        f"• Height: {height} pixels\n"
        f"• Width: {width} pixels\n"
        f"• Aspect Ratio: {aspect_ratio:.2f}:1\n"
        f"• Total Pixels: {height*width:,}\n"
        f"• Data Type: {x_train.dtype}\n\n"
        f"Mean Image Statistics:\n"
        f"• Mean: {mean_val:.3f}\n"
        f"• Std Dev: {std_val:.3f}\n"
        f"• Min: {min_val:.3f}\n"
        f"• Max: {max_val:.3f}\n"
        f"• Median: {median_val:.3f}\n"
    )

    ax2.text(0.5, 0.5, info_text,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8),
             fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.clf()