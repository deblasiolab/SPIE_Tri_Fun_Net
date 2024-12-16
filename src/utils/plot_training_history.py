import matplotlib.pyplot as plt

def plot_training_history(history, save_path=None):
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'

    epochs = range(1, len(history['loss']) + 1)
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training History', fontsize=16, fontweight='bold', y=1.05)

    # Plot 1: Combined Losses
    ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#2ecc71')
    ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Combined Loss Over Epochs', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7, which='both')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.tick_params(labelsize=10)

    # Plot 2: Accuracy
    ax2.plot(epochs, train_accuracy, label='Training Accuracy', linewidth=2, color='#2ecc71')
    ax2.plot(epochs, val_accuracy, label='Validation Accuracy', linewidth=2, color='#e74c3c')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.tick_params(labelsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.show()
    plt.clf()