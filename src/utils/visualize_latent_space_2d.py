import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_latent_space_2d(encoder, x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, save_path=None):
    # Get latent representations
    latent_train = encoder.predict(x_train, verbose=0)
    latent_val = encoder.predict(x_val, verbose=0)
    latent_test = encoder.predict(x_test, verbose=0)

    # Combine latent vectors
    latent_combined = np.vstack([latent_train, latent_val, latent_test])

    # Optimized t-SNE configuration
    tsne = TSNE(
        n_components=2,
        perplexity=5,
        random_state=42
    )

    # Perform t-SNE reduction
    latent_reduced = tsne.fit_transform(latent_combined)

    # Split reduced dimensions back into train, val, test
    n_train = len(x_train)
    n_val = len(x_val)
    train_reduced = latent_reduced[:n_train]
    val_reduced = latent_reduced[n_train:n_train + n_val]
    test_reduced = latent_reduced[n_train + n_val:]

    # Create figure
    plt.figure(figsize=(15, 10))

    # Get unique labels
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    # Plot points for each set
    for idx, label in enumerate(unique_labels):
        # Training points
        mask_train = y_train == label
        plt.scatter(train_reduced[mask_train, 0],
                   train_reduced[mask_train, 1],
                   c=[colors[idx]],
                   marker='o',
                   label=f'{label_encoder.inverse_transform([label])[0]} (Train)',
                   alpha=0.7,
                   s=100)

        # Validation points
        mask_val = y_val == label
        plt.scatter(val_reduced[mask_val, 0],
                   val_reduced[mask_val, 1],
                   c=[colors[idx]],
                   marker='s',
                   label=f'{label_encoder.inverse_transform([label])[0]} (Val)',
                   alpha=0.7,
                   s=100)

        # Test points
        mask_test = y_test == label
        plt.scatter(test_reduced[mask_test, 0],
                   test_reduced[mask_test, 1],
                   c=[colors[idx]],
                   marker='^',
                   label=f'{label_encoder.inverse_transform([label])[0]} (Test)',
                   alpha=0.7,
                   s=100)

    plt.title('2D Latent Space Representation (t-SNE)', fontsize=16, pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)

    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.,
              fontsize=10)

    # Add grid
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()