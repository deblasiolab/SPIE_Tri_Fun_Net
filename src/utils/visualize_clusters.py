import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_clusters(encoder, x_data, cluster_labels, true_labels, label_encoder, save_path=None):
    # Get latent representations
    latent_vectors = encoder.predict(x_data, verbose=0)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    latent_reduced = tsne.fit_transform(latent_vectors)
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Clustering results
    scatter1 = ax1.scatter(latent_reduced[:, 0], latent_reduced[:, 1], 
                          c=cluster_labels, cmap='Set1', alpha=0.7)
    ax1.set_title('Clustering Results')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    
    # Plot 2: True labels
    scatter2 = ax2.scatter(latent_reduced[:, 0], latent_reduced[:, 1], 
                          c=true_labels, cmap='Set1', alpha=0.7)
    ax2.set_title('True Labels')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    
    # Add legends - Fixed version
    handles1, labels1 = scatter1.legend_elements()
    handles2, labels2 = scatter2.legend_elements()
    
    ax1.legend(handles1, [f'Cluster {i}' for i in range(len(handles1))], title='Clusters')
    ax2.legend(handles2, 
              [label_encoder.inverse_transform([i])[0] for i in range(len(handles2))],
              title='True Labels')
    
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()