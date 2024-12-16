import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

def cluster_and_evaluate(encoder, x_data, y_true, n_clusters):
    # Get latent representations
    latent_vectors = encoder.predict(x_data, verbose=0)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    
    # Calculate clustering accuracy using Hungarian algorithm
    def cluster_acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
            
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
    
    # Calculate metrics
    clustering_accuracy = cluster_acc(y_true, cluster_labels)
    ari_score = adjusted_rand_score(y_true, cluster_labels)
    
    return {
        'cluster_labels': cluster_labels,
        'clustering_accuracy': clustering_accuracy,
        'ari_score': ari_score,
        'kmeans_model': kmeans,
        'latent_vectors': latent_vectors
    }