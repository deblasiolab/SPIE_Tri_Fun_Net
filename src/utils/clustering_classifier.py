import numpy as np
from scipy.optimize import linear_sum_assignment

class ClusteringClassifier:
    def __init__(self, encoder, kmeans):
        self.encoder = encoder
        self.kmeans = kmeans
        self.label_mapping = None
    
    def fit(self, X, y):
        # Get cluster assignments
        latent = self.encoder.predict(X, verbose=0)
        cluster_labels = self.kmeans.predict(latent)
        
        # Create confusion matrix between cluster labels and true labels
        n_classes = len(np.unique(y))
        w = np.zeros((n_classes, n_classes), dtype=np.int64)
        
        for i in range(len(y)):
            w[cluster_labels[i], y[i]] += 1
        
        # Use Hungarian algorithm to find optimal mapping
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        self.label_mapping = dict(zip(row_ind, col_ind))
        
    def predict(self, x):
        latent = self.encoder.predict(x, verbose=0)
        cluster_labels = self.kmeans.predict(latent)
        # Map cluster labels to true labels
        return np.array([self.label_mapping[label] for label in cluster_labels])