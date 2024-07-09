import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE



def find_closest_vector(A, B, metric='euclidean'):
    """
    Calculate pairwise distances between vectors in A and B,
    and find the indices and distances of the closest vectors in B for each vector in A.

    Parameters:
    - A: numpy array, shape (m, n)
    - B: numpy array, shape (p, n)
    - metric: str, optional, distance metric for cdist function (default: 'euclidean')

    Returns:
    - closest_indices: numpy array, shape (m,), indices of closest vectors in B for each vector in A
    - closest_distances: numpy array, shape (m,), distances to the closest vectors in B for each vector in A
    """
    A = np.squeeze(np.array(A))
    B = np.squeeze(np.array(B))
    distance_matrix = cdist(A, B, metric=metric)
    closest_indices = np.argmin(distance_matrix, axis=1)
    closest_distances = np.min(distance_matrix, axis=1)
    return closest_indices, closest_distances

def find_top_k_closest_vectors(A, B, k=1, metric='euclidean'):
    """
    Calculate pairwise distances between vectors in A and B,
    and find the indices and distances of the top K closest vectors in B for each vector in A.

    Parameters:
    - A: numpy array, shape (m, n)
    - B: numpy array, shape (p, n)
    - k: int, optional, number of closest vectors to retrieve (default: 1)
    - metric: str, optional, distance metric for cdist function (default: 'euclidean')

    Returns:
    - top_k_indices: numpy array, shape (m, k), indices of top K closest vectors in B for each vector in A
    - top_k_distances: numpy array, shape (m, k), distances to the top K closest vectors in B for each vector in A
    """
    A = np.squeeze(np.array(A))
    B = np.squeeze(np.array(B))
    
    distance_matrix = cdist(A, B, metric=metric)
    
    # Use np.argsort to get the indices of the K smallest distances for each row
    top_k_indices = np.argsort(distance_matrix, axis=1)[:, :k]
        
    return top_k_indices, 0

def jaccard_similarity(label1, label2):
    """
    Calculates the Jaccard similarity index between two one-hot encoded labels.

    Args:
    label1: A numpy array representing the first one-hot encoded label.
    label2: A numpy array representing the second one-hot encoded label.

    Returns:
    The Jaccard similarity index between the two labels.
    """

    intersection = np.sum(np.logical_and(label1, label2))
    union = np.sum(np.logical_or(label1, label2))

    if union == 0:
        return 1.0
    else:
        return float(intersection) / union
    
# def find_closest_vector(x, array_x, distance_metric='cosine'):
#     """
#     Finds the index of the single closest neighbor of a normalized feature vector x
#     in a list of normalized feature vectors array_x, using the specified distance metric.

#     Args:
#         x: The normalized feature vector.
#         array_x: A list of normalized feature vectors.
#         distance_metric: The distance metric to use, either 'cosine' or 'euclidean'.

#     Returns:
#         The index of the closest neighbor in array_x.
#     """
    

#     array_x = np.squeeze(np.array(array_x))
#     x = np.repeat(x, array_x.shape[0], axis=0)
#     distances = cdist(x, array_x, metric=distance_metric)
#     closest_neighbor_index = np.argmin(distances, axis=1)
#     return closest_neighbor_index[0]  # Return the index for the first row

def gentsnePlot(feature_vectors, vectors_label, centroids, figname):
    data_for_tsne = np.concatenate([feature_vectors, centroids], axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data_for_tsne)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:-centroids.shape[0], 0], X_tsne[:-centroids.shape[0], 1],
            c=vectors_label, cmap='viridis', marker='o', s=50, alpha=0.8, label='Data Points')

    # Plotting K-Means centroids
    plt.scatter(X_tsne[-centroids.shape[0]:, 0], X_tsne[-centroids.shape[0]:, 1],
            c='red', marker='X', s=200, label='Centroids')
    plt.title('t-SNE Visualization with K-Means Centroids')
    plt.legend()
    plt.savefig(figname)